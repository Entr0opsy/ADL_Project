import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np

# The key libraries for triplet loss
from pytorch_metric_learning import losses, miners, samplers


class IrisDataset(Dataset):
    def __init__(self, root_dir, processor):
        """
        Args:
            root_dir (str): Path to the main 'train' directory.
            processor (CLIPProcessor): The CLIP processor for transforms.
        """
        self.processor = processor
        self.samples = []
        self.label_to_int = {}
        self.labels_list = [] # For the sampler
        current_label_id = 0

        root_path = Path(root_dir)
        print(f"Scanning directory: {root_path}")

        # Loop through each class folder (e.g., '001', '002')
        for class_dir in sorted(root_path.glob('*')):
            if not class_dir.is_dir():
                continue

            label_name = class_dir.name
            image_dir = class_dir / 'image'

            if not image_dir.exists():
                print(f"Warning: Missing 'images' folder in {class_dir}")
                continue

            # Assign a unique integer ID to this class
            if label_name not in self.label_to_int:
                self.label_to_int[label_name] = current_label_id
                current_label_id += 1

            label_id = self.label_to_int[label_name]

            # Find all images and add them to our samples list
            for img_path in sorted(image_dir.glob('*.jpg')):
                self.samples.append((str(img_path), label_id))
                self.labels_list.append(label_id)

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root_dir}. Check the path and structure.")

        print(f"Found {len(self.samples)} images belonging to {len(self.label_to_int)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image, ensure it's RGB for CLIP
        image = Image.open(img_path).convert("RGB")

        # Use the CLIP processor to resize, crop, and normalize
        processed_image = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Squeeze to remove the extra batch dimension
        return processed_image.pixel_values.squeeze(0), label

    def get_labels(self):
        """Helper method for the Pytorch Metric Learning sampler."""
        return self.labels_list
    
class ClipIrisModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        # Load the full pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)

        # We only want to train the vision parts
        self.vision_model = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection

        # Freeze the text model parameters
        for param in self.clip.text_model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        # Pass images through the vision backbone
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # Get the [CLS] token output
        pooled_output = vision_outputs.pooler_output

        # Project it into the final embedding space
        embedding = self.visual_projection(pooled_output)

        return embedding

#@title Param for CLIP+TRIPLET
# --- Configuration ---
TRAIN_DIR = 'train'  # The path to your 'train' folder
MODEL_NAME = "openai/clip-vit-base-patch32" #@param
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Batching Strategy (P*K) ---
# We will create batches by sampling P classes, then K images from each class.
P = 8 #@param
K = 4 #@param
BATCH_SIZE = P * K # Total batch size (32)
# Adjust P and K based on your dataset size and GPU VRAM

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-6  # Start very low for finetuning
NUM_EPOCHS = 5
MARGIN = 0.4  #@param The margin for the triplet loss

print(f"Using device: {DEVICE}")

# 1. Load Model and Processor
print("Loading model and processor...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = ClipIrisModel(MODEL_NAME).to(DEVICE)

# 2. Load Dataset
print("Loading dataset...")
dataset = IrisDataset(root_dir=TRAIN_DIR, processor=processor)

# 3. Define the Sampler
# This is crucial. MPerClassSampler creates batches using the P*K strategy,
# which is essential for giving the triplet miner good data to work with.
sampler = samplers.MPerClassSampler(
    dataset.get_labels(),  # Get list of all labels
    m=K,                     # K images per class
    length_before_new_iter=len(dataset) # How many samples to draw per epoch
)

# 4. Define the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2 # Speeds up data loading
)

# 5. Define Loss, Miner, and Optimizer (as requested)
loss_func = losses.TripletMarginLoss(margin=MARGIN)

# This is the key: a miner that finds semi-hard triplets within each batch
miner = miners.TripletMarginMiner(
    margin=MARGIN,
    type_of_triplets="semihard" # This is the "semi-hard" part
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

print("Setup complete. Ready for training.")

print("Starting training...")
model.train()  # Set the model to training mode

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    total_loss = 0

    for batch in tqdm(dataloader):
        images, labels = batch

        # Move data to the GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        embeddings = model(images)
        indices_tuple = miner(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Complete. Average Triplet Loss: {avg_loss:.4f}")

print("\n--- Training finished! ---")