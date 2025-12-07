import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
from pathlib import Path

from Evaluation.eval_data_loader import IrisDatasetWithPaths, get_identity_id 

# Add paths for custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'edgeface'))

# Import your modified timm wrapper
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from edgeface.backbones import get_timmfrv2


# --- CONFIGURATION (UPDATE THESE FOR EACH MODEL EVALUATION) ---
PRESERVE_VERTICAL = True  # Preserve vertical (radial) resolution


IRIS_ROOT_FOLDER = "../masked_dataset_augmented"
IRIS_TEST_FOLDER = os.path.join(IRIS_ROOT_FOLDER, "test")
IRIS_TRAIN_FOLDER = os.path.join(IRIS_ROOT_FOLDER, "train")










# --- MAIN EVALUATION SCRIPT ---

def evaluate_model(
    model,
    iris_test_folder,
    iris_train_folder,
    output_filename='scores_iris.txt',
    model_name='edgenext_x_small',
    use_asymmetric_conv=True,
    preserve_vertical=True,
    batch_size=32,
    use_cosine_similarity=True
):
   
    
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f" Model loaded and moved to {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 2. Define Preprocessing Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # 3. Load Data and Create DataLoaders
    print("\nLoading datasets...")
    
    ds_test = IrisDatasetWithPaths(
        root_dir=iris_test_folder,
        transform=transform,
        include_augmented=False
    )
    ds_train = IrisDatasetWithPaths(
        root_dir=iris_train_folder,
        transform=transform,
        include_augmented=False
    )
    
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Test samples: {len(ds_test)}")
    print(f"✓ Train samples: {len(ds_train)}")
    
    # 4. Generate Embeddings
    embedding_dict_test = {}
    embedding_dict_train = {}
    
    print("\nGenerating embeddings...")
    
    with torch.no_grad():
        # Test embeddings
        print("  Processing test set...")
        for images, labels, paths in loader_test:
            images = images.to(device)
            features = model(images).to(device)
            
            # Store each sample's embedding
            for i, path in enumerate(paths):
                idx = get_identity_id(path)
                pose = os.path.basename(path).split('.')[0]
                embedding_dict_test[f"{idx}_{pose}"] = features[i:i+1]
        
        # Train embeddings
        print("  Processing train set...")
        for images, labels, paths in loader_train:
            images = images.to(device)
            features = model(images).to(device)
            
            # Store each sample's embedding
            for i, path in enumerate(paths):
                idx = get_identity_id(path)
                pose = os.path.basename(path).split('.')[0]
                embedding_dict_train[f"{idx}_{pose}"] = features[i:i+1]
    
    print(f"\n✓ Test embeddings: {len(embedding_dict_test)}")
    print(f"✓ Train embeddings: {len(embedding_dict_train)}")
    
    # 5. Compare Embeddings and Log Scores
    print(f"\nCalculating similarities and saving to {output_filename}...")
    
    comparison_count = len(embedding_dict_test) * len(embedding_dict_train)
    print(f"Total comparisons: {comparison_count:,}")
    
    similarity_metric = "cosine_similarity" if use_cosine_similarity else "euclidean_distance"
    

    if (output_filename is not None) or (output_filename.strip() != ""):
        with open('LOGS/'+output_filename, "w") as file:
            # Write header with metadata
            file.write(f"# Model: {model_name}\n")
            file.write(f"# Asymmetric Conv: {use_asymmetric_conv}\n")
            file.write(f"# Preserve Vertical: {preserve_vertical}\n")
            file.write(f"# Similarity Metric: {similarity_metric}\n")
            file.write(f"# Test samples: {len(embedding_dict_test)}\n")
            file.write(f"# Train samples: {len(embedding_dict_train)}\n")
            file.write("#\n")
            file.write("idx1\tpose1\tidx2\tpose2\tisGen\tscore\n")
            
            progress_interval = max(1, len(embedding_dict_test) // 20)  # Show 20 progress updates
            
            for progress_idx, (k_test, v_test) in enumerate(embedding_dict_test.items()):
                # k_test is "001_S1001R01"
                idx1, pose1 = k_test.split("_", 1)
                
                for k_train, v_train in embedding_dict_train.items():
                    idx2, pose2 = k_train.split("_", 1)
                    
                    isGen = 1 if idx1 == idx2 else 0
                    
                    # Calculate similarity/distance
                    if use_cosine_similarity:
                        score = F.cosine_similarity(v_test, v_train).item()
                    else:
                        # Negative Euclidean distance (higher = more similar)
                        score = -torch.dist(v_test, v_train, p=2).item()
                    
                    file.write(f"{idx1}\t{pose1}\t{idx2}\t{pose2}\t{isGen}\t{score}\n")
                
                # Progress update
                if (progress_idx + 1) % progress_interval == 0:
                    progress = (progress_idx + 1) / len(embedding_dict_test) * 100
                    print(f"  Progress: {progress:.1f}%")
    

    
    return embedding_dict_test, embedding_dict_train


