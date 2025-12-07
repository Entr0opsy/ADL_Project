from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class IrisDatasetWithPaths(Dataset):
    def __init__(self, root_dir, transform=None, include_augmented=False):
        """
        Custom Dataset to load images from the structure:
        {root_dir}/{ID}/*.jpg (and similar extensions).
        
        Args:
            root_dir (str): The path to the 'test' or 'train' folder.
            transform (callable, optional): Transform to be applied on the image.
            include_augmented (bool): Kept for compatibility.
        """
        self.root_dir = root_dir
        self.transform = transform 
        self.include_augmented = include_augmented

        self.samples = []  # Stores (path, label)
        self.class_to_idx = {}

        # The Identity folders (e.g., '001', '002', ...) are directly under self.root_dir
        class_folders = sorted(os.listdir(self.root_dir))
        
        for idx, cls in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path):
                continue
            
            # Use the Identity folder name (e.g., '001') as the class label
            self.class_to_idx[cls] = idx

            # The structure is: {root_dir}/{ID}/*.jpg
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, fname)
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Return image, label, and path (required for evaluation script)
        return image, label, path
def get_identity_id(full_path):
    """Extracts the folder name immediately preceding the image file."""
    # 1. Get the directory of the file (should be the ID folder)
    id_dir = os.path.dirname(full_path)
    # 2. Get the name of that directory (which is the ID)
    identity_id = os.path.basename(id_dir)
    # 3. Strip any potential leading/trailing whitespace for robust comparison
    return identity_id.strip()


