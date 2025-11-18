import os
import random
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class IrisDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, include_augmented=True):
        """
        root_dir = ".../masked_dataset"
        split = "train" or "test"
        include_augmented = True -> load images from output/ as well
        """
        self.root_dir = os.path.join(root_dir, split)
        self.include_augmented = include_augmented
        self.transform = transform or transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.samples = []
        self.class_to_idx = {}

        class_folders = sorted(os.listdir(self.root_dir))
        for idx, cls in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[cls] = idx

            # 1️⃣ add original images
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, fname)
                    self.samples.append((img_path, idx))

            # 2️⃣ add augmented images from output/ folder
            if self.include_augmented:
                output_path = os.path.join(class_path, "output")
                if os.path.exists(output_path):
                    for fname in os.listdir(output_path):
                        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(output_path, fname)
                            self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TripletIrisDataset(Dataset):
    """Triplet dataset: returns (anchor, positive, negative)"""
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.dataset = IrisDataset(root_dir, split, transform)
        self.class_to_indices = self._build_class_indices()

    def _build_class_indices(self) -> Dict[int, List[int]]:
        mapping = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            mapping.setdefault(label, []).append(idx)
        return mapping

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_img, anchor_label = self.dataset[idx]

        # 1. Positive sample: same class
        anchor_idx = idx
        
        # Use the pre-built dictionary to efficiently find all indices of the same class
        positive_candidates = [i for i in self.class_to_indices[anchor_label] if i != anchor_idx]
        
        # Handle the case where there is only one sample of a class (it's its own positive)
        if len(positive_candidates) == 0:
            positive_idx = anchor_idx
        else:
            positive_idx = random.choice(positive_candidates)

        positive_img, _ = self.dataset[positive_idx]

        # 2. Negative sample: different class
        # Choose a random class that is NOT the anchor class
        negative_class = random.choice([cls for cls in self.class_to_indices.keys() if cls != anchor_label])
        
        # Choose a random index from the chosen negative class
        negative_idx = random.choice(self.class_to_indices[negative_class])

        negative_img, _ = self.dataset[negative_idx]

        return anchor_img, positive_img, negative_img


def get_iris_loaders(root_dir: str,
                     loader_type: str = 'all',
                     batch_size: int = 32,
                     num_workers: int = 4):
    """Return dataloaders for iris dataset."""
    loaders = {}

    if loader_type in ['arcface', 'all']:
        train_ds = IrisDataset(root_dir, 'train')
        test_ds = IrisDataset(root_dir, 'test')

        loaders['arcface_train'] = DataLoader(train_ds, 
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=num_workers,
                                              pin_memory=True)
        
        loaders['arcface_test'] = DataLoader(test_ds, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=num_workers,
                                             pin_memory=True)

    if loader_type in ['triplet', 'all']:
        triplet_train = TripletIrisDataset(root_dir, 'train')
        triplet_test = TripletIrisDataset(root_dir, 'test')

        loaders['triplet_train'] = DataLoader(triplet_train, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
        loaders['triplet_test'] = DataLoader(triplet_test, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers,
                                             pin_memory=True)
    return loaders


if __name__ == '__main__':
    root = '/home/aryan/Desktop/Adl_assignment_1/masked_dataset'
    iris_loaders = get_iris_loaders(root)

    print(f"ArcFace Train Samples: {len(iris_loaders['arcface_train'].dataset)}")
    print(f"Triplet Train Samples: {len(iris_loaders['triplet_train'].dataset)}")

    for imgs, labels in iris_loaders['arcface_train']:
        print("ArcFace batch:", imgs.shape, labels.shape)
        break

    for anc, pos, neg in iris_loaders['triplet_train']:
        print("Triplet batch:", anc.shape, pos.shape, neg.shape)
        break





