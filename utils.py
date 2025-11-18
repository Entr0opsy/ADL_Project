import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        distance_negative = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

class ArcFaceLoss(nn.Module):
    """
    A mathematically robust implementation of ArcFace Loss.
    This version includes:
    1. L2 normalization of both features and weights.
    2. A safe clamping mechanism for acos to prevent NaN gradients.
    3. Logic to handle the non-monotonicity of cos(theta + m),
       making the loss function strictly decreasing with respect to theta.
    """
    def __init__(self, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.criterion = nn.CrossEntropyLoss()
        # Threshold to prevent cos(theta+m) from becoming non-monotonic
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels, weight):
        # 1. L2 Normalize embeddings (features) and the classification weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        kernel_norm = F.normalize(weight, p=2, dim=1)

        # 2. Calculate the cosine similarity (dot product of normalized vectors)
        cosine = F.linear(embeddings, kernel_norm)
        
        # 3. Safely compute the angle theta
        eps = 1e-7
        cosine_clamped = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))
        
        # 4. Compute phi(theta) = cos(theta + m) using the angle addition formula
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine_clamped * self.cos_m - sine * self.sin_m
        
        # 5. Handle the non-monotonic region of cos(theta + m)
        # This part is crucial for training stability.
        # If theta + m > 180 degrees, use a hard margin penalty instead.
        # This is equivalent to checking if cos(theta) < cos(pi - m)
        condition = (cosine_clamped > self.th).float()
        phi = condition * phi + (1.0 - condition) * (cosine_clamped - self.mm)

        # 6. Create the one-hot encoding for the labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 7. Create the final logits by substituting phi for the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 8. Scale the logits
        output *= self.s
        
        # 9. Compute the final Cross-Entropy loss
        loss = self.criterion(output, labels)
        return loss

def save_model(model, path):
    """Save model state dict"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def plot_training_curves(train_losses, val_losses, train_accuracies=None, val_accuracies=None, 
                         save_dir=None, filename=None):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    if isinstance(train_losses, dict):
        for key in train_losses:
            plt.plot(train_losses[key], label=f'Train {key}')
            plt.plot(val_losses[key], label=f'Val {key}')
    else:
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies if provided
    if train_accuracies is not None and val_accuracies is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir and filename:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
