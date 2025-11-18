import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# --- Import necessary components from your project ---
# Add EdgeFace repository to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'edgeface'))

# Import model definitions and data loader
from data_loader import get_data_loaders
from train_model_A import ModelA
from train_model_B import ModelB
from train_model_C import ModelC

def load_model(model_name, model_path, num_classes, device):
    """Loads the specified model and its trained weights."""
    model = None
    if model_name == 'A':
        model = ModelA(pretrained=False)
    elif model_name == 'B':
        model = ModelB(pretrained=False, num_classes=num_classes)
    elif model_name == 'C':
        model = ModelC(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please train the model first.")

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_embeddings(model, data_loader, device, model_name):
    """Extracts embeddings and labels for the entire dataset."""
    embeddings_list = []
    labels_list = []
    print("Extracting embeddings from the test set...")
    for images, labels in tqdm(data_loader, desc="Extracting"):
        images = images.to(device)
        
        # Forward pass to get embeddings
        embeds = model(images)
        # If the model returns a tuple (e.g., logits, embeddings), take the last element
        if isinstance(embeds, tuple):
            embeds = embeds[-1]

        embeddings_list.append(embeds.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def plot_embeddings(embeddings, labels, method, model_name, save_dir):
    """
    Reduces embedding dimensionality using t-SNE or PCA and saves a scatter plot.
    """
    print(f"\nPerforming dimensionality reduction using {method.upper()}...")
    
    if method == 'tsne':
        # FIX: Changed 'n_iter' to 'max_iter' for compatibility with newer scikit-learn versions
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=1000, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plotting
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='viridis',  # A colorblind-friendly colormap
        alpha=0.7,
        s=10 # point size
    )
    
    # Create a colorbar
    num_classes = len(np.unique(labels))
    cbar = fig.colorbar(scatter, ticks=np.linspace(0, num_classes-1, min(10, num_classes)))
    cbar.set_label('Class Label')
    
    ax.set_title(f'{method.upper()} Visualization of Embeddings for Model {model_name}')
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.grid(True)
    
    save_path = os.path.join(save_dir, f'{method}_visualization_model_{model_name}.png')
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Plot saved successfully.")

def main(args):
    """Main visualization function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = os.path.join('/home/teaching/assgn1/models', f'model_{args.model}')
    model_path = os.path.join(model_dir, f'model_{args.model}_final.pth')

    # Prepare data loader for test set
    print("Preparing test data loader...")
    # We load a subset of data for faster visualization if specified
    batch_size = 256 # Use a larger batch size for faster embedding extraction
    loaders = get_data_loaders('/home/teaching/assgn1/forehead-v1-labeled', loader_type='arcface', batch_size=batch_size)
    test_loader = loaders['arcface_test']
    num_classes = len(test_loader.dataset.class_to_idx)

    # Load the trained model
    model = load_model(args.model, model_path, num_classes, device)

    # 1. Extract embeddings
    embeddings, labels = extract_embeddings(model, test_loader, device, args.model)
    
    # 2. Generate and save the plot
    plot_embeddings(embeddings, labels, args.method, args.model, model_dir)
    
    print("\nVisualization complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Embeddings from Face Recognition Models')
    parser.add_argument('--model', type=str, required=True, choices=['A', 'B', 'C'],
                        help='Model to visualize (A for Triplet, B for ArcFace, C for Combined)')
    
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method to use (tsne or pca)')
    
    args = parser.parse_args()
    main(args)

