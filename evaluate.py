import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve, DetCurveDisplay

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
    # Load the state dict, which is safer
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
        if model_name == 'A':
            embeds = model(images)
        else: # Models B and C
            embeds = model(images)
            # If the model returns a tuple (logits, embeddings), take the embeddings
            if isinstance(embeds, tuple):
                embeds = embeds[-1]


        embeddings_list.append(embeds.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        
    return np.concatenate(embeddings_list), np.concatenate(labels_list)

def calculate_scores(embeddings, labels):
    """Generates genuine and imposter pairs and calculates their scores."""
    # Cosine dissimilarity = 1 - cosine similarity
    
    genuine_scores = []
    imposter_scores = []
    
    unique_labels = np.unique(labels)
    
    # Create a map from label to indices
    label_map = {label: np.where(labels == label)[0] for label in unique_labels}

    print("Calculating genuine and imposter scores...")
    for i in tqdm(range(len(labels)), desc="Generating Pairs"):
        anchor_label = labels[i]
        anchor_embedding = embeddings[i]

        # --- Genuine Pairs ---
        positive_indices = label_map[anchor_label]
        for j in positive_indices:
            if i < j:
                positive_embedding = embeddings[j]
                sim = F.cosine_similarity(torch.from_numpy(anchor_embedding).unsqueeze(0),
                                          torch.from_numpy(positive_embedding).unsqueeze(0))
                genuine_scores.append(1 - sim.item())

        # --- Imposter Pairs ---
        # For efficiency, compare each anchor to one random sample from N other classes
        num_imposter_comparisons = min(10, len(unique_labels) - 1)
        imposter_labels = np.random.choice(
            [l for l in unique_labels if l != anchor_label], 
            num_imposter_comparisons, 
            replace=False
        )
        for imposter_label in imposter_labels:
            imposter_idx = np.random.choice(label_map[imposter_label])
            imposter_embedding = embeddings[imposter_idx]
            sim = F.cosine_similarity(torch.from_numpy(anchor_embedding).unsqueeze(0),
                                      torch.from_numpy(imposter_embedding).unsqueeze(0))
            imposter_scores.append(1 - sim.item())
                
    return np.array(genuine_scores), np.array(imposter_scores)

def calculate_metrics(genuine_scores, imposter_scores):
    """Calculates EER and TMR @ FMR thresholds."""
    
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    # We are calculating dissimilarity, so lower scores are better for genuines
    # For det_curve, the "positive class" (imposters) should have higher scores
    # This is naturally handled by our dissimilarity scores (genuines are close to 0, imposters are higher)
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    # DET curve calculation. pos_label=0 means we consider imposters as the positive class for FMR
    fmr, fnmr, thresholds = det_curve(y_true, y_scores, pos_label=0) 
    
    # EER: Find the point where FMR is closest to FNMR
    eer_idx = np.nanargmin(np.abs(fmr - fnmr))
    eer = (fmr[eer_idx] + fnmr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    
    print("\n" + "="*30)
    print("Performance Metrics")
    print("="*30)
    print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")

    # TMR @ FMR
    tmr_at_fmr = {}
    for fmr_target in [0.1, 0.01, 0.001, 0.0001]:
        try:
            idx = np.where(fmr <= fmr_target)[0][-1]
            tmr = 1 - fnmr[idx]
            tmr_at_fmr[fmr_target] = tmr
            print(f"TMR @ FMR={fmr_target}: {tmr:.4f} (Threshold: {thresholds[idx]:.4f})")
        except IndexError:
            tmr_at_fmr[fmr_target] = -1.0 # Indicate not reached
            print(f"TMR @ FMR={fmr_target}: Could not reach this low FMR.")
            
    return fmr, fnmr, eer

def plot_and_save_det(fmr, fnmr, eer, save_path):
    """Plots and saves the DET curve."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use DetCurveDisplay to plot the pre-calculated FMR and FNMR values
    display = DetCurveDisplay(fpr=fmr, fnr=fnmr)
    display.plot(ax=ax)
    
    # Plot EER point
    ax.plot(eer, eer, 'ro', markersize=8, label=f'EER = {eer:.4f}')
    
    ax.set_title('Detection Error Tradeoff (DET) Curve')
    ax.legend()
    ax.grid(True, which='both')
    
    print(f"\nSaving DET curve to {save_path}...")
    plt.savefig(save_path)
    plt.close()
    print("DET curve saved.")

def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = os.path.join('/home/teaching/assgn1/models', f'model_{args.model}')
    model_path = os.path.join(model_dir, f'model_{args.model}_final.pth')

    # Prepare data loader for test set
    print("Preparing test data loader...")
    loaders = get_data_loaders('/home/teaching/assgn1/forehead-v1-labeled', loader_type='arcface', batch_size=args.batch_size)
    test_loader = loaders['arcface_test']
    num_classes = len(test_loader.dataset.class_to_idx)

    # Load the trained model
    model = load_model(args.model, model_path, num_classes, device)

    # 1. Extract embeddings
    embeddings, labels = extract_embeddings(model, test_loader, device, args.model)
    
    # 2. Perform recognition and calculate scores
    genuine_scores, imposter_scores = calculate_scores(embeddings, labels)
    
    # 3. Save scores to file
    scores_path = os.path.join(model_dir, f'scores_model_{args.model}.txt')
    print(f"\nSaving scores to {scores_path}...")
    with open(scores_path, 'w') as f:
        for score in genuine_scores:
            f.write(f"{score:.6f} 1\n") # 1 for genuine
        for score in imposter_scores:
            f.write(f"{score:.6f} 0\n") # 0 for imposter
    print("Scores saved.")

    # 4. Report Performance metrics and plot DET curve
    fmr, fnmr, eer = calculate_metrics(genuine_scores, imposter_scores)
    
    det_curve_path = os.path.join(model_dir, f'det_curve_model_{args.model}.png')
    plot_and_save_det(fmr, fnmr, eer, det_curve_path)
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Face Recognition Models')
    parser.add_argument('--model', type=str, required=True, choices=['A', 'B', 'C'],
                        help='Model to evaluate (A for Triplet, B for ArcFace, C for Combined)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for extracting embeddings')
    
    args = parser.parse_args()
    main(args)

