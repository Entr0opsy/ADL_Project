
import os
import argparse
import sys
import numpy as np
from sklearn.metrics import roc_curve, DetCurveDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_checkpoint(model_name: str, checkpoint_path: str) -> str:
    """Generate a standardized checkpoint filename.

    Example result: "<checkpoint_path>/<model_name>_checkpoint.pt"
    Ensures the checkpoint directory exists and returns the full path.
    """
    # Ensure directory exists
    if checkpoint_path is None or checkpoint_path == "":
        checkpoint_path = "."
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = f"{model_name}_checkpoint.pt"
    return os.path.join(checkpoint_path, filename)


def write_scores(
    output_file: str,
    embedding_dict_test: dict,
    embedding_dict_train: dict,
    model_name: str = "model",
    use_asymmetric_conv: bool = True,
    preserve_vertical: bool = True,
    use_cosine_similarity: bool = True,
    progress_updates: int = 20,
):
    """Write pairwise similarity scores to `output_file` using the same
    textual pattern as `evaluation.py`.

    - Writes a small header with metadata and then tab-separated rows:
      idx1\tpose1\tidx2\tpose2\tisGen\tscore
    - Returns the absolute path to the written file so it can be passed
      into `load_scores` or other utilities.
    """
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_test = len(embedding_dict_test)
    progress_interval = max(1, total_test // max(1, progress_updates))

    with open(output_file, "w") as file:
        file.write(f"# Model: {model_name}\n")
        file.write(f"# Asymmetric Conv: {use_asymmetric_conv}\n")
        file.write(f"# Preserve Vertical: {preserve_vertical}\n")
        similarity_metric = "cosine_similarity" if use_cosine_similarity else "euclidean_distance"
        file.write(f"# Similarity Metric: {similarity_metric}\n")
        file.write(f"# Test samples: {len(embedding_dict_test)}\n")
        file.write(f"# Train samples: {len(embedding_dict_train)}\n")
        file.write("#\n")
        file.write("idx1\tpose1\tidx2\tpose2\tisGen\tscore\n")

        for progress_idx, (k_test, v_test) in enumerate(embedding_dict_test.items()):
            idx1, pose1 = k_test.split("_", 1)

            for k_train, v_train in embedding_dict_train.items():
                idx2, pose2 = k_train.split("_", 1)
                isGen = 1 if idx1 == idx2 else 0

                # Ensure tensors are on the same device and shape
                # v_test and v_train are expected to be shaped [1, dim]
                try:
                    if use_cosine_similarity:
                        score = F.cosine_similarity(v_test, v_train).item()
                    else:
                        score = -torch.dist(v_test, v_train, p=2).item()
                except Exception:
                    # Fall back to numpy if tensors are not torch tensors
                    try:
                        a = np.asarray(v_test).astype(float)
                        b = np.asarray(v_train).astype(float)
                        if use_cosine_similarity:
                            # cosine similarity
                            a_flat = a.reshape(-1)
                            b_flat = b.reshape(-1)
                            denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
                            score = float(np.dot(a_flat, b_flat) / denom) if denom != 0 else 0.0
                        else:
                            score = -float(np.linalg.norm(a - b))
                    except Exception:
                        score = 0.0

                file.write(f"{idx1}\t{pose1}\t{idx2}\t{pose2}\t{isGen}\t{score}\n")

            if (progress_idx + 1) % progress_interval == 0:
                progress = (progress_idx + 1) / total_test * 100
                print(f"  Progress: {progress:.1f}%")

    return os.path.abspath(output_file)


# ------------------------ Robust Score File Loader ------------------------
def load_scores(path, delimiter="\t", label_col=4, score_col=5, max_lines=None):
    """Load label and score columns robustly by streaming and skipping malformed lines.

    Returns (labels_array, scores_array, skipped_count, total_read)
    """
    labels = []
    scores = []
    skipped = 0
    total = 0

    # Support gzipped files transparently if needed
    open_fn = open
    if path.endswith('.gz'):
        import gzip
        open_fn = gzip.open

    with open_fn(path, 'rt', errors='replace') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            total += 1
            # split by delimiter or whitespace fallback
            parts = line.rstrip('\n').split(delimiter)
            if len(parts) == 1:
                # maybe whitespace separated
                parts = line.split()
            try:
                # guard against short lines
                if max(label_col, score_col) >= len(parts):
                    raise ValueError('not enough columns')
                lab = int(parts[label_col])
                sc = float(parts[score_col])
                labels.append(lab)
                scores.append(sc)
            except Exception:
                skipped += 1
                # continue on malformed lines
                continue

    if len(labels) == 0:
        raise ValueError(f'No valid (label,score) pairs were parsed from {path} (skipped {skipped} lines)')

    return np.array(labels, dtype=int), np.array(scores, dtype=float), skipped, total


def plot_scores(scores_file, max_lines=None):
    
    labels, scores, skipped, total = load_scores(scores_file, max_lines)

    valid_mask = (labels == 0) | (labels == 1)
    labels = labels[valid_mask]
    scores = scores[valid_mask]

    # Split into genuine (label=1) and impostor (label=0)
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    print(f'Read {len(labels)} valid pairs from {total} lines ({skipped} skipped)')
    print('Genuine total:', len(genuine_scores))
    print('Impostor total:', len(impostor_scores))

    # ------------------------ EER Calculation (Vectorized) ------------------------
    fpr, tpr, thresholds = roc_curve(labels, scores)   # fpr = FMR, tpr = TMR
    fnmr = 1 - tpr                                     # FNMR

    # Find point where |FMR - FNMR| is minimum
    eer_index = np.nanargmin(np.abs(fpr - fnmr))
    eer = (fpr[eer_index] + fnmr[eer_index]) / 2

    print(f"\nEqual Error Rate (EER): {eer * 100:.2f}%")

    # ------------------------ TMR at Specific FMR Values ------------------------
    fmr_targets = [0.1, 0.01, 0.001]  # In percentage (%)

    # Sort impostor and genuine scores (only once, for efficiency)
    impostor_scores.sort()
    genuine_scores.sort()

    print("\n--- TMR at Specific FMR values ---")
    for target in fmr_targets:
        # FMR threshold → score at that percentile of impostor distribution
        threshold = np.percentile(impostor_scores, target)
        # TMR = fraction of genuine scores BELOW this threshold
        tmr = np.sum(genuine_scores >= threshold) / len(genuine_scores)
        print(f"TMR at FMR {target}% = {tmr * 100:.2f}%")

    # ------------------------ (Optional) Plot ROC Curve ------------------------
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.grid()
    plt.legend()
    plt.show()

    # ------------------------ (Optional) DET Curve ------------------------
    plt.figure()
    DetCurveDisplay.from_predictions(labels, scores)
    plt.title('DET Curve')
    plt.grid()
    plt.show()

    # ------------------------ (Optional) Histogram ------------------------
    plt.figure()
    plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor', density=True)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Genuine vs Impostor Score Distribution')
    plt.legend()
    plt.grid()
    plt.show()

