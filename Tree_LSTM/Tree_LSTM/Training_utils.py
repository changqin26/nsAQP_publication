# Training_utils.py
import json
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns

from scipy.stats import gaussian_kde

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


###############################################################################
# 1) Custom Weighted BCE
###############################################################################
class CustomBCEWithWeights(nn.Module):
    def __init__(self, positive_weights, negative_weights):
        super(CustomBCEWithWeights, self).__init__()
        self.register_buffer("positive_weights",
                             torch.as_tensor(positive_weights, dtype=torch.float32))
        self.register_buffer("negative_weights",
                             torch.as_tensor(negative_weights, dtype=torch.float32))

    def forward(self, outputs, targets):
        """
        outputs: [batch_size, num_classes] (probabilities, typically post-sigmoid)
        targets: [batch_size, num_classes] (0 or 1)
        """
        pos_w = self.positive_weights
        neg_w = self.negative_weights

        # Because outputs are already sigmoid-ed in your SPINN, we can treat them as probabilities:
        probs = outputs
        eps = 1e-6

        losses = -pos_w * targets * torch.log(probs + eps) \
                 - neg_w * (1 - targets) * torch.log(1 - probs + eps)
        return torch.mean(losses)


def get_loss_function():
    # Example: all weights = 1.0
    positive_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    negative_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    return CustomBCEWithWeights(positive_weights, negative_weights)


###############################################################################
# 2) Training a single epoch
###############################################################################
def train_one_epoch(
    model,
    data_loader,
    train_size,
    optimizer,
    criterion,
    thresholds,
    current_epoch,
    total_epochs,
    training_runtimes
):
    """
    A single epoch of training for a multi-label model.

    data_loader: yields (root_nodes, labels) pairs,
                 where 'root_nodes' is a list of pre-built TreeNode objects,
                 and 'labels' is a [batch_size, num_classes] tensor.
    train_size: total number of samples in training set (for avg loss).
    thresholds: per-class thresholds for turning probabilities -> binary.
    training_runtimes: a list to which we append an epoch runtime string.
    """
    model.train()
    total_loss = 0.0

    # For threshold fine-tuning (KDE):
    tp_probs = {}
    fp_probs = {}

    epoch_start_time = time.time()

    # The key change: enumerate(tqdm(...)) so we get a batch_idx.
    for batch_idx, (root_nodes, labels) in enumerate(
        tqdm(data_loader, desc=f"Epoch {current_epoch}/{total_epochs}")
    ):
        # -----------------------------------------------
        # Optional debug prints for the first batch only:
        #if batch_idx == 0:
           # print(f"[DEBUG] In train_one_epoch, batch_idx={batch_idx}")
            #print(f"         root_nodes batch size: {len(root_nodes)}")
            #print(f"         labels shape: {labels.shape}")
            #if len(root_nodes) > 0 and root_nodes[0] is not None:
               #print(f"         First root_node ID: {root_nodes[0].node_id}")
            #else:
                #print("         First root_node is None!")
        # -----------------------------------------------

        optimizer.zero_grad()

        labels = labels.to(device)

        # Forward pass
        outputs = model(root_nodes)  # shape: [batch_size, num_classes]
        loss = criterion(outputs, labels)

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * labels.size(0)

        # Convert probabilities to binary predictions
        preds_binary = (outputs > thresholds.to(device=outputs.device, dtype=torch.float32)).int()

        # Convert to CPU/NumPy for analysis
        labels_np = labels.cpu().numpy()
        preds_binary_np = preds_binary.cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()

        # Collect TP/FP probabilities for potential KDE threshold updates
        for j in range(outputs.size(1)):  # for each class
            tp_mask = (labels_np[:, j] == 1) & (preds_binary_np[:, j] == 1)
            fp_mask = (labels_np[:, j] == 0) & (preds_binary_np[:, j] == 1)

            class_probs = outputs_np[:, j]
            tp_probs.setdefault(j, []).extend(class_probs[tp_mask].tolist())
            fp_probs.setdefault(j, []).extend(class_probs[fp_mask].tolist())

    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time

    avg_loss = total_loss / train_size

    print(f"[TRAIN] Epoch {current_epoch}/{total_epochs} | "
          f"Loss={avg_loss:.4f} | Time={epoch_total_time:.2f}s")

    training_runtimes.append(f"Epoch: {current_epoch} Loss: {avg_loss:.4f} Time(ms): {epoch_total_time*1000:.0f}")

    return avg_loss, None, tp_probs, fp_probs



###############################################################################
# 3) Evaluate / Validate
###############################################################################
def evaluate_model(
    model,
    data_loader,
    val_size,
    criterion,
    thresholds,
    evaluation_runtimes
):
    """
    Evaluate the model on a validation set.
    data_loader: yields (root_nodes, labels)
    """
    model.eval()
    total_loss = 0.0

    # Track how many predictions ended up being all-zero (before fix),
    # and measure precision in those cases separately
    all_zero_precisions = []
    non_all_zero_precisions = []
    all_zeros_count_eval = 0

    # For threshold analysis
    tp_probs_val = {}
    fp_probs_val = {}

    with torch.no_grad():
        for root_nodes, labels in data_loader:
            eval_start_time = time.time()

            labels = labels.to(device)
            outputs = model(root_nodes)  # shape: [batch_size, num_classes]

            loss = criterion(outputs, labels.float())
            total_loss += loss.item() * labels.size(0)

            # Convert probabilities -> binary using current thresholds
            preds_binary = (outputs > thresholds.to(outputs.device, dtype=torch.float32)).int()


            # Identify how many predictions are all-zero
            all_zero_vectors = (preds_binary.sum(dim=1) == 0)
            all_zeros_count_eval += all_zero_vectors.sum().item()

            # "Fix" those all-zero predictions by turning on the class with the highest prob
            for i in range(len(preds_binary)):
                if all_zero_vectors[i]:
                    max_prob_index = outputs[i].argmax().item()
                    preds_binary[i, max_prob_index] = 1

            # Evaluate precision for each sample
            labels_np = labels.cpu().numpy()
            preds_binary_np = preds_binary.cpu().numpy()
            outputs_np = outputs.detach().cpu().numpy()

            for i in range(len(preds_binary)):
                tp = ((labels_np[i] == 1) & (preds_binary_np[i] == 1)).sum()
                fp = ((labels_np[i] == 0) & (preds_binary_np[i] == 1)).sum()
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    if all_zero_vectors[i]:
                        all_zero_precisions.append(precision)
                    else:
                        non_all_zero_precisions.append(precision)

            # Collect TP/FP probabilities
            for j in range(outputs.size(1)):
                tp_mask = (labels_np[:, j] == 1) & (preds_binary_np[:, j] == 1)
                fp_mask = (labels_np[:, j] == 0) & (preds_binary_np[:, j] == 1)

                class_probs = outputs_np[:, j]
                tp_probs_val.setdefault(j, []).extend(class_probs[tp_mask].tolist())
                fp_probs_val.setdefault(j, []).extend(class_probs[fp_mask].tolist())

            eval_end_time = time.time()
            evaluation_runtimes.append((eval_end_time - eval_start_time)*1000)

    avg_loss = total_loss / val_size
    avg_precision_all_zero = np.mean(all_zero_precisions) if all_zero_precisions else 0
    avg_precision_non_all_zero = np.mean(non_all_zero_precisions) if non_all_zero_precisions else 0


    return avg_loss, avg_precision_all_zero, avg_precision_non_all_zero, tp_probs_val, fp_probs_val, all_zeros_count_eval


###############################################################################
# 4) Threshold Finding (KDE)
###############################################################################
def find_threshold_kde(tp_probs, fp_probs):
    """
    Calculates thresholds using a KDE-based approach for each class index.
    """
    print(">>> find_threshold_kde() called.")

    if not tp_probs or not fp_probs:
        print("No collected TP/FP data. Using default thresholds of 0.5.")
        # Return a threshold=0.5 for each known class
        return {cls: 0.5 for cls in range(len(tp_probs))}

    x_grid = np.linspace(0, 1, 1000)
    thresholds = {}

    for cls in tp_probs.keys():
        if cls not in fp_probs:
            # If there's no FP info for that class, skip or set default
            thresholds[cls] = 0.5
            continue

        tp_list = tp_probs[cls]
        fp_list = fp_probs[cls]

        # If not enough data, fallback
        if len(tp_list) < 2 or len(fp_list) < 2:
            thresholds[cls] = 0.5
            continue

        # If zero variance, fallback
        if np.var(tp_list) == 0 or np.var(fp_list) == 0:
            thresholds[cls] = 0.5
            continue

        try:
            tp_kde = gaussian_kde(tp_list, bw_method=0.1)
            fp_kde = gaussian_kde(fp_list, bw_method=0.1)

            tp_dens = tp_kde(x_grid)
            fp_dens = fp_kde(x_grid)

            # Restrict to overlapping region
            tp_min, tp_max = min(tp_list), max(tp_list)
            fp_min, fp_max = min(fp_list), max(fp_list)
            valid_min = max(tp_min, fp_min)
            valid_max = min(tp_max, fp_max)

            valid_indices = (x_grid >= valid_min) & (x_grid <= valid_max)
            restricted_x = x_grid[valid_indices]
            restricted_tp = tp_dens[valid_indices]
            restricted_fp = fp_dens[valid_indices]

            # Find the first point where TP > FP
            dominance_indices = np.where(restricted_tp > restricted_fp)[0]
            if len(dominance_indices) > 0:
                best_threshold = restricted_x[dominance_indices[0]]
            else:
                best_threshold = 0.5

            thresholds[cls] = best_threshold

        except Exception as e:
            print(f"  [Error in KDE] Class {cls}: {e}")
            thresholds[cls] = 0.5

    return thresholds


###############################################################################
# 5) Optional: Plot Distributions
###############################################################################
def plot_individual_probabilities(tp_probs, fp_probs, dataset_type):
    """
    For each class, plot the distribution of probabilities for
    True Positives vs. False Positives.
    """
    first_figure = True

    for cls, tp_probs_cls in tp_probs.items():
        if cls not in fp_probs:
            continue

        fp_probs_cls = fp_probs[cls]

        tp_data = tp_probs_cls
        fp_data = fp_probs_cls

        plt.figure(figsize=(5, 2.5))

        sns.histplot(tp_data, bins=20, kde=True, color='blue',
                     label='True Positives' if first_figure else None,
                     stat='density', alpha=0.5)
        sns.histplot(fp_data, bins=20, kde=True, color='red',
                     label='False Positives' if first_figure else None,
                     stat='density', alpha=0.5)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('')
        plt.xlabel('')
        #plt.ylim(0, 80)

        if first_figure:
            plt.legend(fontsize=8)
            first_figure = False

        plt.grid(True)
        plt.tight_layout()

        pdf_filename = f'YAGO_{dataset_type}_class_{cls}_50.pdf'
        plt.savefig(pdf_filename, format='pdf')
        plt.show()
