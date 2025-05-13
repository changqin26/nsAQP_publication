import os
import json
import random
from datetime import datetime
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set fixed random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)
# Automatic switch based on inductive setting
IS_INDUCTIVE = True  # <-- Flip this flag to switch behavior

if IS_INDUCTIVE:
    from Tree_LSTM.Tree_LSTM.TripleEncoding import get_rdf2vec_embedding_inductive
    import TripleEncoding
    TripleEncoding.get_rdf2vec_embedding = get_rdf2vec_embedding_inductive

# Local imports
from Tree_LSTM.DataLoading import TreeDataset, get_data_loaders
from Tree_LSTM.Training_utils import (
    train_one_epoch,
    evaluate_model,
    get_loss_function,
    plot_individual_probabilities,
    find_threshold_kde
)

# ------------------------------
# Load raw JSON datasets
# ------------------------------
def load_raw_dataset(directory_path):
    raw_dataset = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            query_id = filename.split('_')[0]
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r') as file:
                    tree_data = json.load(file)
                    raw_dataset[query_id] = tree_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {filename}: {e}")
    return raw_dataset

# ------------------------------
# Initialize SPINN model
# ------------------------------
def initialize_model(n_classes, size, embeddings_folder):
    from Tree_LSTM.SPINN import SPINN
    model = SPINN(n_classes=n_classes, size=size, embeddings_folder=embeddings_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    # Paths and settings (anonymized)
    BASE_DIR = os.getenv("DATASET_BASE_DIR", "BASE_DIR")
    DATASET_NAME = "lubm"

    embeddings_folder = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, "statistics")
    train_directory_path = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, "TreeLSTMInductive", "Train")
    eval_directory_path = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, "TreeLSTMInductive", "Test")
    policy_mapping_file_path = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, "star", "best_policies.json")
    base_output_folder = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, "Results", "Tree_LSTM", "Inductive")

    # Training parameters
    num_epochs = 20
    batch_size = 100

    # Load datasets
    train_raw_dataset = load_raw_dataset(train_directory_path)
    test_raw_dataset = load_raw_dataset(eval_directory_path)


    with open(policy_mapping_file_path, "r") as file:
        query_policy_mapping = json.load(file)

    # One-hot encoding for routing strategies
    policy_to_one_hot = {
        "NoPolicy": [1, 0, 0, 0, 0, 0],
        "Ticket": [0, 1, 0, 0, 0, 0],
        "TpNLJ": [0, 0, 1, 0, 0, 0],
        "MiniOutputFirst": [0, 0, 0, 1, 0, 0],
        "Productivity": [0, 0, 0, 0, 1, 0],
        "ProNLJ": [0, 0, 0, 0, 0, 1],
    }

    train_dataset = TreeDataset(train_raw_dataset, query_policy_mapping, policy_to_one_hot, embeddings_folder)
    eval_dataset = TreeDataset(test_raw_dataset, query_policy_mapping, policy_to_one_hot, embeddings_folder)

    train_loader, _, train_size, _ = get_data_loaders(train_dataset, batch_size=100, train_split=1.0, is_training=True)
    _, eval_loader, _, eval_size = get_data_loaders(eval_dataset, batch_size=100, train_split=1.0, is_training=False)

    model, optimizer = initialize_model(n_classes=6, size=303, embeddings_folder=embeddings_folder)
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = get_loss_function().to(device)

    # Output paths
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_folder = os.path.join(base_output_folder, current_time)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Results saved to: {output_folder}")

    best_model_path = os.path.join(output_folder, "model.pth")
    optimizer_path = os.path.join(output_folder, "optimizer.pth")
    runtime_results_training = os.path.join(output_folder, "runtime_training.json")
    runtime_results_evaluation = os.path.join(output_folder, "runtime_evaluation.json")

    class_thresholds = torch.tensor([0.0] * 6)
    best_loss = float("inf")
    best_tp_probs, best_fp_probs = {}, {}
    training_runtimes, evaluation_runtimes = [], []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, _, tp_probs, fp_probs = train_one_epoch(
            model, train_loader, train_size, optimizer,
            criterion, class_thresholds, epoch, num_epochs,
            training_runtimes
        )
        scheduler.step(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_tp_probs = tp_probs
            best_fp_probs = fp_probs
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            #Save TP/FP probabilities from best epoch
            tp_fp_path = os.path.join(output_folder, "tp_fp_probabilities.json")
            with open(tp_fp_path, "w") as f:
                json.dump({
                    "tp_probs": {str(k): v for k, v in best_tp_probs.items()},
                    "fp_probs": {str(k): v for k, v in best_fp_probs.items()}
                }, f, indent=4)
            print(f"Saved best-epoch TP/FP to {tp_fp_path}")

    # Thresholds from KDE
    if best_tp_probs and best_fp_probs:
        new_thresholds = find_threshold_kde(best_tp_probs, best_fp_probs)
        class_thresholds = torch.tensor(list(new_thresholds.values()))
        with open(os.path.join(output_folder, "thresholds_kde.json"), "w") as f:
            json.dump(new_thresholds, f, indent=4)

    # Evaluation
    model.load_state_dict(torch.load(best_model_path))
    if eval_loader is not None and eval_size > 0:
        test_loss, prec_all_zero, prec_non_all_zero, _, _,  test_all_zero_count = evaluate_model(
            model, eval_loader, eval_size, criterion, class_thresholds, evaluation_runtimes
        )
        with open(runtime_results_evaluation, "w") as f:
            json.dump({"total_times": evaluation_runtimes}, f, indent=4)
        print(f"Saved test evaluation runtime to {runtime_results_evaluation}")

        test_metrics_path = os.path.join(output_folder, "test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump({
                "test_loss": round(test_loss, 4),
                "precision_all_zero": round(prec_all_zero, 4),
                "precision_non_all_zero": round(prec_non_all_zero, 4),
                "num_all_zero_predictions": test_all_zero_count
            }, f, indent=4)
        print(f"Saved test evaluation metrics to {test_metrics_path}")

    print("Finished Training and Evaluation.")
