import os
import json
import random
from datetime import datetime
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set base directory placeholder for anonymous submission
BASE_DIR = "BASE_DIR"
DATASET_NAME = "lubm"  # or "yago"

DATASET_ROOT = os.path.join(BASE_DIR, "Datasets")
STATISTICS_FOLDER = os.path.join(DATASET_ROOT, DATASET_NAME, "statistics")
POLICY_FILE = os.path.join(DATASET_ROOT, "star", "best_policies.json")
QUERY_PLAN_FOLDER = os.path.join(DATASET_ROOT, "plans")
RESULT_FOLDER = os.path.join(DATASET_ROOT, DATASET_NAME, "Results", "Tree_LSTM")

# Set a fixed random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

from Tree_LSTM.Tree_LSTM.DataLoading import TreeDataset, get_data_loaders
from Tree_LSTM.Tree_LSTM.Training_utils import (
    train_one_epoch,
    evaluate_model,
    get_loss_function,
    find_threshold_kde
)

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

def split_dataset_for_training(raw_dataset, fraction=1.0):
    all_keys = list(raw_dataset.keys())
    random.shuffle(all_keys)
    split_size = int(len(all_keys) * fraction)
    selected_keys = all_keys[:split_size]
    return {key: raw_dataset[key] for key in selected_keys}

def initialize_model(n_classes, size, embeddings_folder):
    from Tree_LSTM.Tree_LSTM.SPINN import SPINN
    model = SPINN(n_classes=n_classes, size=size, embeddings_folder=embeddings_folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def save_tp_fp_and_thresholds(output_folder, tp_probs, fp_probs):
    tp_fp_path = os.path.join(output_folder, "tp_fp_probabilities.json")
    with open(tp_fp_path, "w") as f:
        json.dump({"tp_probs": {str(k): v for k, v in tp_probs.items()},
                   "fp_probs": {str(k): v for k, v in fp_probs.items()}}, f, indent=4)
    print(f"Saved TP/FP to {tp_fp_path}")

    if tp_probs and fp_probs:
        thresholds = find_threshold_kde(tp_probs, fp_probs)
        threshold_path = os.path.join(output_folder, "thresholds_kde.json")
        with open(threshold_path, "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"Saved KDE thresholds to {threshold_path}")
        return torch.tensor(list(thresholds.values()))
    else:
        print("No valid TP/FP found. Using default thresholds.")
        return torch.full((6,), 0.5)

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 100
    train_fraction = 1.0

    raw_dataset = load_raw_dataset(QUERY_PLAN_FOLDER)
    raw_dataset = split_dataset_for_training(raw_dataset, fraction=train_fraction)

    with open(POLICY_FILE, "r") as file:
        query_policy_mapping = json.load(file)

    policy_to_one_hot = {
        "NoPolicy": [1, 0, 0, 0, 0, 0],
        "Ticket": [0, 1, 0, 0, 0, 0],
        "TpNLJ": [0, 0, 1, 0, 0, 0],
        "MiniOutputFirst": [0, 0, 0, 1, 0, 0],
        "Productivity": [0, 0, 0, 0, 1, 0],
        "ProNLJ": [0, 0, 0, 0, 0, 1],
    }

    dataset = TreeDataset(raw_dataset, query_policy_mapping, policy_to_one_hot, STATISTICS_FOLDER)
    train_loader, val_loader, train_size, val_size = get_data_loaders(dataset, batch_size, 0.8, True, 0)

    model, optimizer = initialize_model(6, 303, STATISTICS_FOLDER)
    model = model.to(device)
    criterion = get_loss_function().to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min', 0.5, 3, verbose=True)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_folder = os.path.join(RESULT_FOLDER, current_time)
    os.makedirs(output_folder, exist_ok=True)
    print("Results will be saved in:", output_folder)

    best_model_path = os.path.join(output_folder, "model.pth")
    optimizer_path = os.path.join(output_folder, "optimizer.pth")
    runtime_results_training = os.path.join(output_folder, "runtime_training.json")
    runtime_results_evaluation = os.path.join(output_folder, "runtime_evaluation.json")

    class_thresholds = torch.tensor([0.0] * 6)
    best_loss = float("inf")
    training_runtimes = []
    evaluation_runtimes = []
    best_tp_probs, best_fp_probs = {}, {}

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
            print(f"Epoch {epoch}: Best loss {best_loss:.4f}")

    with open(runtime_results_training, "w") as f:
        json.dump(training_runtimes, f, indent=4)

    class_thresholds = save_tp_fp_and_thresholds(output_folder, best_tp_probs, best_fp_probs)

    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    except FileNotFoundError:
        print("No best model found. Evaluating current weights.")

    if val_loader is not None and val_size > 0:
        test_loss, prec_all_zero, prec_non_all_zero, _, _, test_all_zero_count = evaluate_model(
            model, val_loader, val_size, criterion, class_thresholds, evaluation_runtimes
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


    print("Finished Training and Evaluation!")
