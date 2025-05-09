import json
import os
import numpy as np
import torch
import time
from datetime import datetime
import random

from matplotlib import pyplot as plt
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde
from tqdm import tqdm
from pathlib import Path
from models_policy import TripleModelForPolicyClassification
from utils import get_query_graph_data_new, StatisticsLoader, ToUndirectedCustom
import seaborn as sns
import hashlib

BASE_DIR = os.getenv("DATASET_BASE_DIR")
if BASE_DIR is None:
    raise ValueError("Please set the DATASET_BASE_DIR environment variable.")

def json_hash(json_path):
    with open(json_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()

def find_threshold_kde(tp_probs, fp_probs):
    if not tp_probs or not fp_probs:
        print("No true positives or false positives collected. Using default thresholds.")
        return {cls: 0.5 for cls in range(len(tp_probs))}  # Default threshold for each class

    x_grid = np.linspace(0, 1, 1000)
    thresholds = {}
    for cls in tp_probs:
        tp_kde = gaussian_kde(tp_probs[cls])
        fp_kde = gaussian_kde(fp_probs[cls])

        tp_dens = tp_kde(x_grid)
        fp_dens = fp_kde(x_grid)

        # Calculate the range where actual data exists
        tp_min, tp_max = min(tp_probs[cls]), max(tp_probs[cls])
        fp_min, fp_max = min(fp_probs[cls]), max(fp_probs[cls])
        valid_min = max(tp_min, fp_min)  # Minimum where both TP and FP exist
        valid_max = min(tp_max, fp_max)  # Maximum where both TP and FP exist

        # Restrict x_grid and densities to the valid range
        valid_indices = (x_grid >= valid_min) & (x_grid <= valid_max)
        restricted_x_grid = x_grid[valid_indices]
        restricted_tp_dens = tp_dens[valid_indices]
        restricted_fp_dens = fp_dens[valid_indices]

        # Find the first dominance point within the restricted range
        dominance_indices = np.where(restricted_tp_dens > restricted_fp_dens)[0]
        if len(dominance_indices) > 0:
            best_threshold = restricted_x_grid[dominance_indices[0]]
        else:
            best_threshold = 0.5

        thresholds[cls] = best_threshold
    return thresholds

def prepare_data_list(data_points, statistics, device, num_classes, inductive):
    data_list = []
    n_atoms = 0
    for datapoint in data_points:
        if inductive == 'full':
            data, n_atoms = get_query_graph_data_new(
                datapoint, statistics, device, unknown_entity='false', n_atoms=n_atoms
            )
        else:
            data, n_atoms = get_query_graph_data_new(
                datapoint, statistics, device, n_atoms=n_atoms
            )
            #print("Evaluation (inductive='full') entity embeddings:")
            #print(data["entity"].x)  # Prints the embeddings tensor for entities

        data = ToUndirectedCustom(merge=False)(data)
        data = data.to_homogeneous()
        data = data.to(device)

        y = torch.tensor(datapoint['y'], dtype=torch.float32)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.size(0) == 1 and y.size(1) != num_classes:
            y = y.unsqueeze(0)

        data_list.append((data, y))
    return data_list, n_atoms

def custom_collate_fn(batch):
    data_list, target_list = [], []
    for data, target in batch:
        data_list.append(data)
        target_list.append(target)
    return Batch.from_data_list(data_list), torch.stack(target_list)


class PolicyClassification:
    def __init__(self, dataset_name, graphfile, sim_measure, DATASETPATH):
        self.dataset_name = dataset_name
        self.graphfile = graphfile
        self.sim_measure = sim_measure
        self.DATASETPATH = DATASETPATH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.statistics = self.load_statistics()

    def load_statistics(self):
        try:
            stats_path = os.path.join(self.DATASETPATH, self.dataset_name, self.dataset_name + "_embeddings.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    return json.load(f)
            else:
                return StatisticsLoader(os.path.join(self.DATASETPATH, self.dataset_name, "statistics"))
        except Exception as e:
            print(f"No statistics found: {e}")
            exit()

    def train_GNN(
        self, train_data, test_data, thresholds, num_classes=6, epochs=100, train=True,
        eval_folder=None, inductive='false', preparation_time=None, batch_size=100
    ):
        assert preparation_time is not None

        # Folder for Results
        starttime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = Path(self.DATASETPATH) / self.dataset_name / "Results" / "GNN" / starttime
        result_path.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

        print("Starting Training...")

        # Initialize the model
        model = TripleModelForPolicyClassification(num_classes=num_classes).to(self.device).double()
        print(f"Model is on device: {next(model.parameters()).device}")
        print("Number of Parameters: ", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        best_epoch = None
        best_model_state = None

        # We will store best TP/FP
        best_tp_probs = {i: [] for i in range(num_classes)}
        best_fp_probs = {i: [] for i in range(num_classes)}

        offline_start_time = time.time()
        training_runtimes = []

        if train:
            starttime_training = time.time()
            X, n_atoms = prepare_data_list(train_data, self.statistics, self.device, num_classes, inductive)
            X_test, _ = prepare_data_list(test_data, self.statistics, self.device, num_classes, inductive)
            preparation_time += time.time() - starttime_training

            offline_end_time = time.time()
            print(f"Offline Preparation Time: {offline_end_time - offline_start_time:.2f} seconds")

            train_loader = DataLoader(X, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

            for epoch in tqdm(range(epochs), desc="Training Epochs"):
                epoch_start_time = time.time()
                epoch_loss = 0.0

                # Local dictionaries for each epoch
                tp_probs_current = {i: [] for i in range(num_classes)}
                fp_probs_current = {i: [] for i in range(num_classes)}

                # --- Train loop ---
                for batch_data, batch_y in train_loader:
                    optimizer.zero_grad()
                    batch_data = batch_data.to(self.device)
                    batch_y = batch_y.to(self.device)

                    out = model(
                        batch_data.x.double(), batch_data.edge_index,
                        batch_data.edge_type, batch_data.edge_attr.double(),
                        batch=batch_data.batch
                    )

                    loss = criterion(out, batch_y)
                    epoch_loss += loss.item() * batch_y.size(0)
                    loss.backward()
                    optimizer.step()

                    # Collect probabilities and convert to predictions
                    probabilities = torch.sigmoid(out)
                    # shape: [batch_size, num_classes]
                    preds = (probabilities > torch.tensor(
                        [thresholds[i] for i in range(num_classes)], device=self.device
                    )).int()

                    # Accumulate TP/FP for the current epoch
                    for j in range(num_classes):
                        tp_mask = (batch_y[:, j] == 1) & (preds[:, j] == 1)
                        fp_mask = (batch_y[:, j] == 0) & (preds[:, j] == 1)
                        tp_probs_current[j].extend(probabilities[tp_mask, j].detach().cpu().numpy())
                        fp_probs_current[j].extend(probabilities[fp_mask, j].detach().cpu().numpy())

                # End of epoch
                #print(f"Number of batches in train_loader: {len(train_loader)}")
                avg_epoch_loss = epoch_loss / len(train_data)
                epoch_end_time = time.time()
                epoch_total_time = epoch_end_time - epoch_start_time

                # Track runtime
                training_runtimes.append(
                    f"Epoch: {epoch + 1} Loss: {avg_epoch_loss:.4f} Time: {epoch_total_time * 1000:.0f} ms"
                )

                # Check if this is the best epoch so far
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict()

                    # Save the best epoch's TP/FP
                    best_tp_probs = tp_probs_current
                    best_fp_probs = fp_probs_current

            # After training, we have best_tp_probs / best_fp_probs for the best epoch
            # Save best TP/FP probabilities to JSON
            tp_fp_save = {
                "tp_probs": best_tp_probs,
                "fp_probs": best_fp_probs
            }
            tp_fp_path = result_path / "tp_fp_probabilities_lubm.json" # or "tp_fp_probabilities_yago.json" for yago dataset
            with open(tp_fp_path, "w") as f:
                json.dump(tp_fp_save, f, indent=4)
            print(f"TP/FP probabilities saved to {tp_fp_path}")
            print(f"[TRAIN] Hash: {json_hash(tp_fp_path)}")

            # Recompute thresholds from best epoch
            new_thresholds = find_threshold_kde(best_tp_probs, best_fp_probs)
            for cls, val in new_thresholds.items():
                print(f"[TRAIN] Class {cls} – threshold = {val:.6f}")
            thresholds_path = result_path / "thresholds_kde.json"
            with open(thresholds_path, "w") as f:
                json.dump(new_thresholds, f, indent=4)
            print(f"Thresholds saved to {thresholds_path}")

            #print("Calculated new thresholds:", new_thresholds)

            # Save training runtimes to JSON
            training_runtime_path = result_path / "runtime_training.json"
            with open(training_runtime_path, "w") as file:
                json.dump(training_runtimes, file, indent=4)
            print(f"Training runtimes saved to {training_runtime_path}")

            # Load the best model state
            if best_model_state:
                model.load_state_dict(best_model_state)

            # Save model and optimizer
            model_path = os.path.join(self.DATASETPATH, self.dataset_name, "Results", "GNN", starttime, "model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            optimizer_path = os.path.join(self.DATASETPATH, self.dataset_name, "Results", "GNN", starttime, "optimizer.pth")
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"Optimizer state saved to {optimizer_path}")
        else:
            # If not training, we might still want to create a test_loader, etc.
            test_loader = DataLoader(
                prepare_data_list(test_data, self.statistics, self.device, num_classes, inductive)[0],
                batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
            )
            new_thresholds = {i: 0.5 for i in range(num_classes)}  # default

        # -------------- Evaluation --------------
        print("Evaluating...")
        model.eval()

        total_loss = 0
        # Separate accumulators for all-zero and non-all-zero queries
        precision_accumulator_all_zero = []
        precision_accumulator_non_all_zero = []
        all_zeros_count_eval = 0
        all_zeros_details_eval = []  # Store details for the first 5 all-zero predictions

        # For storing final evaluation TP/FP
        tp_probs_eval = {i: [] for i in range(num_classes)}
        fp_probs_eval = {i: [] for i in range(num_classes)}

        total_times = []

        with torch.no_grad():
            # We assume test_loader was defined above (when `train=True`).
            # If train=False, we created it inline above with default thresholds, etc.
            for batch_data, batch_y in test_loader:
                query_start_time = time.time()
                batch_data = batch_data.to(self.device)
                batch_y = batch_y.to(self.device)

                out = model(
                    batch_data.x.double(), batch_data.edge_index,
                    batch_data.edge_type, batch_data.edge_attr.double(),
                    batch=batch_data.batch
                )
                query_end_time = time.time()
                total_time = (query_end_time - query_start_time) * 1000
                total_times.append(total_time)

                loss = criterion(out, batch_y)
                total_loss += loss.item()* batch_y.size(0)

                probabilities = torch.sigmoid(out)
                # Apply new_thresholds from best epoch
                preds = (probabilities > torch.tensor(
                    [new_thresholds[i] for i in range(num_classes)], device=self.device
                )).int()

                # Identify all-zero predictions
                all_zeros_mask = (preds.sum(dim=1) == 0)
                all_zeros_count_eval += all_zeros_mask.sum().item()

                # Handle all-zero predictions by assigning the policy with the highest probability
                for i in range(len(preds)):
                    if all_zeros_mask[i]:
                        max_prob_index = probabilities[i].argmax().item()
                        preds[i, max_prob_index] = 1
                        if len(all_zeros_details_eval) < 5:
                            all_zeros_details_eval.append({
                                "probability": probabilities[i].cpu().numpy().tolist(),
                                "prediction": preds[i].cpu().numpy().tolist()
                            })

                # Calculate precision for all-zero vs. non-all-zero
                for i, (data_point, target) in enumerate(zip(preds, batch_y)):
                    true_positive = (data_point == 1) & (target == 1)
                    predicted_positive = (data_point == 1)

                    if all_zeros_mask[i]:
                        # Precision for all-zero queries
                        if predicted_positive.sum().item() > 0:
                            precision = true_positive.sum().item() / predicted_positive.sum().item()
                            precision_accumulator_all_zero.append(precision)
                    else:
                        # Precision for non-all-zero queries
                        if predicted_positive.sum().item() > 0:
                            precision = true_positive.sum().item() / predicted_positive.sum().item()
                            precision_accumulator_non_all_zero.append(precision)

                # Collect data for final eval TP/FP
                for j in range(num_classes):
                    tp_mask_eval = (batch_y[:, j] == 1) & (preds[:, j] == 1)
                    fp_mask_eval = (batch_y[:, j] == 0) & (preds[:, j] == 1)
                    tp_probs_eval[j].extend(probabilities[tp_mask_eval, j].cpu().numpy())
                    fp_probs_eval[j].extend(probabilities[fp_mask_eval, j].cpu().numpy())
        #print(f"Number of batches in test_loader: {len(test_loader)}")
        avg_loss = total_loss / len(test_data)
        avg_precision_all_zero = (
            np.mean(precision_accumulator_all_zero)
            if precision_accumulator_all_zero else 0
        )
        avg_precision_non_all_zero = (
            np.mean(precision_accumulator_non_all_zero)
            if precision_accumulator_non_all_zero else 0
        )

        print(f"Evaluation: Loss: {avg_loss:.4f}")
        print(f"All-Zero Predictions (Eval): {all_zeros_count_eval}")
        print(f"Eval Precision (All-Zero): {avg_precision_all_zero:.4f}")
        print(f"Eval Precision (Non-All-Zero): {avg_precision_non_all_zero:.4f}")

        # Save evaluation runtimes
        runtime_results = {
            "mean_total_query_time": np.mean(total_times),
            "total_times": total_times
        }

        eval_runtime_path = os.path.join(self.DATASETPATH, self.dataset_name, "Results", "GNN", starttime,
                                         "runtime_evaluation.json")
        with open(eval_runtime_path, "w") as file:
            json.dump(runtime_results, file, indent=4)

        print(f"Runtime results saved to {eval_runtime_path}")

        metrics = {
            "test_loss": round(avg_loss, 4),
            "precision_all_zero": round(avg_precision_all_zero, 4),
            "precision_non_all_zero": round(avg_precision_non_all_zero, 4),
            "num_all_zero_predictions": all_zeros_count_eval
        }

        test_metrics_path = f"{self.DATASETPATH}{self.dataset_name}/Results/GNCE/{starttime}/test_metrics.json"
        with open(test_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Test metrics saved to {test_metrics_path}")

        tp_fp_path = f"{self.DATASETPATH}{self.dataset_name}/Results/GNCE/{starttime}/tp_fp_probabilities.json"
        with open(tp_fp_path, "w") as f:
            json.dump({
                "tp_probs": {str(k): v for k, v in best_tp_probs.items()},
                "fp_probs": {str(k): v for k, v in best_fp_probs.items()}
            }, f, indent=4)
        print(f"Saved best-epoch TP/FP to {tp_fp_path}")

        # ---------------- Return final results as a dict ----------------
        return metrics


def train_GNCE(dataset: str, query_type: str, eval_folder: str, query_filename: str,
               train: bool = True, inductive='false', DATASETPATH=None, num_classes=6):
    # Total counter for preparation, i.e., data loading and transforming to PyG graphs
    preparation_time = 0
    assert inductive in ['false', 'full']
    assert num_classes is not None

    model = PolicyClassification(dataset, None, sim_measure="cosine", DATASETPATH=DATASETPATH)

    eval_folder = Path(eval_folder) / "GNN"
    eval_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if inductive == 'false':
        with open(os.path.join(DATASETPATH, dataset, query_type, query_filename)) as f:
            data = json.load(f)
        random.Random(4).shuffle(data)
        # Use only half of the data
        #half_length = len(data) // 2
        #data = data[:half_length]
        train_data = data[:int(0.8 * len(data))]
        test_data = data[int(0.8 * len(data)):]
    else:
        with open(os.path.join(DATASETPATH, dataset, query_type, "disjoint_train.json")) as f:
            train_data = json.load(f)
        with open(os.path.join(DATASETPATH, dataset, query_type, "disjoint_test.json")) as f:
            test_data = json.load(f)
        train_data = train_data[:]
        test_data = test_data[:]


    preparation_time += time.time() - start_time

    print("Training on:", len(train_data), "queries")
    print("Evaluating on:", len(test_data), "queries")

    # Initial thresholds (0.0 or 0.5, up to you)
    thresholds = torch.tensor([0.0]*num_classes)

    results = model.train_GNN(
        train_data, test_data, thresholds=thresholds, num_classes=num_classes,
        epochs=100, train=train, eval_folder=eval_folder, inductive=inductive,
        preparation_time=preparation_time
    )

    return results

# Example usage
if __name__ == "__main__":
    dataset = 'lubm' # or 'yago'
    query_type = 'star'
    query_filename = 'Joined_Queries.json'

    BASE_DIR = os.getenv("DATASET_BASE_DIR", "BASE_DIR")
    eval_folder = os.path.join(BASE_DIR, "Datasets", dataset)
    DATASETPATH = os.path.join(BASE_DIR, "Datasets")


    results = train_GNCE(
        dataset=dataset,
        query_type=query_type,
        eval_folder=eval_folder,
        query_filename=query_filename,
        train=True,
        inductive='false',
        DATASETPATH=DATASETPATH,
        num_classes=6
    )
    print("Final Results:", results)
