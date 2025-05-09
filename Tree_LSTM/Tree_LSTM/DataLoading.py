# DataLoading.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from Tree_LSTM.Tree_LSTM.TreeBuildNorm import build_tree


class TreeDataset(Dataset):
    """
    On-demand tree-building dataset. For each query:
      - We store the label in __init__
      - In __getitem__, we actually build the tree.
    """

    def __init__(self, raw_data_dict, query_policy_mapping, policy_to_one_hot, embeddings_folder):
        """
        Args:
            raw_data_dict (dict): { query_id -> tree_data } from JSON
            query_policy_mapping (dict): { query_id -> str or [str] }
            policy_to_one_hot (dict): { policy_str -> list[int] }, e.g. 6-dim
            embeddings_folder (str): Path to embeddings
        """
        super().__init__()
        self.raw_data_dict = raw_data_dict
        self.embeddings_folder = embeddings_folder

        # Build a map: query_id -> label_tensor
        self.query_id_to_label = {}
        missing_queries = []  # to track query IDs not in query_policy_mapping
        for query_id, tree_data in raw_data_dict.items():
            if query_id not in query_policy_mapping:
                missing_queries.append(query_id)
                continue

            policy_list = query_policy_mapping[query_id]
            if isinstance(policy_list, str):
                policy_list = [policy_list]

            # multi-hot label
            label_size = len(next(iter(policy_to_one_hot.values())))  # e.g. 6
            aggregated_label_vector = [0] * label_size
            for policy_str in policy_list:
                if policy_str in policy_to_one_hot:
                    one_hot_vec = policy_to_one_hot[policy_str]
                    # Combine via logical OR (a or b)
                    aggregated_label_vector = [
                        (a or b) for (a, b) in zip(aggregated_label_vector, one_hot_vec)
                    ]

            label_tensor = torch.tensor(aggregated_label_vector, dtype=torch.float32)
            self.query_id_to_label[query_id] = label_tensor

        # Keep a sorted list of all valid query_ids (so we can index them)
        self.query_ids = sorted(list(self.query_id_to_label.keys()))


    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        """
        1) Look up the query_id.
        2) Retrieve raw tree data.
        3) Build the tree on-demand (call build_tree).
        4) Return (root_node, label_tensor).
        """

        query_id = self.query_ids[idx]
        tree_data = self.raw_data_dict[query_id]
        label_tensor = self.query_id_to_label[query_id]

        # Build the tree now (on demand)
        node_info_dict = {}
        root_node = build_tree(tree_data,
                               self.embeddings_folder,
                               node_info_dict,
                               query_id=query_id)


        return (root_node, label_tensor)


def get_data_loaders(
    dataset,
    batch_size=100,
    train_split=0.8,
    is_training=True,
    num_workers=0,
    pin_memory=False
):
    """
    Creates DataLoaders for training/validation from the dataset.
    """
    if is_training:
        if 0.0 < train_split < 1.0:
            train_size = int(train_split * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = dataset
            val_dataset = []
            train_size = len(dataset)
            val_size = 0
    else:
        train_dataset = []
        val_dataset = dataset
        train_size = 0
        val_size = len(dataset)

    # Collate function to combine lists of (root_node, label_tensor)
    def collate_fn(batch):
        root_nodes = []
        labels = []
        for (root_node, label_tensor) in batch:
            root_nodes.append(root_node)
            labels.append(label_tensor)
        labels_tensor = torch.stack(labels)
        return root_nodes, labels_tensor

    train_loader = None
    if train_size > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=is_training,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    return train_loader, val_loader, train_size, val_size
