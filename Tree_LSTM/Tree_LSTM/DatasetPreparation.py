import json
import os
import random


def load_raw_dataset(directory_path):
    raw_dataset = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Assuming the files are stored as JSON
            query_id = filename.split('_')[0]  # Extract query_id from filename
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r') as file:
                    tree_data = json.load(file)
                    raw_dataset[query_id] = tree_data  # Map query_id to tree_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {filename}: {e}")
    return raw_dataset

def split_dataset_for_training(raw_dataset, fraction=0.5):
    all_keys = list(raw_dataset.keys())
    random.shuffle(all_keys)  # Shuffle to ensure randomness
    training_size = int(len(all_keys) * fraction)  # Use the given fraction for training

    # Select keys for training
    training_keys = all_keys[:training_size]

    # Create a new dataset for training
    training_dataset = {key: raw_dataset[key] for key in training_keys}
    return training_dataset

