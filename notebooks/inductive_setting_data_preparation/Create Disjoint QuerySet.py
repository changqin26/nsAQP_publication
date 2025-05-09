#!/usr/bin/env python
# coding: utf-8

import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Configuration ===
BASE_DIR = "BASE_DIR"  # Placeholder for anonymous submission
DATASET_NAME = "lubm"  # or "yago"
QUERY_TYPE = "star"
TRAIN_SPLIT_PROB = 0.002  # Default: 0.002 for lubm/star

# === Paths ===
QUERY_FILE_PATH = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, QUERY_TYPE, "Joined_Queries.json")
DISJOINT_TRAIN_PATH = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, QUERY_TYPE, "disjoint_train.json")
DISJOINT_TEST_PATH = os.path.join(BASE_DIR, "Datasets", DATASET_NAME, QUERY_TYPE, "disjoint_test.json")

# === Load input data ===
with open(QUERY_FILE_PATH) as f:
    data = json.load(f)

# === Splitting logic ===
train_data = []
test_data = []
train_entities = set()
test_entities = set()

random.shuffle(data)

for query in tqdm(data):
    if len(train_data) == 0:
        train_data.append(query)
        train_entities.update([a[2] for a in query["triples"] if not "?" in a[2]])
        continue
    if len(train_data) < TRAIN_SPLIT_PROB * len(data):
        train_data.append(query)
        train_entities.update([a[2] for a in query["triples"] if not "?" in a[2]])
        continue

    entity_list = [a[2] for a in query["triples"] if not "?" in a[2]]
    if any(entity in train_entities for entity in entity_list):
        if all(entity not in test_entities for entity in entity_list):
            train_data.append(query)
            train_entities.update(entity_list)
    else:
        test_data.append(query)
        test_entities.update(entity_list)

# === Check for overlap ===
overlap_found = any(e in train_entities for e in test_entities)
if overlap_found:
    print("Error: Overlapping entities found in disjoint split!")
else:
    print("No overlap between train and test entities.")

# === Summary ===
print("Summary of disjoint split:")
print(f"- Training queries: {len(train_data)}")
print(f"- Test queries: {len(test_data)}")

# === Plot query size distributions ===
plt.figure()
plt.hist([len(q["triples"]) for q in train_data], alpha=0.6, label='Train')
plt.hist([len(q["triples"]) for q in test_data], alpha=0.6, label='Test')
plt.legend()
plt.title("Query Triple Count Distribution")
plt.xlabel("Number of triples")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# === Save output ===
with open(DISJOINT_TRAIN_PATH, "w") as f:
    json.dump(train_data, f)
with open(DISJOINT_TEST_PATH, "w") as f:
    json.dump(test_data, f)

print(f" Saved to:\n- {DISJOINT_TRAIN_PATH}\n- {DISJOINT_TEST_PATH}")
