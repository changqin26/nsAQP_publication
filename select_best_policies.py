import json
import os
import re
from collections import defaultdict

BASE_DIR = "BASE_DIR"

# Define the paths for all 6 folders
folders = {
    'Productivity': os.path.join(BASE_DIR, 'Productivity'),
    'ProNLJ': os.path.join(BASE_DIR, 'ProNLJ'),
    'Ticket': os.path.join(BASE_DIR, 'Ticket'),
    'TpNLJ': os.path.join(BASE_DIR, 'TpNLJT'),
    'NoPolicy': os.path.join(BASE_DIR, 'NoPolicy'),
    'MiniOutputFirst': os.path.join(BASE_DIR, 'MiniOutput')
}

# Regex patterns to match valid line formats
valid_pattern_1 = re.compile(r"^\('Plan order:', \{.*\}\)$")
valid_pattern_2 = re.compile(r'^Q\d+\.sparql\s+(None|\d+\.\d+)\s+\d+\.\d+\s+\d+\s+\d+\s+\d+$')

invalid_files = []

# Phase 1: Clean files and identify invalid ones
for policy, folder in folders.items():
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                if len(lines) < 1:
                    continue

                first_line = lines[0].strip()
                if not first_line and len(lines) > 1:
                    first_line_index = 1
                    first_line = lines[first_line_index].strip()
                else:
                    first_line_index = 0

                if valid_pattern_1.match(first_line):
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.writelines(lines[first_line_index + 1:])
                elif not valid_pattern_2.match(first_line):
                    invalid_files.append(file_path)
        except Exception:
            continue

# Phase 2: Load and process all valid query files
query_data = defaultdict(lambda: defaultdict(list))
best_policies = defaultdict(dict)

def calculate_threshold(min_time):
    if min_time <= 5:
        return 0.30
    elif min_time <= 59:
        return max(0.30 - 0.005 * min_time, 0.05)
    return 0

for policy, folder in folders.items():
    for file_name in os.listdir(folder):
        match = re.match(r'^Q(\d+)_\w+\.txt$', file_name)
        if match:
            query_number = int(match.group(1))
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                if len(lines) < 1:
                    continue

                first_line = lines[0].strip()
                if not first_line and len(lines) > 1:
                    line_to_process = lines[1]
                else:
                    line_to_process = lines[0]

                parts = line_to_process.strip().split()
                if len(parts) >= 6:
                    second_value = None if parts[1] == 'None' else float(parts[1])
                    third_value = float(parts[2])
                    fourth_value = int(parts[3])
                    requests = int(parts[4])
                    inter_re = int(parts[5])

                    if second_value is not None:
                        query_data[query_number][policy].append(
                            (second_value, third_value, fourth_value, requests, inter_re)
                        )

# Phase 3: Evaluate policies by categories
for query_number, policy_data in query_data.items():
    third_values = []
    fourth_values = []
    policies = {}

    for policy, values in policy_data.items():
        for second_value, third_value, fourth_value, requests, inter_re in values:
            third_values.append((policy, third_value, fourth_value, requests, inter_re))
            fourth_values.append(fourth_value)
            policies[policy] = {
                'ExecutionTime': third_value,
                'Results': fourth_value,
                'Requests': requests,
                'InterRe': inter_re
            }

    if not fourth_values:
        continue

    # Category 1: All fourth values are the same
    if len(set(fourth_values)) == 1:
        min_time = min(third_value for _, third_value, _, _, _ in third_values)
        threshold = calculate_threshold(min_time)
        best_policy = [
            policy for policy, third_value, _, _, _ in third_values
            if (third_value - min_time) / min_time < threshold
        ]
        best_policies[f"Q{query_number}"] = {policy: policies[policy] for policy in best_policy}
        continue

    # Category 2: All third values in [59, 61] and fourth values differ
    all_third_in_range = all(59 <= third_value <= 61 for _, third_value, _, _, _ in third_values)
    if all_third_in_range:
        max_results = max(policy['Results'] for policy in policies.values())
        selected_policies_dict = {
            policy_name: policy for policy_name, policy in policies.items()
            if 59 <= policy['ExecutionTime'] < 61 and
               abs(max_results - policy['Results']) / max_results < 0.10
        }
        if selected_policies_dict:
            best_policies[f"Q{query_number}"] = selected_policies_dict
        continue

    # Category 3: At least two fourth values are the same and larger than others
    max_fourth_value = max(fourth_values)
    max_count = fourth_values.count(max_fourth_value)
    if max_count >= 2 and all(f < max_fourth_value for f in fourth_values if f != max_fourth_value):
        filtered_third_values = [
            (policy, third_value) for policy, third_value, fourth_value, _, _ in third_values
            if fourth_value == max_fourth_value
        ]
        min_time = min(third_value for _, third_value in filtered_third_values)
        threshold = calculate_threshold(min_time)
        best_policy = [
            policy for policy, third_value in filtered_third_values
            if (third_value - min_time) / min_time < threshold
        ]
        best_policies[f"Q{query_number}"] = {policy: policies[policy] for policy in best_policy}
        continue

    # Category 5: At least one policy has third value > 61
    has_value_above_61 = any(third_value > 61 for _, third_value, _, _, _ in third_values)
    if has_value_above_61:
        max_results = max(policy['Results'] for policy in policies.values())
        selected_policies_dict = {
            policy_name: policy for policy_name, policy in policies.items()
            if policy['ExecutionTime'] > 61 and
               abs(max_results - policy['Results']) / max_results < 0.10
        }
        if selected_policies_dict:
            best_policies[f"Q{query_number}"] = selected_policies_dict

# Save the results
output_path = os.path.join(BASE_DIR, "best_policies.json")
with open(output_path, 'w') as json_file:
    json.dump(best_policies, json_file, indent=4)
