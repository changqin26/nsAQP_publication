import logging
import traceback
import math
import random
from Tree_LSTM.Tree_LSTM.TripleEncoding import extract_rdf_triples, encode_triple_pattern

# logging.basicConfig(level=logging.DEBUG, filename='function_trace.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

card_to_random_numbers = {}

type_to_random_numbers = {
    "IndependentOperator": [],
    "DependentOperator": [],
    "Xnjoin": [],
    "Fjoin": []
}

class TreeNode:
    def __init__(self, node_id, node_type, label, feature_vector, node_info_dict, left=None, right=None,
                 is_parent=False):
        self.node_id = node_id
        self.node_type = node_type
        self.label = label
        self.feature_vector = feature_vector
        self.left = left
        self.right = right
        self.is_parent = is_parent

        # Update node_info_dict with the initial feature vector
        node_info_dict[node_id] = self.feature_vector

    def __str__(self):
        return f"ID: {self.node_id}, Type: {self.node_type}, Label: {self.label}, Feature Vector: {self.feature_vector[:5]}"

def get_type_random_numbers(node_type):
    return type_to_random_numbers.get(node_type, [0] * 50)  # Default to zeros if type is not mapped

def get_log_card_features(card):
    logging.debug("Stack trace for get_log_card_features call:")
    logging.debug(''.join(traceback.format_stack()))
    if isinstance(card, list):
        logging.error(f"Received list instead of single card value: {card}")
        return [card[0]] * 100 if card else [math.log(1)] * 100  # Default to log(1) if list is empty
    try:
        card_value = max(int(card), 1)  # Avoid log(0)
        log_card_value = math.log(card_value)
        return [log_card_value] * 100
    except (TypeError, ValueError):
        logging.error(f"Invalid card value received: {card}")
        return [math.log(1)] * 100  # Default to log(1) for invalid input

def summarize_vector(vector):
    return (
        vector[:5],  # 1-5
        vector[100:105],  # 101-105
        vector[200:205],  # 201-205
    )

def build_tree(node_data, embeddings_folder, node_info_dict, query_id=None):
    global random_numbers_for_variables
    random_numbers_for_variables = {}  # Clear random numbers for each query/tree

    # Ensure the random seed is unique for each query
    if query_id is not None:
        seed = hash(query_id) % (2 ** 32)  # Generate a consistent seed based on query_id
        random.seed(seed)
    # If node_data is a list containing one JSON object, unwrap
    if isinstance(node_data, list):
        if len(node_data) == 0:
            return None
        elif len(node_data) == 1:
            node_data = node_data[0]
        else:
            # If there's more than one object in the list,
            # you must decide how to handle multiple root nodes
            raise ValueError("Expected a single-object list, got multiple objects.")
    # Check if node_data is None or empty
    if not node_data:
        return None

    # Extract basic node information
    node_id = node_data.get("id")
    node_type = node_data.get("type")
    label = node_data.get("label")
    is_parent = node_type in ["Fjoin", "Xnjoin"]

    # Extract triples using `extract_rdf_triples`
    triples = extract_rdf_triples(node_data)
    #print(f"Extracted triples for Node ID {node_id}: {triples}")

    encoded_label = None
    if node_type in ["IndependentOperator", "DependentOperator"]:
        # Use the first triple from the extracted triples, if available
        if triples:
            triple = triples[0]
            try:
                encoded_label = encode_triple_pattern(*triple, embeddings_folder)
                #print("[DEBUG] Raw encoded triple pattern:", encoded_label)

                #summarized = summarize_vector(encoded_label)
                #print(f"Encoded Label for Node ID {node_id} (Type: {node_type}): {summarized}")
            except Exception as e:
                #print(f"Error encoding label for Node ID {node_id}: {e}")
                pass
    """
        if len(label_parts) == 3:
            try:
                encoded_label = encode_triple_pattern(*label_parts, embeddings_folder)
            except Exception:
                pass
        """
    # Determine feature vector
    if encoded_label:
        feature_vector = encoded_label
    else:
        feature_vector = [0] * 303

    # Create the TreeNode object
    node = TreeNode(node_id, node_type, label, feature_vector, node_info_dict)

    if node_data.get("left"):
        #print(f"Building left child for node {node_id}")
        node.left = build_tree(node_data.get("left"), embeddings_folder, node_info_dict, query_id)

    if node_data.get("right"):
        #print(f"Building right child for node {node_id}")
        node.right = build_tree(node_data.get("right"), embeddings_folder, node_info_dict, query_id)

    # Return the constructed node
    return node
