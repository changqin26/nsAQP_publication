import json
import re
import os
import random
import math
import numpy as np


# Function to reset the random seed for each query
random_numbers_for_variables = {}

# New dictionary to store position-specific encodings for each variable
position_specific_encodings = {}


def reset_seed_for_query():
    random_seed = random.randint(1, 10000)
    random.seed(random_seed)


def extract_rdf_triples(node):
    triples = []
    if 'label' in node and isinstance(node['label'], str):
        label_parts = re.findall(r'(\?\w+|<[^>]+>)', node['label'])
        if len(label_parts) == 3:
            triples.append(label_parts)
    for child_key in ['left', 'right']:
        if child_key in node:
            triples.extend(extract_rdf_triples(node[child_key]))
    return triples


def encode_variable(variable, position):
    if variable not in random_numbers_for_variables:
        random_numbers_for_variables[variable] = random.uniform(0, 1)

    variable_embedding = [random_numbers_for_variables[variable]] + [1] * 50
    position_specific_encoding = [1] * 50
    variable_embedding += position_specific_encoding
    return variable_embedding


np.random.seed(0)
fixed_random_iri_embedding = list(np.random.rand(100))


def get_rdf2vec_embedding_inductive(iri, embeddings_folder):
    return fixed_random_iri_embedding, 0


def get_rdf2vec_embedding_filebased(iri, embeddings_folder):
    iri_clean = iri.strip('<>').replace('http://', 'http:||').replace('https://', 'https:||')
    iri_clean = iri_clean.replace('/', '|')
    file_path = os.path.join(embeddings_folder, f"{iri_clean}.json")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        embedding = data.get("embedding", [0] * 100)
        occurrence = data.get("occurrence", 0)
        return embedding, occurrence
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [0] * 100, 0


# Select embedding strategy
USE_INDUCTIVE = True
get_rdf2vec_embedding = get_rdf2vec_embedding_inductive if USE_INDUCTIVE else get_rdf2vec_embedding_filebased


def encode_element(element, position, embeddings_folder):
    if element.startswith("?"):
        variable_encoding = encode_variable(element, position)
        return variable_encoding
    else:
        embedding, occurrence = get_rdf2vec_embedding(element, embeddings_folder)
        embedding.append(math.log(max(occurrence, 1)))
        return embedding


def encode_triple_pattern(subject, predicate, obj, embeddings_folder):
    subject_encoding = encode_element(subject, 10, embeddings_folder)
    predicate_encoding = encode_element(predicate, 20, embeddings_folder)
    object_encoding = encode_element(obj, 30, embeddings_folder)
    triple_encoding = subject_encoding + predicate_encoding + object_encoding
    desired_length = 303
    padded_encoding = triple_encoding[:desired_length] + [0] * (desired_length - len(triple_encoding))
    return padded_encoding
