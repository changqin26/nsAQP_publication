import random

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import json
from tqdm import tqdm
import os
import sys
import time
import shutil

# Set base directory placeholder for anonymous submission
BASE_DIR = "BASE_DIR"

# Adding edited version of pyrdf2vec to path
sys.path.append(os.path.join(BASE_DIR, "GitHub", "pyRDF2Vec", "pyrdf2vec"))

def uri_to_id(uri):
    return uri.split('/')[-1]

"""
This file manages the generation of embeddings for the different entities
"""

uri_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{<{}> ?p ?o.}}
        UNION
        {{?s <{}> ?o.}}
        UNION
        {{?s ?p <{}>.}}

        }}
        LIMIT 30000000000
        """

literal_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{?s ?p '{}'.}}

        }}
        LIMIT 3000000000000
        """

def get_embeddings(dataset_name, kg_file, entities=None, remote=True, sparql_endpoint="http://127.0.0.1:8902/sparql/"):
    GRAPH = KG(sparql_endpoint, skip_verify=True)

    timing_dict = {}

    if not remote:
        if entities is None:
            train_entities = [entity.name for entity in list(GRAPH._entities)]
            test_entities = [entity.name for entity in list(GRAPH._vertices)]
            entities = set(train_entities + test_entities)

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10, vector_size=100),
        walkers=[RandomWalker(4, max_walks=5, with_reverse=True, n_jobs=12, md5_bytes=None)],
        verbose=2, batch_mode='onefile'
    )

    occurrences = {}
    print("Calculating Occurrences")
    occurrence_from_file = True

    occurrence_start_time = time.time()

    if occurrence_from_file:
        with open(kg_file, "r") as file:
            for line in tqdm(file):
                line = line.strip().split(" ")
                s = line[0].replace("<", "").replace(">", "")
                p = line[1].replace("<", "").replace(">", "")
                o = line[2].replace("<", "").replace(">", "")
                occurrences[s] = occurrences.get(s, 0) + 1
                occurrences[p] = occurrences.get(p, 0) + 1
                occurrences[o] = occurrences.get(o, 0) + 1
    else:
        raise NotImplementedError

    with open(os.path.join(BASE_DIR, "Datasets", dataset_name, "embedding_occurrences.json"), "w") as fp:
        json.dump(occurrences, fp)

    occurrence_end_time = (time.time() - occurrence_start_time) * 1000
    n_atoms_occurrence = len(occurrences)
    time_per_atom_occurrence = occurrence_end_time / n_atoms_occurrence
    timing_dict['occurrence_total_time'] = occurrence_end_time
    timing_dict['n_atoms_occurrence'] = n_atoms_occurrence
    timing_dict['time_per_atom_occurrence'] = time_per_atom_occurrence

    folder_path = os.path.join(BASE_DIR, "Datasets", "walks")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    os.makedirs(folder_path)
    print(f"Recreated folder: {folder_path}")

    embedding_start_time = time.time()

    print("Starting to fit model")
    transformer.fit(GRAPH, entities)
    print("Finished fitting model")

    test_entities_cleaned = []
    embeddings_test = []
    occurrences_test = []

    print("Calculating Embeddings")
    for entity in tqdm(entities):
        try:
            embedding, literals = transformer.transform(GRAPH, [uri_to_id(entity)])
            test_entities_cleaned.append(entity)
            embeddings_test += embedding
            occurrences_test.append(occurrences.get(entity, 0))
        except:
            print(f"Error with entity: {entity}")
            raise

    embedding_end_time = (time.time() - embedding_start_time) * 1000
    n_atoms_embeddings = len(test_entities_cleaned)
    time_per_atom_embedding = embedding_end_time / n_atoms_embeddings
    timing_dict['embedding_total_time'] = embedding_end_time
    timing_dict['n_atoms_embeddings'] = n_atoms_embeddings
    timing_dict['time_per_atom_embedding'] = time_per_atom_embedding
    timing_dict['time_per_atom_statistic'] = time_per_atom_occurrence + time_per_atom_embedding

    with open(os.path.join(BASE_DIR, "Datasets", dataset_name, "embedding_timing.json"), "w") as fp:
        json.dump(timing_dict, fp, indent=4)

    print("Saving statistics")
    for i in tqdm(range(len(test_entities_cleaned))):
        statistics_dict = {"embedding": embeddings_test[i].tolist(), "occurrence": occurrences_test[i]}
        file_name = test_entities_cleaned[i].replace("/", "|")
        with open(os.path.join(BASE_DIR, "Datasets", dataset_name, "statistics", file_name + ".json"), "w") as fp:
            json.dump(statistics_dict, fp)

if __name__ == "__main__":
    entities = []

    with open(os.path.join(BASE_DIR, "Datasets", "lubm", "star", "Joined_Queries.json"), 'r') as f:
        queries = json.load(f)
    for query in queries:
        entities += query['x']

    entities = list(set(entities))

    print(f'Using {len(entities)} entities for RDF2Vec')

    print('Starting...')
    get_embeddings("lubm", os.path.join(BASE_DIR, "Datasets", "lubm", "graph", "lubm.ttl"), remote=True, entities=entities, sparql_endpoint="http://localhost:8890/sparql/")
