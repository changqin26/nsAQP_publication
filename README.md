# nsAQP: Neuro-Symbolic Adaptive Query Processing over Knowledge Graphs

This repository contains the implementation of nsAQP, a neuro-symbolic AQP approach that combines
representation learning and supervised learning to predict the optimal
routing policy for a given query.  It is based on Knowledge Graph Embeddings, Graph Neural Networks, Tree Long-Short Term Memory Networks
and Multi-Layer Perceptron.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Requirements

- You need to have Python 2.7 installed to run nLDE query engine. 
- You need to have Python 3 installed to train learned models. The code was tested with Python 3.10.

### Implementation for nLDE

The implementation of `nLDE` in this project is adapted from its [original source](https://github.com/maribelacosta/nlde), with modifications to support our new routing polices.

### Setup for Learned Models

This project includes the following learned model modules:

- **`GNN`** – A GNN-based policy predictor adapted from the [GNCE project](https://github.com/DE-TUM/GNCE/tree/master), using a modified version of [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec).
- **`Tree_LSTM`** – A Tree-LSTM-based classifier for routing policies using tree-structured query plans.

Each module has its own setup instructions.
```sh
$ cd GNN
$ pip install -r requirements.txt
```
You might adapt the version of the torch and torch-geometric packages to your needs,
based on your CUDA version.

Note that we have included an adapted version of the pyrdf2vec code from
https://github.com/IBCNServices/pyRDF2Vec that saves the random walks to disk for 
scaling to large graphs on systems with limited RAM.

```sh
$ cd Tree_LSTM
$ pip install -r requirements.txt
```
This module uses a Tree-LSTM to learn query-plan structures.

## Usage

### Data
- For running nLDE, we need Knowledge Graph in the .hdt format. 
- For running learned models, we except you to have a Knowledge Graph in the .ttl or .nt format, as well as it served over a SPARQL endpoint.
  
#### Example Data
The used datasets, queries, query plans, optimal policy assignment and results from the paper can be found under the folder [Datasets/lubm](https://tio.lv.tab.digital/s/9KkKoGC8jxbMFyn) and [Datasets/yago](https://tio.lv.tab.digital/s/WMkWB27ow6o7wX6).

### Best Polices Generation

All queries are first executed by nLDE using all available routing policies, and for each execution, the following statistics are recorded: execution time, number of results, number of requests, and number of intermediate results. 
Based on these metrics, the optimal routing strategies are selected using the script `select_best_policies.py`.

#### GNN
For the GNN-based model, we further expect you to have a file containing the queries you want to predict in
the following format:
```
{"x": ["http://example.org/entity1", ...], "y": [1,1,0,1,1,0], 
"query": ["SELECT * WHERE..."], 
"triples": [["http://example.org/entity1", "http://example.org/predicate1", "http://example.org/entity2"], ...]}
```

Here, "x" is the list of entities that are part of the query, "y" is a multi-label one-hot vector indicating the optimal routing policies selected for the query,
"query" is the SPARQL query, and "triples" is the list of triples that are part of the query.

Note: This structure is adapted from the data input structure used in the [GNCE project](https://github.com/DE-TUM/GNCE/tree/master). These files are under `Datasets/{dataset_name}/star/` with the name `Joined_Queries.json`.


#### Tree_LSTM
For the Tree-LSTM model, a JSON file is expected containing performance statistics for each query under different routing policies:
```
{
  "Q18109": {
    "Productivity": {
      "ExecutionTime": 0.1249,
      "Results": 9,
      "Requests": 6,
      "InterRe": 42
    },
    "ProNLJ": {
      "ExecutionTime": 0.1240,
      ...
    },
    ...
  },
  ...
}
```
Here, each key is a query ID (e.g., "Q18109"), and its value is a dictionary of routing strategies evaluated for that query.
Each strategy reports: ExecutionTime(runtime in seconds), Results(number of query answers), Requests(number of total requests) and InterRe
(intermediate results). Based on these metrics, the optimal routing strategy per query is selected and stored in a `best_policies.json` file under `Datasets/{dataset_name}/star/`.

This file provides the optimal routing strategy per query and serves as the supervised target for Tree-LSTM training.

### Embedding Generation

The first step for both `GNN` and `Tree_LSTM` is to generate embeddings for the entities involved in the queries.

This project reuses and adapts the embedding generation procedure from the [GNCE project](https://github.com/DE-TUM/GNCE).  
The script `embeddings_generator.py` performs this task.

#### Configuration

In the main function of `embeddings_generator.py`, set the following variables:

- `QUERY_FILE_PATH`: Path to the JSON file containing the queries (in the format shown above)
- `KG_FILE_PATH`: Path to the Knowledge Graph file (`.ttl` or `.nt`)
- `KG_ENDPOINT`: The SPARQL endpoint of the Knowledge Graph
- `KG_NAME`: A short identifier for your dataset

#### Output

The generated embeddings are saved under:

```
/Datasets
    /KG_NAME
        /graph
            graph.nt
        /Results
        /queries
            query_file.json
        /statistics
            /entity1.json
            /entity2.json
            ...
```

Each entity file in the `statistics/` folder contains:

- The embedding vector for that entity
- Metadata such as its frequency across queries

> These embeddings are used in both `GNN`  and `Tree_LSTM`.


### Training

GNN and Tree_LSTM models use training scripts with the same filenames located in their respective folders: 
- `policy_prediction.py`: for standard evaluation, where all entities are **seen** during training.
  - For Tree_LSTM, query plans are used as input, and routing strategies are loaded from `best_policies.json`
  - For GNN, queries and corresponding policy targets are loaded from `Joined_Queries.json`
- `policy_prediction_inductive.py`: for inductive evaluation, where entities in the test set are **unseen** during training.
  - For Tree_LSTM, the queries are located in `TreeLSTMInductive/Train/` and `.../Test/`
  - For GNN, the query files are `disjoint_train.json` and `disjoint_test.json`

They are trained on 80% of the queries and evaluated on the rest.

The best model checkpoint is saved as:
- ```Datasets/KG_NAME/Results/Model_NAME/Timestamp``` for standard evaluation.
- ```Datasets/KG_NAME/Results/Model_NAME/Inductive/Timestamp``` for inductive evaluation.

Each `Results` folder contains:
- `model.pth`: the trained model weights
- `optimizer.pth`: the optimizer state
- `runtime_training.json`: time spent per epoch 
- `runtime_evaluation.json`: time spent during inference per epoch
- `thresholds_kde.json`: class-wise thresholds computed using KDE
- `tp_fp_probabilities.json`: TP/FP prediction distributions
-  `test_metrics.json`:  test set performance metrics, including loss, precision for non-all-zero prediction, precision for all-zero prediction and its count.



## License

This project is licensed under the AGPL-3.0 license - see the LICENSE file for details.
