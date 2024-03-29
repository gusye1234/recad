<div align="center">
  <h1>RecAD</h1>
  <p><strong>A unified library for recommender attack and defense</strong></p>
    <p>
    <a href="https://github.com/gusye1234/recad/actions?query=workflow%3Atest">
      <img src="https://github.com/gusye1234/recad/actions/workflows/demo.yaml/badge.svg">
    </a>
    <a href="https://pypi.org/project/recad/">
      <img src="https://img.shields.io/pypi/v/recad.svg">
    </a>
  </p>
</div>


RecAD is a unified library that aims to establish an open benchmark for recommender attack and defense. With just a few lines of code, you can quickly construct an attacking pipeline. 

The library currently supports the following modules:
- **Datasets**: ml1m, yelp, Amazon-game, Epinions, Book-crossing, BeerAdvocate, dianping, food, ModCloth, ratebeer, RentTheRunway. More details can be found in the `data/` folder.
- **Victim Models**: MF, LightGCN, NCF.
- **Attack Models**: Heuristic (random, average, segment, bandwagon); AUSH; AIA; Legup.

We are open to contributions and suggestions for adding more datasets and models.

## Installation

You can install the library using `pip`:

```
pip install recad
```

Or you can install it from source:

```
git clone https://github.com/gusye1234/recad.git
cd recad
pip install -e "."
```

## Quick Start

You can try running it from the command line:

```
recad_runner --attack="aush" --victim="lightgcn"
```

Alternatively, you can write your own script:

```python
from recad import dataset, model, workflow

dataset_name = "ml1m"
config = {
    "victim_data": dataset.from_config("implicit", dataset_name, need_graph=True),
    "attack_data": dataset.from_config("explicit", dataset_name).partial_sample(
        user_ratio=0.2
    ),
    "victim": model.from_config("victim", "lightgcn"),
    "attacker": model.from_config("attacker", "aush"),
    "rec_epoch": 20,
}
workflow_inst = workflow.from_config("no defense", **config)
workflow_inst.execute()
```

```python
from recad import dataset, model, workflow

dataset_name = "ml1m"
config = {
    "victim_data": dataset.from_config("implicit", dataset_name, need_graph=True),
    "attack_data": dataset.from_config("explicit", dataset_name).partial_sample(
        user_ratio=0.2
    ),
    "defense_data": dataset.from_config("explicit", dataset_name),
    "victim": model.from_config("victim", "lightgcn"),
    "attacker": model.from_config("attacker", "uba"),
    "rec_epoch": 20,
    "defender": model.from_config("defender", "PCASelectUsers"),
}

workflow_inst = workflow.from_config("defense", **config)
workflow_inst.execute()
```

## Documentation

The `recad` library consists of three main modules: `dataset`, `model`, and `workflow`.

### Dataset

You can use the `recad.dataset` module to work with datasets. Some examples of what you can do with this module are:

```python
import recad

# Print the supported datasets
print(recad.print_datasets())

# Print the configuration options for a dataset
recad.dataset_print_help("ml1m")

# Initialize a dataset with default parameters
dataset = recad.dataset.from_config("implicit", "ml1m")

# Initialize a dataset and modify the parameters
dataset = recad.dataset.from_config("implicit", "ml1m", test_batch_size=50, device="cuda")
```

### Model

You can use the `recad.model` module to work with models. Here are some examples:

```python
# Print the supported models
print(recad.print_models())

# Print the configuration options for a model
recad.model_print_help("lightgcn")

# Lazy-initialize a model with default parameters
# Not a torch.nn.Module class! Can't be used for training and inferring
victim_model = recad.model.from_config("victim", "lightgcn")

# Lazy-initialize a model and modify the parameters
# Not a torch.nn.Module class! Can't be used for training and inferring
victim_model = recad.model.from_config("victim", "lightgcn", latent_dim_rec=256, lightGCN_n_layers=2)

# Some models have parameters that are related to runtime modules, such as dataset
# `victim_model` now is a torch.nn.Module 
dataset = recad.dataset.from_config("implicit", "ml1m")
victim_model = recad.model.from_config("victim", "lightgcn", dataset=dataset).I()
```

### Workflow

The `recad.workflow` module allows you to define and execute workflows. Here is an example:

```python
# Print the supported workflows
print(recad.print_workflows())

# Print the configuration options for a workflow
recad.workflow_print_help("no defense")

# Initialize a workflow by providing the required components
config = {
        "victim_data": ..., # Your dataset for the victim model
        "attack_data": ..., # Your dataset for the attacker model
        "victim": ..., # Your victim model
        "attacker": ..., # Your attacker model
        "rec_epoch": ..., # Number of training epochs for the victim model
        "attack_epoch": ..., # Number of training epochs for the attacker model
    }
workflow_inst = workflow.from_config("no defense", **config)

# Start the workflow
workflow_inst.execute()
```

Each component instance in `recad` has a `print_help` method that provides **runtime** information about the input and output:

```python
dataset.print_help()
# (ml1m)Information:
# ╒════════════════════╤═══════════════════════════════════╕
# │ n_users            │ 5950                              │
# ╞════════════════════╪═══════════════════════════════════╡
# │ n_items            │ 3702                              │
# ├────────────────────┼───────────────────────────────────┤
# │ train_interactions │ 468649                            │
# ├────────────────────┼───────────────────────────────────┤
# │ valid_interactions │ 49390                             │
# ├────────────────────┼───────────────────────────────────┤
# │ test_interactions  │ 49494                             │
# ├────────────────────┼───────────────────────────────────┤
# │ train_dict         │ <class 'dict'> 5950               │
# ├────────────────────┼───────────────────────────────────┤
# │ valid_dict         │ <class 'dict'> 5583               │
# ├────────────────────┼───────────────────────────────────┤
# │ test_dict          │ <class 'dict'> 5677               │
# ├────────────────────┼───────────────────────────────────┤
# │ graph              │ torch torch.float32, (9652, 9652) │
# ╘════════════════════╧═══════════════════════════════════╛
# (ml1m)Batch data:
# ╒════════════════╤═════════════╤═══════════════╕
# │ name           │ type        │ shape         │
# ╞════════════════╪═════════════╪═══════════════╡
# │ users          │ torch.int64 │ batch[0~2048] │
# ├────────────────┼─────────────┼───────────────┤
# │ positive_items │ torch.int64 │ batch[0~2048] │
# ├────────────────┼─────────────┼───────────────┤
# │ negative_items │ torch.int64 │ batch[0~2048] │
# ╘════════════════╧═════════════╧═══════════════╛
```

## Contribution

To contribute to the project, make sure you have `pre-commit` installed to format your commits properly.

```bash
pip install pre-commit
pre-commit install
```
