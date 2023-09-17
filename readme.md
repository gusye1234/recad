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

RecAD is a unified library aiming at establishing an open benchmark for recommender attack and defense. With a few line of codes, you can quickly construct a attacking pipeline. The supported modules currently include:

##### Datasets

* ml1m
* yelp
* Amazon-game
* Epinions
* Book-crossing
* BeerAdvocate
* dianping
* food
* ModCloth
* ratebeer
* RentTheRunway
* ...

##### Victim Models

* MF
* LightGCN
* NCF

##### Attack Models

* Heuristic: random, average, segment, bandwagon
* AUSH
* AIA
* Legup

## Install 

```
git clone https://github.com/gusye1234/recad.git
cd recad
pip install -e "."
```

## Quick Start

Try it from command line:
```
cd example
python from_command.py --attack="aush" --victim="lightgcn"
```

Or you can write your own script:
```python
from recad import dataset, model, workflow

dataset_name = "ml1m"
config = {
    # quickly asscess the dataset with implicit feedback
    "victim_data": dataset.from_config("implicit", dataset_name, need_graph=True),
    # sample part of the explicit dataset as the attack data
    "attack_data": dataset.from_config("explicit", dataset_name).partial_sample(
        user_ratio=0.2
    ),
    # set up models config, and later will be instantiated in workflow
    "victim": model.from_config("victim", "lightgcn"),
    "attacker": model.from_config("attacker", "aush"),
    "rec_epoch": 20,
}
workflow = workflow.from_config("no defense", **config)
# run the attacking
workflow.execute()
```

## Docs

`recad` is designed to help users use and debug interactively.

```python
import recad

# how many datasets we support?
print(recad.print_datasets())

# how many models we support?
print(recad.print_models())
```

For each component, we support a `from_config` method:

```python
import recad

dataset = recad.dataset.from_config("ml1m")

# lazy init, can't function
attack_model = recad.model.from_config("attack", "random")
# actually init
attack_model = recad.model.from_config("attack", "random", dataset=dataset).I()
..
```

Confused about what a component is doing? Each component in `recad` will have a `print_help` method to return the input/output information:

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

Install `pre-commit` first to make sure the commits you made is well-formatted:

```shell
pip install pre-commit
pre-commit install
```
