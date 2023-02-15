<div align="center">
  <h1>reck</h1>
  <p><strong>A unified framework for recommender system attacking</strong></p>
      <p>
    <a href="https://github.com/gusye1234/reck/blob/main/todo.md">
      <img src="https://img.shields.io/badge/stability-unstable-yellow.svg">
    </a>
  </p>
</div>

### Quick Start

```python
import reck

# quickly asscess the dataset with implicit feedback
data = reck.dataset.from_config("implicit", dataset_name, need_graph=True)

# sample part of the explicit dataset as the attack data
attack_data = reck.dataset.from_config("explicit", dataset_name).partial_sample(
    user_ratio=0.2
)

# set up models
rec_model = reck.model.from_config("victim", "lightgcn", dataset=data)
attack_model = reck.model.from_config("attacker", "average", dataset=attack_data)


config = {
    "victim_data": data,
    "attack_data": attack_data,
    "victim": rec_model,
    "attacker": attack_model,
    "rec_epoch": 0,
}
workflow = reck.workflow.Normal.from_config(**config)
# run the attacking
workflow.execute()
```

### Have a look

`reck` is designed to help users use and debug interactively.

```python
import reck

# how many datasets we support?
print(reck.print_datasets())

# how many models we support?
print(reck.print_models())
```

For each component, we support a `from_config` method to instance it:

```python
import reck

dataset = reck.dataset.from_config("ml1m")
attack_model = reck.model.from_config("attack", "random")
..
```

Confused about what a component is doing? Each component in `reck` will have a `print_help` method to return the input/output information:

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

### Contribution

Install `pre-commit` first to make sure the commits you made is well-formatted:

```shell
pip install pre-commit
pre-commit install
```
