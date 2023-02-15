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

dataset = reck.dataset.from_config("yelp")
attack_model = reck.model.from_config("attack", "random")
..
```

Confused about what a component is doing? Each component in `reck` will have a `print_help` method to return the input/output information:

```python
dataset.print_help()
# ╒═══════════════╤═══════════════╤══════════════╕
# │ name          │ type          │ shape        │
# ╞═══════════════╪═══════════════╪══════════════╡
# │ rating_mat    │ torch.float64 │ (943, ?~500) │
# ├───────────────┼───────────────┼──────────────┤
# │ rating_mask   │ torch.float64 │ (943, ?~500) │
# ├───────────────┼───────────────┼──────────────┤
# │ keep_rate_net │ torch.float32 │ ()           │
# ╘═══════════════╧═══════════════╧══════════════╛
```

### Contribution

Install `pre-commit` first to make sure the commits you made is well-formatted:

```shell
pip install pre-commit
pre-commit install
```
