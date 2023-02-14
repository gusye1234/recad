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

# quickly asscess the dataset
print(reck.default.dataset_print_help("ml100k"))
data = reck.dataset.pre_built.NPYDataset.from_config("ml100k", verbose=True)

victim_model = reck.model.from_config("victim", "LightGCN")
attack_model = reck.model.from_config("attack", "AUSH")
workflow = reck.workflow.from_config("normal", epoch=10)(dataset, victim_model, attack_model)

workflow.train()
workflow.evaluate()
```



### Have a look

`reck` is designed to help users use and debug interactively. 

```python
import reck

# how many datasets we support?
print(reck.show_datasets())

# how many models we support?
print(reck.show_models())

# how many attack workflows we support?
print(reck.show_workflows())
```

For each component, we support a `from_config` method to instance it:

```python
import reck

dataset = reck.dataset.from_config("ml100k")
victim_model = reck.model.from_config("victim", "LightGCN")
attack_model = reck.model.from_config("attack", "AUSH")
workflow = reck.workflow.from_config("normal")(dataset, victim_model, attack_model)
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

