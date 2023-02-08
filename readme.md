<div align="center">
  <h1>reck</h1>
  <p><strong>A unified framework for recommender system attacking</strong></p>
</div>

### Quick Start

```python
import reck

# quickly asscess the dataset
reck.default.data_help_info("ml100k")
data = reck.dataset.pre_built.PreBuilt.from_config("ml100k", verbose=True)
```



### Contribution

Install `pre-commit` first to make sure the commits you made is well-formatted:

```shell
pip install pre-commit
pre-commit install
```

