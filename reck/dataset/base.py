from tabulate import tabulate
from functools import partial
from typing import Dict, Generator


class BaseDataset(object):
    fmt_tab = partial(tabulate, headers='firstrow', tablefmt='fancy_grid')

    @classmethod
    def from_config(cls, default_config, user_config):
        for k in user_config.keys():
            assert k in default_config, f"Unexpected key [{k}] for {cls}"
        default_config.update(user_config)
        return cls(**default_config)

    @property
    def batch_describe(self) -> Dict:
        """return the data description of this dataset
        for example:
            {
                "X": (torch.tensor, (-1, 224, 224, 3)),
                "Y": (torch.tensor, (-1, 965)),
            }
        :raise: NotImplementedError

        :return: a dict of each data field
        """
        raise NotImplementedError

    def generate_batch(self, **kwargs) -> Generator:
        raise NotImplementedError

    def help_info(self, **kwargs):
        batch_des = self.batch_describe
        assert isinstance(batch_des, dict), "Wrong return type of batch_describe"

        headers = [["name", "type", "shape"]]
        for k, v in batch_des.items():
            assert len(v) == 2
            headers.append((k, str(v[0]), str(v[1])))
        BaseDataset.fmt_tab(headers)
