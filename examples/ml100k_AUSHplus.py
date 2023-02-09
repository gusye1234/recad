import sys

sys.path.append("..")

import reck

reck.dataset_help_info("ml100k")
data = reck.dataset.pre_built.DatDataset.from_config("ml100k", verbose=True)

for d in data.generate_batch():
    print({k: v.dtype for k, v in d.items()})
    print({k: v.shape for k, v in d.items()})
