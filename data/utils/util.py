import json
import math
import os
import pickle
import sys
import types
from typing import Dict, List, Tuple, Union

from path import Path
from torch.utils.data import Subset, random_split

_CURRENT_DIR = Path(__file__).parent.abspath()
_ARGS_DICT = json.load(open(_CURRENT_DIR.parent / "args.json", "r"))


def _register_mrpc_alias():
    """Register 'dataset.MRPCDataset' alias so pickles created by run_mrpc.py
    (which imported MRPCDataset as bare 'dataset') can be unpickled here."""
    if "dataset" not in sys.modules:
        try:
            from data.utils.dataset import MRPCDataset as _cls
        except ModuleNotFoundError:
            # running from within data/utils/ — fall back to sibling import
            from dataset import MRPCDataset as _cls
        m = types.ModuleType("dataset")
        m.MRPCDataset = _cls
        sys.modules["dataset"] = m


def get_dataset(
    dataset: str,
    client_id: int,
) -> Dict[str, Subset]:
    if dataset == "mrpc":
        _register_mrpc_alias()
    client_num_in_each_pickles = _ARGS_DICT["client_num_in_each_pickles"]
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    trainset = client_dataset["train"]
    valset = client_dataset["val"]
    testset = client_dataset["test"]
    return {"train": trainset, "val": valset, "test": testset}


def get_client_id_indices(
    dataset,
) -> Union[Tuple[List[int], List[int], int], Tuple[List[int], int]]:
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    with open(pickles_dir / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    if _ARGS_DICT["type"] == "user":
        return seperation["train"], seperation["test"], seperation["total"]
    else:  # NOTE: "sample"
        return seperation["id"], seperation["total"]
