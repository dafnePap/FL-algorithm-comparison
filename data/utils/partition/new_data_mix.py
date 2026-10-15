from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Dataset

def mixed_iid_sorted_partition(
    ori_dataset,
    target_dataset,
    num_clients,
    s_iid=0.0,
    dataset_name=None,        # ← add this
    transform=None,
    target_transform=None,
):
    y = np.concatenate([ds.targets for ds in ori_dataset], axis=0).astype(np.int64) #all labels
    x = np.concatenate([ds.data for ds in ori_dataset], axis=0)#all samples
    N = len(y)   #num of samplesl
    base = N // num_clients #base number of samples per client
    rem  = N %  num_clients 
    # for remaining samples
    sizes = np.array([base + (i < rem) for i in range(num_clients)], dtype=np.int64)#allows it to multiplicate with float s_iid
    m_iid = (sizes * s_iid).astype(np.int64) #number of random/IID samples that client gets
    m_non = sizes - m_iid
   
    perm = np.random.permutation(N) #shuffled list of indices of labels
    
    sorted_idx = np.argsort(y, kind="stable") #group indices by label
    
    iid_ptr = 0
    non_ptr = 0
    client_idxs = []
    """
    Example
    perm     = [7, 2, 9, 1, 5, 0, 8, 3, 6, 4]
    num_clients = 3
    m_iid   = [4, 3, 3], iid_ptr = 0
    for clinet0:    iid_part = perm[0 : 0 + 4] = [7, 2, 9, 1]
                    iid_ptr += 4 → iid_ptr = 4         
    """
    for ci in range(num_clients):
        iid_part = perm[iid_ptr : iid_ptr + m_iid[ci]]#from the shuffled indexes, the client gets miid indexes
        iid_ptr += m_iid[ci]
        non_part = sorted_idx[non_ptr : non_ptr + m_non[ci]]
        non_ptr += m_non[ci]
        idxs = np.concatenate([iid_part, non_part])
        np.random.shuffle(idxs)
        client_idxs.append(idxs)

    # build datasets + stats
    stats: Dict[int, Dict[str, object]] = {}
    datasets: List[Dataset] = []
    for cid, idxs in enumerate(client_idxs):
        stats[cid] = {"x": int(len(idxs)), "y": Counter(y[idxs].tolist())}
        datasets.append(target_dataset(
            data=x[idxs],
            targets=y[idxs],
            dataset_name=dataset_name,   # ← add this
            transform=transform,
            target_transform=target_transform,
        ))
    return datasets, stats

                
    return datasets, stats