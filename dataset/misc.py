import torch
from typing import Dict, List


def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in batch_data:
        if torch.is_tensor(batch_data[key][0]):
            batch_data[key] = torch.stack(batch_data[key])
    return batch_data