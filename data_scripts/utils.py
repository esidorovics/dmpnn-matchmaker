from .data import SynergyDataset
from args import TrainArgs
from random import Random
from typing import List, Optional, Set, Tuple, Union
from logging import Logger
import numpy as np


def split_data(data: SynergyDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[SynergyDataset,
                                               SynergyDataset,
                                               SynergyDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and sum(sizes) == 1):
        raise ValueError('Valid split sizes must sum to 1 and must have three sizes: train, validation, and test.')

    random = Random(seed)

    if args is not None:
        train_index, val_index, test_index = \
            args.train_index, args.val_index, args.test_index

    if split_type == 'mm-split':
        train_indices = np.loadtxt(train_index, dtype=np.int)
        val_indices = np.loadtxt(val_index, dtype=np.int)
        test_indices = np.loadtxt(test_index, dtype=np.int)

        train = [data[i] for i in train_indices]
        valid = [data[i] for i in val_indices]
        test = [data[i] for i in test_indices]
        return SynergyDataset(train), SynergyDataset(valid), SynergyDataset(test)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return SynergyDataset(train), SynergyDataset(val), SynergyDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')
