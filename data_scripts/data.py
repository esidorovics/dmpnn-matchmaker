import threading
from collections import OrderedDict
import csv
from logging import Logger
import pickle
from random import Random
from typing import Dict, Iterator, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from .features_generators import get_features_generator, FeaturesGenerator
from .featurization import BatchMolGraph, MolGraph
from .scaler import StandardScaler
# from .utils import load_features

# from .data import MoleculeDatapoint, MoleculeDataset
# from .scaffold import log_scaffold_stats, scaffold_split
from args import TrainArgs
# from chemprop.features import load_features, load_valid_atom_features

SMILES_TO_MOL: Dict[str, Chem.Mol] = {}
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


class SynergyDatapoint:
    def __init__(self,
                 smiles: List[str],
                 cell_line: np.ndarray,
                 target: float,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 shuffler: Random = None):

        # self.smiles = [smiles1, smiles2]
        self.drugs = [Drug(s, f, fg) for s, f, fg in zip(smiles, features, features_generator)]
        if shuffler is not None:
            shuffler.shuffle(self.drugs)
        # self.drugs = [Drug(smiles1, features, features_generator), 
        #               Drug(smiles2, features, features_generator)]
        self.cell_line = cell_line
        self.target = target
        self.weight = 1

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_target, self.raw_cell_line = self.target, self.cell_line

    @property
    def smiles(self) -> List[str]:
        return [d.smile for d in self.drugs]

    @property
    def mol(self) -> List[Chem.Mol]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = [SMILES_TO_MOL.get(s, Chem.MolFromSmiles(s)) for s in self.smiles]
        for s, m in zip(self.smiles, mol):
            SMILES_TO_MOL[s] = m
        return mol

    @property
    def features(self) -> List[np.ndarray]:
        return [d.features for d in self.drugs]

    @property
    def raw_features(self) -> List[np.ndarray]:
        return [d.raw_features for d in self.drugs]

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        self.target, self.cell_line = self.raw_target, self.raw_cell_line
        for drug in self.drugs:
            drug.reset_features()

    def set_targets(self, target: float):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.target = target

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        for d, f in zip(self.drugs, features):
            d.set_features(f)

    def set_cell_line_features(self, features: np.ndarray) -> None:
        self.cell_line = features


class Drug:
    def __init__(self, 
                 smile: str, 
                 features: np.ndarray, 
                 features_generator: List[str] = None):
        self.smile = smile
        self.features_generator = features_generator
        self.features = features

        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        if self.features_generator is not None:
            self.features = []
            for fg in self.features_generator:
                features_generator = get_features_generator(fg)

                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))
                # for H2
                elif self.mol is not None and self.mol.GetNumHeavyAtoms() == 0:
                    # not all features are equally long, so use methane as dummy molecule to determine length
                    self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
            self.features = np.array(self.features)

        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        self.raw_features = self.features

    @property
    def mol(self) -> List[Chem.Mol]:
        if self.smile in SMILES_TO_MOL:
            return SMILES_TO_MOL[self.smile]
        m = Chem.MolFromSmiles(self.smile)
        SMILES_TO_MOL[self.smile] = m
        return m

    def set_features(self, features:np.ndarray) -> None:
        self.features = features

    def reset_features(self) -> None:
        self.features = self.raw_features

class SynergyDataset(Dataset):
    def __init__(self, data: List[SynergyDatapoint]):
        self._data = data
        self._scaler = None
        self._cell_scaler = None
        self._batch_graph = None
        self._random = Random()
        self._weights = None

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        mol_graph = MolGraph(m)
                        SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def targets(self) -> List[float]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.target for d in self._data]

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features[1]) if len(self._data) > 0 and self._data[0].features is not None else None

    def cell_lines(self) -> List[List[float]]:
        """
        Returns cell lines associated with the datapoint
        """
        return [d.cell_line for d in self._data]

    def cell_line_size(self) -> int:
        """
        Returns the size of the Cell line.
        """
        return len(self._data[0].cell_line) if len(self._data) > 0 else None

    def loss_weights(self) -> None:
        return [d.weight for d in self._data]

    def calculate_weights(self) -> float:
        targets = [d.target for d in self._data]
        min_synergy = min(targets)
        for d in self._data:
            d.weight = np.log(d.target - min_synergy + np.e)

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [[d.raw_target] for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i][0])

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        for d in self._data:
            d.set_features([self._scaler.transform(f) for f in d.features])

        return self._scaler

    def normalize_cell_lines(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        if len(self._data) == 0 or self._data[0].cell_line is None:
            return None

        if scaler is not None:
            self._cell_scaler = scaler
        elif self._cell_scaler is None:        
            cell_lines = [d.cell_line for d in self._data]
            self._cell_scaler = StandardScaler(replace_nan_token=replace_nan_token).fit(cell_lines)

        for d in self._data:
            d.set_cell_line_features(self._cell_scaler.transform(d.raw_cell_line.reshape(1, -1))[0])
        return self._cell_scaler

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> List[SynergyDatapoint]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class SynergySampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""
    def __init__(self,
                 dataset: SynergyDataset,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.shuffle = shuffle
        self._random = Random(seed)
        self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[SynergyDatapoint]) -> SynergyDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = SynergyDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class SynergyDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: SynergyDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang
        self._sampler = SynergySampler(
            dataset=self._dataset,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(SynergyDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].target for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[SynergyDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(SynergyDataLoader, self).__iter__()




def get_data(path: str,
             cell_lines: str,
             args: Union[TrainArgs] = None,
             skip_none_targets: bool = False,
             shuffle_drugs: bool = True,
             seed: int = 0) -> SynergyDataset:

    dataset = pd.read_csv(path)
    targets = dataset[args.target_column].to_numpy()
    cell_line_data = np.loadtxt(cell_lines, delimiter=',')
    drug1 = dataset[args.smiles_columns[0]].to_numpy()
    drug2 = dataset[args.smiles_columns[1]].to_numpy()
    shuffler = None
    if shuffle_drugs:
        shuffler = Random(seed)

    if targets.shape[0] != cell_line_data.shape[0]:
        raise ValueError('Cell line feature line should be the same size as dataset file')

    if args.features_path is not None:
        features_data = []
        for feat_path in args.features_path:
            features_data.append(np.loadtxt(feat_path, delimiter=',', dtype=np.float32))
        features_data = np.asarray(features_data)
    else:
        features_data = None
    
    if features_data is not None and len(features_data) != 2:
        raise ValueError('Only 2 files can be provided (one for each drug)')
    features_generator = [args.features_generator] * 2

    datapoints = []
    for i, (smile1, smile2, cell, target) in enumerate(zip(drug1, drug2, cell_line_data, targets)):
        if args.datapoints_path is not None and os.path.exists(os.path.join(args.datapoints_path, f'{i}.pickle')):
            with open(os.path.join(args.datapoints_path, f'{i}.pickle'), 'rb') as f:
                dp = pickle.load(f) 
        else:
            dp = SynergyDatapoint([smile1, smile2], cell, target, 
                            features=features_data[:,i] if features_data is not None else [None, None],
                            features_generator=features_generator, shuffler=shuffler)
            if args.datapoints_path is not None:
                with open(os.path.join(args.datapoints_path, f'{i}.pickle'), 'wb') as f:
                    pickle.dump(dp, f)
        datapoints.append(dp)

    data = SynergyDataset(datapoints)
    return data
