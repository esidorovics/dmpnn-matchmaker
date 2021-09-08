from typing import List, Optional, Tuple
from typing_extensions import Literal
import os
import torch
from tap import Tap
import re


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = ['drug1', 'drug2']
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 2
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    result_dir: str = 'results'
    """Result directory"""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    no_cell_line_scaling: bool = False
    """Turn off scaling of cell line features."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    mode: str = 'common'
    no_input_shuffle: bool = False

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @property
    def shuffle_input_drugs(self) -> bool:
        return not self.no_input_shuffle

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler` to the additional molecule-level features."""
        return not self.no_features_scaling

    @property
    def cell_line_scaling(self) -> bool:
        """Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler` to the additional molecule-level features."""
        return not self.no_cell_line_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))

    def process_args(self) -> None:
        # Load checkpoint paths

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        if self.smiles_columns is None:
            self.smiles_columns = [None] * self.number_of_molecules
        elif len(self.smiles_columns) != self.number_of_molecules:
            raise ValueError('Length of smiles_columns must match number_of_molecules.')

        # # Validate atom descriptors
        # if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
        #     raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
        #                      'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        dir_names = [name for name in os.listdir(self.result_dir) 
                        if os.path.isdir(os.path.join(self.result_dir, name))]

        iter = [re.findall(r'\d+', dname)[0] for dname in dir_names if re.findall(r'\d+', dname)]
        if len(iter) > 0:
            n = max(list(map(int, iter)))+1
        else:
            n = 0
        self.save_dir = os.path.join(self.result_dir, str(n).zfill(3)+'-'+self.mode)
        os.makedirs(self.save_dir)





class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_path: str = 'data/small_DrugCombData.csv'
    """Path to data CSV file."""
    cell_lines: str = 'data/small_cell_line_gex.csv'
    """Path to the cell line features"""

    target_column: List[str] = 'loewe'
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    split_type: Literal['random', 'mm-split'] = 'mm-split'
    """Method of splitting the data into train/val/test."""
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    """Split proportions for train/validation/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    train_index: str = None
    """Which fold to use as val for leave-one-out cross val."""
    val_index: str = None
    """Which fold to use as val for leave-one-out cross val."""
    test_index: str = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 1000
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000

    datapoints_path: str = None
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    mpn_shared: bool = True
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0
    """Dropout probability."""
    mm_dropout: float = 0.5
    mm_in_dropout: float = 0.2
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    dsn_architecture: str = '2048-4096-2048'
    dsn_output: int = None
    spn_architecture: str = '2048-1024'
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm', 'attention'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    embedding_agg: Literal['concat', 'maxpooling', 'sum'] = 'concat'

    # Training arguments
    epochs: int = 1000
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""

    mode: str = 'train'

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None
        self._cell_line_size = None
        self._dsn_layers = None
        self._spn_layers = None



    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return self.features_generator is not None or self.features_path is not None

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def cell_line_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._cell_line_size

    @cell_line_size.setter
    def cell_line_size(self, features_size: int) -> None:
        self._cell_line_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size


    
    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_dir  # Prevents the temporary directory from being deleted upon function return

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        if self.datapoints_path is not None and not os.path.exists(self.datapoints_path):
            os.makedirs(self.datapoints_path)

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        # Test settings
        if self.test:
            self.epochs = 0


class HyperoptArgs(TrainArgs):
    mode: str = 'hyper-opt'
    num_iters: int = 150
    resource_gpu: float = 1
    resource_cpu: int = 3
