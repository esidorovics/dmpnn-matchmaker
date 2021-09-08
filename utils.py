from typing import Any, Callable, List, Tuple, Union
from functools import wraps
from datetime import timedelta
import logging
import os
from time import time

from models import MatchMaker
from argparse import Namespace
import torch
import logging
from data_scripts import StandardScaler
from args import TrainArgs

def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """
    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """
        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')

            return result

        return wrap

    return timeit_decorator

    pass

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def save_checkpoint(path: str,
                    model: MatchMaker,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    cell_line_scaler: StandardScaler = None,
                    args: TrainArgs = None) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None,
        'cell_line_scaler': {
            'means': cell_line_scaler.means,
            'stds': cell_line_scaler.stds
        } if cell_line_scaler is not None else None

    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    device: torch.device = None,
                    logger: logging.Logger = None) -> MatchMaker:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state['args']), skip_unsettable=True)
    loaded_state_dict = state['state_dict']

    if device is not None:
        args.device = device

    # Build model
    model = MatchMaker(args)
    # model_state_dict = model.state_dict()

    # # Skip missing parameters and parameters of mismatched size
    # pretrained_state_dict = {}
    # for loaded_param_name in loaded_state_dict.keys():
    #     # Backward compatibility for parameter names
    #     if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
    #         param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
    #     else:
    #         param_name = loaded_param_name

    #     # Load pretrained parameter, skipping unmatched parameters
    #     if param_name not in model_state_dict:
    #         info(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
    #     elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
    #         info(f'Warning: Pretrained parameter "{loaded_param_name}" '
    #              f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
    #              f'model parameter of shape {model_state_dict[param_name].shape}.')
    #     else:
    #         debug(f'Loading pretrained parameter "{loaded_param_name}".')
    #         pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    # model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(loaded_state_dict)

    if args.cuda:
        debug('Moving model to cuda')
    model = model.to(args.device)

    return model
