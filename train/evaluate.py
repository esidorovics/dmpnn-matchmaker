from collections import defaultdict
import logging
from typing import Dict, List

from .predict import predict
from data_scripts import SynergyDataLoader, StandardScaler
from models import MatchMaker
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
import torch

def pearson(y : List[float],
            pred: List[float]) -> float:
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    # print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y : List[float],
            pred: List[float]) -> float:
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    # print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def mse(y : List[float],
            pred: List[float]) -> float:
    err = mean_squared_error(y, pred)
    # print("Mean squared error is {}".format(err))
    return err

def squared_error(y : List[float],
            pred: List[float]) -> float:
    errs = []
    for i in range(len(y)):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)


def evaluate_predictions(preds: List[float],
                         targets: List[float],
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    info = logger.info if logger is not None else print

    # Compute metric
    results = {}
    results['Pearson'] = pearson(targets, preds)
    results['Spearman'] = spearman(targets, preds)
    results['MSE'] = mse(targets, preds)
    # results['Squared error'] = squared_error(targets, preds)
    return results


def evaluate(model: MatchMaker,
             data_loader: SynergyDataLoader,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             device: torch.device = 'cpu') -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler,
        device=device,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        logger=logger
    )

    return results
