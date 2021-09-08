from data_scripts.scaler import StandardScaler
from typing import List

import torch
from tqdm import tqdm
import numpy as np

from data_scripts import SynergyDataLoader, SynergyDataset, StandardScaler
from models import MatchMaker


def predict(model: MatchMaker,
            data_loader: SynergyDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            device: torch.device = 'cpu') -> List[float]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    # for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
    for batch in data_loader:
        # Prepare batch
        batch: SynergyDataset
        # mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()
        mol_batch, features_batch, cell_batch = batch.batch_graph(), batch.features(), batch.cell_lines()
        # print(type(features_batch[0]))
        # print(model.device)
        cell_batch = torch.FloatTensor(cell_batch).to(device)


        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, cell_batch, features_batch)#, atom_descriptors_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            np.expand_dims(batch_preds, 0)
            batch_preds = scaler.inverse_transform(batch_preds).flatten()

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
