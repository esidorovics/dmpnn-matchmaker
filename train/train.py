import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from args import TrainArgs
from data_scripts import SynergyDataLoader, SynergyDataset
from models import MatchMaker
from models import NoamLR, compute_gnorm, compute_pnorm


def train(model: MatchMaker,
          data_loader: SynergyDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None
          ) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    # for batch in tqdm(data_loader, total=len(data_loader), leave=False):
    for batch in data_loader:
        # Prepare batch

        batch: SynergyDataset
        mol_batch, features_batch, cell_batch, target_batch, weights= \
            batch.batch_graph(), batch.features(), batch.cell_lines(), batch.targets(), batch.loss_weights()
        cell_batch = torch.FloatTensor(cell_batch).to(args.device)
        weights = torch.FloatTensor(weights).to(args.device)

        # mask = torch.Tensor([x is not None for x in target_batch])
        # targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # Run model
        # model.zero_grad()
        optimizer.zero_grad()
        preds = model(mol_batch, cell_batch, features_batch)
        # Move tensors to correct device
        # mask = mask.to(preds.device)
        targets = torch.Tensor(target_batch).to(preds.device)
        # class_weights = torch.ones(targets.shape, device=preds.device)

        loss = loss_func(preds, targets) * weights
        loss = loss.mean() #.sum() / mask.sum()
        loss_sum += loss.item()

        iter_count += 1

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)
        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            lrs = [0, 1]
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter


