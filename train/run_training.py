from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
# from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from torch.optim import Adam, Optimizer
from models import MatchMaker
from models import NoamLR
from .train import train
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from args import TrainArgs
from constant import MODEL_FILE_NAME
from data_scripts import SynergyDataLoader, SynergyDataset, get_data, split_data
from utils import load_checkpoint, save_checkpoint
from ray import tune

# from data_scripts import get_class_sizes, set_cache_graph, split_data
# from chemprop.models import MoleculeModel
# from chemprop.nn_utils import param_count
# from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
#     save_checkpoint, save_smiles_splits


def run_training(args: TrainArgs,
                 data: SynergyDataset,
                 logger: Logger = None,
                 tuning: bool = None) -> Dict[str, List[float]]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, 
                                                 sizes=args.split_sizes, seed=args.seed, 
                                                 num_folds=args.num_folds, args=args, logger=logger)
    train_data.calculate_weights()

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.cell_line_scaling:
        cell_scaler = train_data.normalize_cell_lines(replace_nan_token=0)
        val_data.normalize_cell_lines(cell_scaler)
        test_data.normalize_cell_lines(cell_scaler)
    else:
        cell_scaler = None

    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    scaler = None#train_data.normalize_targets()
    

    loss_func = nn.MSELoss(reduction='none')

        # Create data loaders
    train_data_loader = SynergyDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = SynergyDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_data_loader = SynergyDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    try:
        writer = SummaryWriter(log_dir=args.save_dir)
    except:
        writer = SummaryWriter(logdir=args.save_dir)

    
    if args.checkpoint_path is not None:
        debug(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(args.checkpoint_path)
    else:
        debug(f'Building model')
        model = MatchMaker(args)
        model.to(args.device)

    debug(model)

    optimizer = Adam(model.parameters())#, lr=args.init_lr)#, weight_decay=0)
    # scheduler = None
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr]
    )

    best_score = float('inf') 
    best_epoch, n_iter = 0, 0
    patience = 100
    patience_level=0
    for epoch in range(args.epochs):
        debug(f'Epoch {epoch}')

        n_iter = train(
            model=model,
            data_loader=train_data_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
            logger=logger,
            writer=writer
        )
        val_scores = evaluate(
            model=model,
            data_loader=val_data_loader,
            scaler=scaler,
            logger=logger,
            device=args.device
        )

        for metric, score in val_scores.items():
            debug(f'Validation {metric} = {score:.6f}')
            writer.add_scalar(f'validation_{metric}', score, n_iter)

        patience_level += 1
        if tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                save_checkpoint(path, model, scaler, features_scaler, cell_scaler, args)
            tune.report(mse=val_scores['MSE'], pearson=val_scores['Pearson'])

        if val_scores['MSE'] < best_score:
            best_score, best_epoch = val_scores['MSE'], epoch
            save_checkpoint(os.path.join(args.save_dir, MODEL_FILE_NAME), model, scaler, features_scaler, cell_scaler, args)
            patience_level = 0
        if patience_level > patience:
            break

    
    # Evaluate on test set using model with best validation score
    info(f'Model best validation loss = {best_score:.6f} on epoch {best_epoch}')
    model = load_checkpoint(os.path.join(args.save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)
    test_preds = predict(
        model=model,
        data_loader=test_data_loader,
        scaler=scaler,
        device=args.device
    )
    test_scores = evaluate_predictions(
        preds=test_preds,
        targets=test_data.targets(),
        logger=logger
    )
    for metric, scores in test_scores.items():
        info(f'Model test {metric} = {scores:.6f}')
        writer.add_scalar(f'test_{metric}', scores, 0)

    writer.close()
    test_preds_dataframe = pd.DataFrame(data={
                                        'Synergy': test_data.targets(),
                                        'Predictions': test_preds})

    test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)
