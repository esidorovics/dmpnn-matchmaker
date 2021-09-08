from typing import List, Union

from rdkit import Chem
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
from args import TrainArgs
from data_scripts import BatchMolGraph
from .nn_utils import initialize_weights
from .mpn import MPN



def concat(embedding1, embedding2):
    if torch.rand(1)[0] < 0.5:
        return torch.cat((embedding1, embedding2), 1)
    return torch.cat((embedding2, embedding1), 1)


def maxpooling(embedding1, embedding2):
    emb = torch.stack((embedding1, embedding2))
    return torch.max(emb, 0)[0]
    

def sum(embedding1, embedding2):
    return embedding1 + embedding2


def get_agg_func(embedding, dsn_layer):
    if embedding == 'concat':
        return concat, dsn_layer*2
    elif embedding == 'maxpooling':
        return maxpooling, dsn_layer
    elif embedding == 'sum':
        return sum, dsn_layer


class MatchMaker(nn.Module):
    def __init__(self, args: TrainArgs):
        super(MatchMaker, self).__init__()
        self.create_encoder(args)
        dsn_layers = list(map(int, args.dsn_architecture.split('-')))
        if 'embedding_agg' not in vars(args):
            args.embedding_agg = 'deterministic_concat'
        if args.dsn_output is not None:
            dsn_layers.append(args.dsn_output)

        self.embedding_agg, spn_input_size = get_agg_func(args.embedding_agg, dsn_layers[-1])
        spn_layers = list(map(int, args.spn_architecture.split('-')))
        spn_layers.insert(0, spn_input_size)

        self.create_dsn(args, dsn_layers)
        self.create_spn(args, spn_layers)
        initialize_weights(self)
    
    def create_encoder(self, args:TrainArgs) -> None:
        self.encoder = MPN(args)

    def create_dsn(self, args: TrainArgs,
                         dsn_layers: List[int]) -> None:
        first_layer_dim = args.cell_line_size
        if args.use_input_features:
            first_layer_dim += args.features_size
        if not args.features_only:
            first_layer_dim += args.hidden_size
        
        dsn_layers.insert(0, first_layer_dim)
        dropout = nn.Dropout(args.mm_dropout)
        in_dropout = nn.Dropout(args.mm_in_dropout)
        
        dsn = []
        for i, size in enumerate(dsn_layers[1:], 1):
            dsn.append(nn.Linear(dsn_layers[i-1], size))
            if i < len(dsn_layers)-1 :
                dsn.append(nn.ReLU())
                if i == 1:
                    dsn.append(in_dropout)
                else:
                    dsn.append(dropout)
        self.dsn = nn.Sequential(*dsn)
      
 
    def create_spn(self, args: TrainArgs,
                         spn_layers: List[int]) -> None:
        dropout = nn.Dropout(args.mm_dropout)
        
        spn = []
        for i, size in enumerate(spn_layers[1:], 1):
            spn.extend([nn.Linear(spn_layers[i-1], size),
                        nn.ReLU()])
            if i == len(spn_layers)-1:
                spn.append(nn.Linear(size, 1))
            else:
                spn.append(dropout)
        self.spn = nn.Sequential(*spn)

    

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                cell_batch: List[np.ndarray],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        
        drug_encodings = self.encoder(batch, features_batch)
        x1 = torch.cat((drug_encodings[0], cell_batch), 1)
        x2 = torch.cat((drug_encodings[1], cell_batch), 1)
        embedding1 = self.dsn(x1)
        embedding2 = self.dsn(x2)
        combination = self.embedding_agg(embedding1, embedding2)
        out = self.spn(combination)
        return out.flatten()

