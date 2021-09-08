from ray import tune
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from args import HyperoptArgs
from utils import timeit, create_logger
from copy import deepcopy
from train import run_training
from ray.tune.schedulers import AsyncHyperBandScheduler
from constant import HYPEROPT_LOGGER_NAME
from data_scripts import get_data
from ax.service.ax_client import AxClient
from ray.tune.suggest.ax import AxSearch
from ax import ParameterType
import os
import shutil
# SPACE = {
#     pass
# }

@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args: HyperoptArgs) -> None:
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.save_dir, quiet=True)
    data = get_data(
        path=args.data_path,
        cell_lines=args.cell_lines,
        args=args,
        skip_none_targets=True
    )
    args.features_size = data.features_size()
    args.cell_line_size = data.cell_line_size()

    def objective(config, data=None):
        hyper_args = deepcopy(args)
        data.reset_features_and_targets()
        for k, v in config.items():
            setattr(hyper_args, k, v)
        run_training(hyper_args, data, tuning=True)

    ax = AxClient(enforce_sequential_optimization=False)
    ax.create_experiment(
        name="hyper_search",
        parameters=[
            {"name": "init_lr", "type": "range", "bounds": [1e-6, 1e-2], "log_scale": True},
            {"name": "depth", "type": "range", "bounds": [2, 6], 'value_type': 'int'},
            {"name": "batch_size", "type": "choice", "values": [2**x for x in range(3, 8)]},
            {"name": "hidden_size", "type": "choice", "values": [x*100 for x in range(3, 25)]},
            {"name": "dropoud", "type": "choice", "values": [x/20 for x in range(0,12)]},
            {"name": "mm_dropoud", "type": "choice", "values": [x/20 for x in range(0,12)]},
            {"name": "mm_in_dropoud", "type": "choice", "values": [x/20 for x in range(0,12)]},
            {"name": "dns_num_layers", "type": "range", "bounds": [1, 6], 'value_type': 'int'},
            {"name": "dsn_hidden_size", "type": "choice", "values": [x*100 for x in range(3, 25)]},
            {"name": "spn_num_layers", "type": "range", "bounds": [1, 6], 'value_type': 'int'},
            {"name": "spn_hidden_size", "type": "choice", "values": [x*100 for x in range(3, 25)]},
        ],
        objective_name="mse",
        minimize=True,
    )
    results = tune.run(
        tune.with_parameters(objective, data=data),
        num_samples = args.num_iters,
        search_alg=AxSearch(
            ax_client=ax,
            mode='min'
        ),
        # metric='mse',
        resources_per_trial={'cpu':args.resource_cpu, 'gpu': args.resource_gpu}, 
        # mode='min',
        scheduler=ASHAScheduler(
            metric='mse',
            mode='min',
            max_t=args.epochs
        )
    )
    best_trial = results.get_best_trial("mse", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mse"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["pearson"]))
    logger.info("Best trial config: {}".format(best_trial.config))
    best_checkpoint_dir = best_trial.checkpoint.value
    shutil.copytree(best_checkpoint_dir, os.path.join(args.save_dir, 'model'))

    



if __name__ == '__main__':
    hyperopt(HyperoptArgs().parse_args())












