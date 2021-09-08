from args import TrainArgs
from data_scripts.data import get_data
from utils import timeit, create_logger
from constant import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
import os
import sys
from train import run_training



@timeit(logger_name=TRAIN_LOGGER_NAME)
def train(args: TrainArgs) -> None:
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')
    debug('Args')
    debug(args)

    args.save(os.path.join(args.save_dir, 'args.json'))
    debug('Loading data')
    data = get_data(
        path=args.data_path,
        cell_lines=args.cell_lines,
        args=args,
        skip_none_targets=True, 
        shuffle_drugs=args.shuffle_input_drugs, 
        seed=args.seed
    )
    args.features_size = data.features_size()
    args.cell_line_size = data.cell_line_size()
    args.seed = 0
    # data.reset_features_and_targets()
    model_scores = run_training(args, data, logger)




if __name__ == '__main__':
    train(TrainArgs().parse_args())