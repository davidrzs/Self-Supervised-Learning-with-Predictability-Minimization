import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_corr() -> argparse.Namespace:
    """Parses arguments for correlation analysis.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add knn args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--pretrained_checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)

    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
