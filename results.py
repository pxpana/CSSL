import argparse
import pickle as pkl
from continuum.metrics.logger import Logger as continuum_Logger

def results(args):
    """View results from logs."""
    if args.name is None:
        raise ValueError("Please provide a name for the folder with the logger.pkl using --name")

    filename = f"logs/{args.name}/logger.pkl"

    try:
        with open(filename, "rb") as f:
            logger_dict = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found. Please check the name and path.")

    logger = continuum_Logger(list_keywords=["performance"])
    logger.logger_dict = logger_dict
    print(logger_dict["current_epoch"])
    print(logger_dict.get("current_epoch", 0))
    logger.current_task = logger_dict.get("current_task", 0)
    logger.current_epoch = logger_dict.get("current_epoch", 0)
    logger._update_dict_architecture(update_task=True)

    print(f"Metric result: {logger.my_pretty_metric}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Results from Logs")

    parser.add_argument("--name", type=str, help="Nmae of folder with results")
    args, remaining_args = parser.parse_known_args()

    results(args)
