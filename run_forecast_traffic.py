import argparse
import subprocess
import sys


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


parser = argparse.ArgumentParser(description="Run PDDM forecasting on traffic for all horizons.")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--root_path", type=str, default="./data/traffic")
parser.add_argument("--data_path", type=str, default="traffic.csv")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--nsample", type=int, default=5)
parser.add_argument("--num_sample_features", type=int, default=None)
parser.add_argument("--fast_val", type=int, default=None)
args = parser.parse_args()

for pred_len in [96, 192, 336, 720]:
    command = [
        sys.executable,
        "exe_forecasting.py",
        "--config",
        args.config,
        "--dataset",
        "traffic",
        "--root_path",
        args.root_path,
        "--data_path",
        args.data_path,
        "--device",
        args.device,
        "--seq_len",
        "96",
        "--pred_len",
        str(pred_len),
        "--nsample",
        str(args.nsample),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.num_sample_features is not None:
        command.extend(["--num_sample_features", str(args.num_sample_features)])
    if args.fast_val is not None:
        command.extend(["--fast_val", str(args.fast_val)])
    run_command(command)
