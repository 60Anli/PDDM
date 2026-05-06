import argparse
import datetime
import json
import os

import torch
import yaml

from dataset_ett import get_ett_dataloader
from utils import evaluate
from Mixup增强 import train
from 多时间步和层间main_model import CSDI_ETT


parser = argparse.ArgumentParser(description="PDDM-LLM ETTm1 imputation")
parser.add_argument("--config", type=str, default="base_llm.yaml")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=50)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--data_path", type=str, default="./data/ETT_processed/ETTm1")
parser.add_argument("--raw_data_path", type=str, default="./data/ETT_raw/ETTm1.csv")
parser.add_argument("--eval_length", type=int, default=24)
parser.add_argument("--missing_ratio", type=float, default=0.0015)
parser.add_argument("--missing_pattern", type=str, default="block", choices=["random", "block"])
parser.add_argument("--target_strategy", type=str, default="block", choices=["random", "block"])
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()
print(args)

config_path = os.path.join("config", args.config)
with open(config_path, "r", encoding="utf-8-sig") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.target_strategy
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/ettm1_{current_time}/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler_params = get_ett_dataloader(
    data_path=args.data_path,
    raw_data_path=args.raw_data_path,
    batch_size=config["train"]["batch_size"],
    eval_length=args.eval_length,
    missing_ratio=args.missing_ratio,
    missing_pattern=args.missing_pattern,
    num_workers=args.num_workers,
    llm_config=config["model"].get("llm", {}),
    target_strategy=config["model"].get("target_strategy", args.target_strategy),
)

model = CSDI_ETT(config=config, device=args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load(os.path.join("./save", args.modelfolder, "model.pth")))

scaler = torch.as_tensor(scaler_params["scale_"], device=args.device).float()
mean_scaler = torch.as_tensor(scaler_params["mean_"], device=args.device).float()

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
