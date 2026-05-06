import argparse
import datetime
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset_forecasting import get_forecasting_dataloader
from utils import evaluate_forecasting, train_forecasting


def load_forecasting_model_class():
    base_dir = Path(__file__).resolve().parent
    model_path = None
    for name in os.listdir(base_dir):
        if (
            name.endswith('.py')
            and 'main_model.py' in name
            and name != 'main_model.py'
        ):
            model_path = base_dir / name
            break
    if model_path is None:
        raise FileNotFoundError('Could not locate the multistep main model file in the project directory.')
    spec = importlib.util.spec_from_file_location('pddm_multistep_main_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CSDI_Forecasting


def safe_load_state_dict(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device_arg


def resolve_model_path(modelfolder):
    if not modelfolder:
        return ''

    if os.path.isfile(modelfolder):
        return modelfolder

    if os.path.isdir(modelfolder):
        for filename in ['model_best.pth', 'model_last.pth', 'model.pth']:
            candidate = os.path.join(modelfolder, filename)
            if os.path.exists(candidate):
                return candidate

    candidate_folder = os.path.join('save', modelfolder)
    if os.path.isdir(candidate_folder):
        for filename in ['model_best.pth', 'model_last.pth', 'model.pth']:
            candidate = os.path.join(candidate_folder, filename)
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(f'Could not resolve checkpoint from: {modelfolder}')


def build_save_folder(args):
    if args.output_dir:
        return args.output_dir

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    foldername = (
        f'./save/forecasting_{args.dataset}_sl{args.seq_len}_pl{args.pred_len}_{current_time}/'
    )
    return foldername


parser = argparse.ArgumentParser(description='PDDM long-term forecasting')
parser.add_argument('--config', type=str, default='base_forecasting.yaml')
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--dataset', type=str, default='traffic', choices=['traffic', 'electricity'])
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--modelfolder', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--task_name', type=str, default='forecasting')
parser.add_argument('--is_forecast', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=96, choices=[96, 192, 336, 720])
parser.add_argument('--features', type=str, default='M', choices=['M', 'MS', 'S'])
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--timeenc', type=int, default=0)
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--benchmark_mode', type=int, default=1)
parser.add_argument('--metric_on_original_scale', type=int, default=0)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--itr_per_epoch', type=float, default=None)
parser.add_argument('--num_workers', type=int, default=None)
parser.add_argument('--valid_epoch_interval', type=int, default=None)
parser.add_argument('--patience', type=int, default=None)
parser.add_argument('--fast_val', type=int, default=None)
parser.add_argument('--val_max_batches', type=int, default=None)
parser.add_argument('--val_num_samples', type=int, default=None)
parser.add_argument('--val_diffusion_steps', type=int, default=None)
parser.add_argument('--nsample', type=int, default=5)
parser.add_argument('--aggregate', type=str, default='mean', choices=['mean', 'median'])
parser.add_argument('--num_sample_features', type=int, default=None)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--unconditional', action='store_true')

args = parser.parse_args()
args.device = resolve_device(args.device)

if not args.root_path:
    args.root_path = os.path.join('.', 'data', args.dataset)
if not args.data_path:
    args.data_path = f'{args.dataset}.csv'

config_path = os.path.join('config', args.config)
with open(config_path, 'r', encoding='utf-8-sig') as f:
    config = yaml.safe_load(f)

if args.epochs is not None:
    config['train']['epochs'] = args.epochs
if args.batch_size is not None:
    config['train']['batch_size'] = args.batch_size
if args.lr is not None:
    config['train']['lr'] = args.lr
if args.itr_per_epoch is not None:
    config['train']['itr_per_epoch'] = args.itr_per_epoch
if args.valid_epoch_interval is not None:
    config['train']['valid_epoch_interval'] = args.valid_epoch_interval
if args.patience is not None:
    config['train']['patience'] = args.patience
if args.num_workers is not None:
    config['train']['num_workers'] = args.num_workers
if args.fast_val is not None:
    config['train']['fast_val'] = bool(args.fast_val)
if args.val_max_batches is not None:
    config['train']['val_max_batches'] = args.val_max_batches
if args.val_num_samples is not None:
    config['train']['val_num_samples'] = args.val_num_samples
if args.val_diffusion_steps is not None:
    config['train']['val_diffusion_steps'] = args.val_diffusion_steps
if args.num_sample_features is not None:
    config['model']['num_sample_features'] = args.num_sample_features

args.batch_size = config['train']['batch_size']
args.num_workers = config['train'].get('num_workers', 0)
args.llm_config = config.get('model', {}).get('llm', {})
args.target_strategy = config['model'].get('target_strategy', 'test')

benchmark_mode = bool(args.benchmark_mode)
metric_on_original_scale = bool(args.metric_on_original_scale)
if benchmark_mode and args.dataset in {'traffic', 'electricity'} and args.features != 'M':
    print(
        f"Benchmark mode is enabled for {args.dataset}. "
        f"Overriding features='{args.features}' to features='M' for full multivariate forecasting."
    )
    args.features = 'M'

config['model']['task_name'] = args.task_name
config['model']['is_forecast'] = bool(args.is_forecast)
config['model']['is_unconditional'] = args.unconditional
config['model']['target_strategy'] = 'test'
config['model']['seq_len'] = args.seq_len
config['model']['pred_len'] = args.pred_len
config['model']['benchmark_mode'] = benchmark_mode

set_seed(args.seed)

print(json.dumps(config, indent=4))
print(f'Using device: {args.device}')
print(f"Dataset file: {os.path.join(args.root_path, args.data_path)}")
print(
    f"Validation mode: {'FAST_VAL' if config['train'].get('fast_val', False) else 'FULL_VAL'}"
)
print(f'Features mode: {args.features}')
print(
    f"Metric scale: {'ORIGINAL_SCALE' if metric_on_original_scale else 'SCALED_BENCHMARK'}"
)

foldername = build_save_folder(args)
os.makedirs(foldername, exist_ok=True)

train_loader, valid_loader, test_loader, scaler, mean_scaler, target_dim = get_forecasting_dataloader(
    args, device=args.device
)
config['model']['target_dim'] = target_dim

with open(os.path.join(foldername, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {
            'args': vars(args),
            'config': config,
        },
        f,
        indent=4,
    )

CSDI_Forecasting = load_forecasting_model_class()
model = CSDI_Forecasting(config, args.device, target_dim=target_dim).to(args.device)

if args.modelfolder:
    checkpoint_path = resolve_model_path(args.modelfolder)
    safe_load_state_dict(model, checkpoint_path, args.device)
    train_info = {
        'best_model_path': checkpoint_path,
        'last_model_path': checkpoint_path,
        'best_epoch': -1,
        'best_valid_loss': float('nan'),
    }
else:
    train_info = train_forecasting(
        model,
        config['train'],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        val_nsample=args.nsample,
    )
    checkpoint_path = train_info['best_model_path'] or train_info['last_model_path']
    if checkpoint_path:
        safe_load_state_dict(model, checkpoint_path, args.device)

val_metrics = evaluate_forecasting(
    model,
    valid_loader,
    nsample=config['train']['val_num_samples'] if config['train'].get('fast_val', False) else args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    split_name='val',
    aggregate=args.aggregate,
    max_batches=config['train']['val_max_batches'] if config['train'].get('fast_val', False) else None,
    diffusion_steps=config['train']['val_diffusion_steps'] if config['train'].get('fast_val', False) else None,
    mode_name='FAST_VAL' if config['train'].get('fast_val', False) else 'FULL_VAL',
    metric_on_original_scale=metric_on_original_scale,
)
test_metrics = evaluate_forecasting(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    split_name='test',
    aggregate=args.aggregate,
    mode_name='FULL_TEST',
    metric_on_original_scale=metric_on_original_scale,
)

summary = {
    'dataset': args.dataset,
    'task_name': args.task_name,
    'seq_len': args.seq_len,
    'pred_len': args.pred_len,
    'features': args.features,
    'metric_on_original_scale': metric_on_original_scale,
    'target_dim': target_dim,
    'checkpoint': checkpoint_path,
    'train_info': train_info,
    'val': val_metrics,
    'test': test_metrics,
}

summary_path = os.path.join(foldername, 'summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4)

print('\nForecasting summary')
print(json.dumps(summary, indent=4))