import numpy as np
import os
import json
from contextlib import contextmanager
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)


def compute_validation_loss(model, valid_loader):
    """Average diffusion loss on the validation split."""
    model.eval()
    avg_loss_valid = 0.0

    with torch.no_grad():
        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                loss = model(valid_batch, is_train=0)
                avg_loss_valid += loss.item()
                it.set_postfix(
                    ordered_dict={
                        "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                    },
                    refresh=False,
                )

    return avg_loss_valid / max(len(valid_loader), 1)


@contextmanager
def temporary_diffusion_steps(model, num_steps=None):
    """Temporarily reduce diffusion steps for fast validation only."""
    if num_steps is None:
        yield
        return

    original_num_steps = model.num_steps
    model.num_steps = min(int(num_steps), int(original_num_steps))
    try:
        yield
    finally:
        model.num_steps = original_num_steps


def train_forecasting(
    model,
    config,
    train_loader,
    valid_loader=None,
    foldername="",
    val_nsample=1,
):
    """
    Forecasting training loop with validation loss tracking, early stopping,
    and best checkpoint saving.
    """

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    valid_epoch_interval = int(config.get("valid_epoch_interval", 1))
    patience = int(config.get("patience", 10))
    fast_val = bool(config.get("fast_val", False))
    val_max_batches = config.get("val_max_batches", None)
    val_num_samples = int(config.get("val_num_samples", val_nsample))
    val_diffusion_steps = config.get("val_diffusion_steps", None)
    val_aggregate = config.get("val_aggregate", "mean")

    best_model_path = ""
    last_model_path = ""
    if foldername:
        os.makedirs(foldername, exist_ok=True)
        best_model_path = os.path.join(foldername, "model_best.pth")
        last_model_path = os.path.join(foldername, "model_last.pth")

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch_no in range(config["epochs"]):
        avg_loss = 0.0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

        lr_scheduler.step()

        if last_model_path:
            torch.save(model.state_dict(), last_model_path)

        if valid_loader is None or (epoch_no + 1) % valid_epoch_interval != 0:
            continue

        val_mode = "FAST_VAL" if fast_val else "FULL_VAL"
        current_val_max_batches = val_max_batches if fast_val else None
        current_val_nsample = val_num_samples if fast_val else val_nsample
        current_val_diffusion_steps = val_diffusion_steps if fast_val else None

        print(
            f"\n[{val_mode}] epoch={epoch_no} "
            f"max_batches={current_val_max_batches if current_val_max_batches is not None else 'ALL'} "
            f"num_samples={current_val_nsample} "
            f"diffusion_steps={current_val_diffusion_steps if current_val_diffusion_steps is not None else model.num_steps}"
        )
        valid_metrics = evaluate_forecasting(
            model,
            valid_loader,
            nsample=current_val_nsample,
            scaler=1,
            mean_scaler=0,
            foldername="",
            split_name="val",
            aggregate=val_aggregate,
            max_batches=current_val_max_batches,
            diffusion_steps=current_val_diffusion_steps,
            mode_name=val_mode,
            save_metrics=False,
        )
        valid_score = valid_metrics["mse"]

        if valid_score + 1e-8 < best_valid_loss:
            best_valid_loss = valid_score
            best_epoch = epoch_no
            epochs_without_improvement = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
            print(
                "\n best validation mse updated to ",
                best_valid_loss,
                "at epoch",
                epoch_no,
            )
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch_no}. "
                f"Best epoch: {best_epoch}, best validation mse: {best_valid_loss:.6f}"
            )
            break

    # When validation is disabled or skipped by a large valid_epoch_interval,
    # we still need a usable checkpoint for the evaluation phase.
    if best_epoch < 0:
        if best_model_path:
            torch.save(model.state_dict(), best_model_path)
        best_epoch = epoch_no
        best_valid_loss = float("nan")

    return {
        "best_model_path": best_model_path or last_model_path,
        "last_model_path": last_model_path,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
    }


def evaluate_forecasting(
    model,
    data_loader,
    nsample=5,
    scaler=1,
    mean_scaler=0,
    foldername="",
    split_name="test",
    aggregate="mean",
    max_batches=None,
    diffusion_steps=None,
    mode_name="FULL_VAL",
    save_metrics=True,
    metric_on_original_scale=False,
):
    """
    Forecasting evaluation in the original data scale.

    For generative outputs, we use a fixed aggregation over samples:
    - `mean`: average all generated trajectories
    - `median`: pointwise median over generated trajectories
    """

    if aggregate not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregate mode: {aggregate}")

    with torch.no_grad():
        model.eval()
        mse_total = 0.0
        mae_total = 0.0
        evalpoints_total = 0.0

        scaler = torch.as_tensor(scaler, device=next(model.parameters()).device).float()
        mean_scaler = torch.as_tensor(mean_scaler, device=scaler.device).float()

        if scaler.ndim == 1:
            scaler = scaler.view(1, 1, -1)
        if mean_scaler.ndim == 1:
            mean_scaler = mean_scaler.view(1, 1, -1)

        with temporary_diffusion_steps(model, diffusion_steps):
            with tqdm(data_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    samples, c_target, eval_points, _, _ = model.evaluate(test_batch, nsample)

                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)

                    feature_eval_mask = test_batch.get("eval_feature_mask", None)
                    if feature_eval_mask is not None:
                        feature_eval_mask = feature_eval_mask.to(c_target.device).float()
                        eval_points = eval_points * feature_eval_mask

                    if aggregate == "mean":
                        point_forecast = samples.mean(dim=1)
                    else:
                        point_forecast = samples.median(dim=1).values

                    if metric_on_original_scale:
                        point_forecast = point_forecast * scaler + mean_scaler
                        c_target = c_target * scaler + mean_scaler

                    mse_current = ((point_forecast - c_target) * eval_points) ** 2
                    mae_current = torch.abs((point_forecast - c_target) * eval_points)

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "mode": mode_name,
                            "mse": mse_total / max(evalpoints_total, 1.0),
                            "mae": mae_total / max(evalpoints_total, 1.0),
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                    if max_batches is not None and batch_no >= int(max_batches):
                        break

    metrics = {
        "mse": mse_total / max(evalpoints_total, 1.0),
        "mae": mae_total / max(evalpoints_total, 1.0),
        "eval_points": evalpoints_total,
        "nsample": nsample,
        "aggregate": aggregate,
        "mode_name": mode_name,
        "max_batches": max_batches,
        "diffusion_steps": diffusion_steps if diffusion_steps is not None else model.num_steps,
        "metric_on_original_scale": metric_on_original_scale,
    }

    if foldername and save_metrics:
        os.makedirs(foldername, exist_ok=True)
        metrics_path = os.path.join(foldername, f"{split_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    scale_tag = "ORIG_SCALE" if metric_on_original_scale else "SCALED_BENCHMARK"
    print(f"[{mode_name}][{scale_tag}] {split_name.upper()} MSE: {metrics['mse']:.6f}")
    print(f"[{mode_name}][{scale_tag}] {split_name.upper()} MAE: {metrics['mae']:.6f}")
    return metrics
