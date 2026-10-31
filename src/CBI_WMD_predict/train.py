"""Frozen BrainIAC features, Harvey site splits, and linear CBI/WMD heads."""

import argparse
import csv
import itertools
from pathlib import Path
import sys

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CBI_WMD_predict.common import (
    BLOCK_NUMBERS, FOLDS, MODALITIES, load_features, read_csv, write_csv, write_json,
)


METRICS = ("auroc", "auprc", "balanced_accuracy", "nll")
HISTORY_FIELDS = ["trial", "lr", "penalty", "alpha", "epoch", "eligible",
                  *[f"{split}_{key}" for split in ("train", "val") for key in METRICS]]
RESULT_FIELDS = ["modality", "embedding_field", "block", "target", "fold", "status", "n_train", "n_val", "n_test",
                 "lr", "penalty", "alpha", "epoch", "selection_val_auroc",
                 *[f"test_{key}" for key in METRICS]]


def metric_values(labels, logits):
    """AUROC is undefined if the dropped training subset contains only one class."""
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not len(labels) or not np.isfinite(logits).all():
        raise ValueError("Metrics require nonempty labels and finite logits")
    probabilities = np.exp(-np.logaddexp(0, -logits))
    classes = np.unique(labels)
    accuracy = [((probabilities[labels == value] >= 0.5) == value).mean()
                for value in classes]
    return {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(classes) == 2 else None,
        "auprc": float(average_precision_score(labels, probabilities)) if labels.sum() else 0.0,
        "balanced_accuracy": float(np.mean(accuracy)),
        "nll": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
    }


def penalty(model, kind, alpha):
    parameters = torch.nn.utils.parameters_to_vector(model.parameters())
    return (alpha / 2) * (parameters.abs().sum() if kind == "L1" else parameters.square().sum())


def evaluate(model, features, labels, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32, device=device)).flatten()
    values = logits.cpu().numpy()
    return metric_values(labels, values), values


def fit_trial(train_x, train_y, val_x, val_y, *, lr, kind, alpha, epochs,
              batch_size, seed, device, history_callback=None):
    """Keep the earliest best eligible epoch; validation alone ranks candidates."""
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    loader = DataLoader(TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y)),
                        batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    best = None
    for epoch in range(epochs):
        model.train()
        online_labels, online_logits = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = loss + penalty(model, kind, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            online_labels.append(y.detach().cpu().numpy())
            online_logits.append(logits.detach().cpu().numpy())
        train_metrics = metric_values(np.concatenate(online_labels), np.concatenate(online_logits))
        val_metrics, _ = evaluate(model, val_x, val_y, device)
        eligible = (train_metrics["auroc"] is not None
                    and train_metrics["auroc"] > val_metrics["auroc"])
        if eligible and (best is None or val_metrics["auroc"] > best["val_auroc"]):
            best = {"state_dict": {key: value.detach().cpu().clone()
                                   for key, value in model.state_dict().items()},
                    "epoch": epoch, "val_auroc": val_metrics["auroc"],
                    "train_auroc": train_metrics["auroc"]}
        if history_callback is not None:
            history_callback(dict(epoch=epoch, eligible=int(eligible),
                                  **{f"train_{key}": train_metrics[key] for key in METRICS},
                                  **{f"val_{key}": val_metrics[key] for key in METRICS}))
    summary = dict(lr=lr, penalty=kind, alpha=alpha,
                   status="ok" if best is not None else "no_eligible_checkpoint",
                   epoch=best["epoch"] if best else "",
                   val_auroc=best["val_auroc"] if best else "")
    return best, summary


def read_inputs(path):
    """Read one modality manifest and derive Harvey splits from its site IDs."""
    rows = read_csv(path, ("patient_id", "study_id", "site_id", "CBI", "WMD",
                           "embedding_path", "modality"))
    modes = {row["modality"] for row in rows}
    if len(modes) != 1 or not modes.issubset(MODALITIES):
        raise ValueError("Manifest must contain exactly one modality: T1, T2 or T1T2")
    modality = modes.pop()
    rows.sort(key=lambda row: (row["patient_id"], row["study_id"]))
    for name in ("patient_id", "study_id"):
        if any(not row[name] for row in rows) or len({row[name] for row in rows}) != len(rows):
            raise ValueError(f"Manifest must have a nonempty, unique {name} per row")
    keys = ("t1_unique_id", "t2_unique_id") if modality == "T1T2" else (f"{modality.lower()}_unique_id",)
    assignments = {}
    for row in rows:
        if any(not row.get(key) for key in keys):
            raise ValueError(f"{row['study_id']}: missing selected {modality} unique_id")
        if modality == "T1T2" and row[keys[0]] == row[keys[1]]:
            raise ValueError(f"{row['study_id']}: T1 and T2 must use distinct series")
        feature_path = Path(row["embedding_path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        row["embedding_path"] = str(feature_path.resolve())
        for fold, groups in FOLDS.items():
            split = next((role for role, sites in groups.items() if int(row["site_id"]) in sites), None)
            if split is None:
                raise ValueError(f"{row['study_id']}: site {row['site_id']} is not in Harvey assignments")
            assignments[row["study_id"], fold] = split
    return {row["study_id"]: row for row in rows}, assignments, modality


def task_rows(indexed, assignments, target, fold):
    grouped = {role: [] for role in ("train", "val", "test")}
    for study, row in indexed.items():
        if row[target] == "":
            continue
        if row[target] not in ("0", "1"):
            raise ValueError(f"Nonbinary {target} for {study}: {row[target]}")
        grouped[assignments[study, fold]].append(row)
    return grouped


def summarize(output, results):
    write_csv(output / "results.csv", results, RESULT_FIELDS)
    summary = []
    for modality, target in itertools.product(MODALITIES, ("CBI", "WMD")):
        subset = [row for row in results if row["modality"] == modality and row["target"] == target]
        if not subset:
            continue
        successful = [row for row in subset if row["status"] == "ok"]
        row = dict(modality=modality, embedding_field=subset[0]["embedding_field"],
                   block=subset[0]["block"], target=target, n_folds=len(subset), n_successful=len(successful))
        for metric in METRICS:
            values = [item[f"test_{metric}"] for item in successful]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
        summary.append(row)
    fields = ["modality", "embedding_field", "block", "target", "n_folds", "n_successful",
              *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]]
    write_csv(output / "summary.csv", summary, fields)


def run(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if any(not np.isfinite(x) or x <= 0 for x in args.learning_rates):
        raise ValueError("Learning rates must be finite and positive")
    if any(not np.isfinite(x) or x < 0 for x in args.alphas):
        raise ValueError("Regularization strengths must be finite and nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    indexed, assignments, modality = read_inputs(args.manifest)
    for target, fold in itertools.product(args.targets, args.folds):
        groups = task_rows(indexed, assignments, target, fold)
        for role, rows in groups.items():
            if {row[target] for row in rows} != {"0", "1"}:
                raise ValueError(f"{target}, fold {fold}, {role}: both classes are required")
        if len(groups["train"]) < args.batch_size:
            raise ValueError(f"{target}, fold {fold}: training size is smaller than batch size")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
                  sklearn_version=sklearn.__version__, modality=modality, encoder_frozen=True,
                  selection="train_online_auroc > val_auroc; maximize val_auroc; earliest tie",
                  test_evaluation="only after hyperparameter and epoch selection", site_splits=FOLDS)
    write_json(args.output_dir / "run_config.json", config)
    results = []
    for target, fold in itertools.product(args.targets, args.folds):
        folder = args.output_dir / modality / target / f"fold_{fold}"
        folder.mkdir(parents=True)
        groups = task_rows(indexed, assignments, target, fold)
        labels = {role: np.array([int(row[target]) for row in rows], dtype=np.float32)
                  for role, rows in groups.items()}
        train_x = load_features(groups["train"], modality, args.embedding_field, args.block)
        val_x = load_features(groups["val"], modality, args.embedding_field, args.block)
        best, trials = None, []
        with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for trial, (lr, kind, alpha) in enumerate(itertools.product(args.learning_rates, args.penalties, args.alphas)):
                setting = dict(trial=trial, lr=lr, penalty=kind, alpha=alpha)
                def log_epoch(row):
                    writer.writerow(dict(setting, **row))
                candidate, trial_summary = fit_trial(
                    train_x, labels["train"], val_x, labels["val"], lr=lr, kind=kind, alpha=alpha,
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                    device=device, history_callback=log_epoch)
                trials.append(dict(trial=trial, **trial_summary))
                stream.flush()
                if candidate is not None and (best is None or candidate["val_auroc"] > best["val_auroc"]):
                    best = dict(candidate, **setting)
                print(f"{modality}/{target}/fold_{fold}: trial {trial} {trial_summary}", flush=True)
        write_csv(folder / "trials.csv", trials,
                  ["trial", "lr", "penalty", "alpha", "status", "epoch", "val_auroc"])
        result = dict(modality=modality, embedding_field=args.embedding_field,
                      block=args.block if args.block is not None else "", target=target, fold=fold,
                      status="ok" if best else "no_eligible_checkpoint",
                      **{f"n_{role}": len(rows) for role, rows in groups.items()})
        if best is None:
            write_json(folder / "metrics.json", result)
        else:
            model = nn.Linear(train_x.shape[1], 1).to(device)
            model.load_state_dict(best["state_dict"])
            train_metrics, _ = evaluate(model, train_x, labels["train"], device)
            val_metrics, _ = evaluate(model, val_x, labels["val"], device)
            test_x = load_features(groups["test"], modality, args.embedding_field, args.block)
            test_metrics, test_logits = evaluate(model, test_x, labels["test"], device)
            chosen = {key: value for key, value in best.items() if key != "state_dict"}
            metadata = dict(chosen, modality=modality, target=target, fold=fold,
                            in_features=train_x.shape[1], seed=args.seed,
                            embedding_field=args.embedding_field, block=args.block,
                            manifest=str(args.manifest.resolve()))
            torch.save(dict(state_dict=best["state_dict"], metadata=metadata), folder / "best_model.pt")
            write_json(folder / "metrics.json", dict(result, selected=metadata,
                                                      train_eval=train_metrics, val=val_metrics, test=test_metrics))
            test_predictions = [dict(patient_id=row["patient_id"], study_id=row["study_id"],
                                     site_id=row["site_id"], modality=modality,
                                     embedding_field=args.embedding_field, block=result["block"], target=target, fold=fold,
                                     label=int(row[target]), logit=float(logit),
                                     probability=float(np.exp(-np.logaddexp(0, -float(logit)))))
                                for row, logit in zip(groups["test"], test_logits)]
            write_csv(folder / "predictions.csv", test_predictions,
                      ["patient_id", "study_id", "site_id", "modality", "embedding_field", "block",
                       "target", "fold", "label", "logit", "probability"])
            result.update({key: best[key] for key in ("lr", "penalty", "alpha", "epoch")})
            result.update(selection_val_auroc=best["val_auroc"],
                          **{f"test_{key}": test_metrics[key] for key in METRICS})
        results.append(result)
        summarize(args.output_dir, results)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="One T1, T2 or T1T2 manifest; modality is read from its rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.1, 0.01, 0.001, 0.0001])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 0])
    parser.add_argument("--penalties", choices=("L1", "L2"), nargs="+", default=["L1", "L2"])
    parser.add_argument("--folds", type=int, choices=list(FOLDS), nargs="+", default=list(FOLDS))
    parser.add_argument("--targets", choices=("CBI", "WMD"), nargs="+", default=["CBI", "WMD"])
    parser.add_argument("--embedding-field", choices=("final_embed", "block_embed"), default="final_embed")
    parser.add_argument("--block", type=int, choices=BLOCK_NUMBERS,
                        help="Actual encoder block number, required only with --embedding-field block_embed")
    args = parser.parse_args(argv)
    if (args.embedding_field == "block_embed") != (args.block is not None):
        parser.error("block_embed requires --block 3/6/9/12; final_embed must not specify --block")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
