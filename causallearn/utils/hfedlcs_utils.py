import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRUE_GRAPH_PATTERNS = (
    "true_dag.txt",
    "dag.txt",
    "adj.txt",
    "graph.txt",
    "true_graph.txt",
)


def _is_true_graph_file(file_name: str) -> bool:
    lowered = file_name.lower()
    if lowered in TRUE_GRAPH_PATTERNS:
        return True
    if lowered.endswith("_graph.txt"):
        return True
    if lowered.endswith("_dag.txt"):
        return True
    return False


def _read_numeric_table(path: str) -> np.ndarray:
    separators = (r"\s+", ",", "\t")
    for separator in separators:
        try:
            frame = pd.read_csv(path, sep=separator, engine="python", header=None)
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if numeric.empty:
                continue
            if numeric.isna().any().any():
                continue
            return numeric.to_numpy(dtype=float)
        except Exception:
            continue
    raise ValueError(f"Unable to parse numeric data from {path}.")


def load_numeric_adjacency(path: str) -> np.ndarray:
    adjacency = _read_numeric_table(path)
    adjacency = np.asarray(adjacency, dtype=int)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Ground-truth adjacency matrix must be square.")
    return adjacency


def _find_true_graph_file(dataset_dir: str) -> Optional[str]:
    for pattern in TRUE_GRAPH_PATTERNS:
        candidate = os.path.join(dataset_dir, pattern)
        if os.path.exists(candidate):
            return candidate
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            if _is_true_graph_file(file_name):
                return os.path.join(root, file_name)
    return None


def discover_benchmark_datasets(root_dir: str) -> Dict[str, str]:
    datasets = {}
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and name.endswith("_data"):
            datasets[name] = path
    return datasets


def load_heterogeneous_dataset(
    dataset_dir: str,
    max_clients: Optional[int] = None,
    true_dag_path: Optional[str] = None,
) -> Dict[str, object]:
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    client_files = []
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            lowered = file_name.lower()
            if not lowered.endswith(".txt"):
                continue
            if _is_true_graph_file(file_name):
                continue
            client_files.append(os.path.join(root, file_name))
    client_files = sorted(client_files)
    if len(client_files) == 0:
        raise FileNotFoundError(f"No client .txt files found under {dataset_dir}")

    if max_clients is not None:
        client_files = client_files[: int(max_clients)]

    client_data = []
    domain_blocks = []
    client_meta = []
    feature_dim = None

    for client_id, client_file in enumerate(client_files):
        matrix = _read_numeric_table(client_file)
        if feature_dim is None:
            feature_dim = matrix.shape[1]
        elif matrix.shape[1] != feature_dim:
            raise ValueError("All client files must share the same feature dimension.")
        client_data.append(matrix)
        domain_blocks.append(np.full((matrix.shape[0], 1), client_id, dtype=int))
        client_meta.append(
            {
                "client_id": client_id,
                "path": client_file,
                "n_samples": int(matrix.shape[0]),
            }
        )

    data = np.vstack(client_data)
    c_indx = np.vstack(domain_blocks)

    true_graph_file = true_dag_path or _find_true_graph_file(dataset_dir)
    true_dag = None
    if true_graph_file is not None and os.path.exists(true_graph_file):
        true_dag = load_numeric_adjacency(true_graph_file)

    return {
        "dataset_dir": dataset_dir,
        "data": data,
        "c_indx": c_indx,
        "client_data": client_data,
        "client_files": client_files,
        "client_meta": client_meta,
        "num_clients": len(client_files),
        "true_dag": true_dag,
        "true_dag_path": true_graph_file,
    }


def _to_matrix(graph_or_matrix, num_observed: Optional[int] = None) -> np.ndarray:
    if hasattr(graph_or_matrix, "G"):
        matrix = np.asarray(graph_or_matrix.G.graph)
    else:
        matrix = np.asarray(graph_or_matrix)
    if num_observed is not None:
        matrix = matrix[:num_observed, :num_observed]
    return matrix


def graph_to_directed_adjacency(graph_or_matrix, num_observed: Optional[int] = None) -> np.ndarray:
    matrix = _to_matrix(graph_or_matrix, num_observed=num_observed)
    directed = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
    if np.any(matrix < 0):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[j, i] == 1 and matrix[i, j] == -1:
                    directed[i, j] = 1
    else:
        directed = (matrix != 0).astype(int)
        np.fill_diagonal(directed, 0)
    return directed


def graph_to_skeleton_adjacency(graph_or_matrix, num_observed: Optional[int] = None) -> np.ndarray:
    matrix = _to_matrix(graph_or_matrix, num_observed=num_observed)
    skeleton = ((matrix != 0) | (matrix.T != 0)).astype(int)
    np.fill_diagonal(skeleton, 0)
    return skeleton


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = 0.0 if tp + fp == 0 else float(tp) / float(tp + fp)
    recall = 0.0 if tp + fn == 0 else float(tp) / float(tp + fn)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_skeleton_metrics(true_adj: np.ndarray, est_adj: np.ndarray) -> Dict[str, float]:
    true_skeleton = graph_to_skeleton_adjacency(true_adj)
    est_skeleton = graph_to_skeleton_adjacency(est_adj)
    d = true_skeleton.shape[0]
    true_upper = true_skeleton[np.triu_indices(d, k=1)]
    est_upper = est_skeleton[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(est_upper)
    cond = np.flatnonzero(true_upper)
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    false_neg = np.setdiff1d(cond, pred, assume_unique=True)
    _, _, f1 = _precision_recall_f1(len(true_pos), len(false_pos), len(false_neg))
    shd = len(false_pos) + len(false_neg)
    return {"skeleton_f1": f1, "skeleton_shd": float(shd)}


def compute_direction_metrics(true_adj: np.ndarray, est_adj: np.ndarray) -> Dict[str, float]:
    true_directed = graph_to_directed_adjacency(true_adj)
    est_directed = graph_to_directed_adjacency(est_adj)
    pred = np.flatnonzero(est_directed)
    cond = np.flatnonzero(true_directed)
    cond_reversed = np.flatnonzero(true_directed.T)
    cond_skeleton = np.concatenate((cond, cond_reversed))
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    _, _, f1 = _precision_recall_f1(
        len(true_pos), len(false_pos) + len(reverse), len(false_neg)
    )
    pred_lower = np.flatnonzero(np.tril(est_directed + est_directed.T))
    cond_lower = np.flatnonzero(np.tril(true_directed + true_directed.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {"direction_f1": f1, "direction_shd": float(shd)}


def evaluate_local_causal_graph(
    true_adj: np.ndarray,
    graph_or_matrix,
    num_observed: Optional[int] = None,
) -> Dict[str, float]:
    est_directed = graph_to_directed_adjacency(graph_or_matrix, num_observed=num_observed)
    est_skeleton = graph_to_skeleton_adjacency(graph_or_matrix, num_observed=num_observed)
    true_adj = np.asarray(true_adj, dtype=int)
    if num_observed is not None:
        true_adj = true_adj[:num_observed, :num_observed]
    metrics = {}
    metrics.update(compute_skeleton_metrics(true_adj, est_skeleton))
    metrics.update(compute_direction_metrics(true_adj, est_directed))
    return metrics


def plot_client_scaling(
    results: Sequence[Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "H-FedLCS Performance",
) -> Tuple[plt.Figure, np.ndarray]:
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    frame = pd.DataFrame(results).sort_values("num_clients")
    required = ["skeleton_f1", "skeleton_shd", "direction_f1", "direction_shd"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing metrics for plotting: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    specs = [
        ("skeleton_f1", "Skeleton F1"),
        ("skeleton_shd", "Skeleton SHD"),
        ("direction_f1", "Direction F1"),
        ("direction_shd", "Direction SHD"),
    ]
    colors = ["#0f4c81", "#bc4b51", "#2d6a4f", "#7f5539"]

    for axis, (metric, label), color in zip(axes, specs, colors):
        axis.plot(
            frame["num_clients"],
            frame[metric],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )
        axis.set_xlabel("Number of clients")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        axis.legend(frameon=True)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
