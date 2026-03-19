import argparse
import os
import sys
import time

import numpy as np

sys.path.append("")

from causallearn.search.ConstraintBased.HFedLCS import hfedlcs
from causallearn.utils.hfedlcs_utils import (
    evaluate_local_causal_graph,
    load_heterogeneous_dataset,
    load_numeric_adjacency,
    plot_client_scaling,
)


np.set_printoptions(suppress=True, precision=3)


def set_random_seed(seed: int):
    np.random.seed(seed)


def simulate_dag(d: int, s0: int, graph_type: str = "ER") -> np.ndarray:
    _ = graph_type
    dag = np.zeros((d, d), dtype=int)
    order = np.random.permutation(d)
    candidates = [(i, j) for i in range(d) for j in range(i + 1, d)]
    edge_num = min(int(s0), len(candidates))
    if edge_num > 0:
        chosen = np.random.choice(len(candidates), size=edge_num, replace=False)
        for index in chosen:
            left, right = candidates[index]
            dag[order[left], order[right]] = 1
    return dag


def _topological_order(dag: np.ndarray):
    indegree = dag.sum(axis=0).astype(int)
    queue = [node for node in range(dag.shape[0]) if indegree[node] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in np.where(dag[node] != 0)[0]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order


def my_simulate_linear_gaussian(dag: np.ndarray, K: int, total_n: int, sem_type: str = "gauss"):
    _ = sem_type
    samples_per_client = total_n // K
    d = dag.shape[0]
    data = np.zeros((samples_per_client * K, d))
    topo_order = _topological_order(dag)
    signed_weights = dag * np.random.uniform(0.5, 1.5, size=dag.shape)
    signed_weights *= np.random.choice([-1.0, 1.0], size=dag.shape)

    for client_id in range(K):
        start = client_id * samples_per_client
        end = start + samples_per_client
        for node in topo_order:
            parents = np.where(dag[:, node] != 0)[0]
            noise = np.random.normal(loc=0.0, scale=1.0 + 0.1 * client_id, size=samples_per_client)
            value = noise
            if len(parents) > 0:
                value += data[start:end][:, parents] @ signed_weights[parents, node]
            data[start:end, node] = value
    return data, signed_weights


def my_simulate_general_hetero(dag: np.ndarray, K: int, total_n: int, sem_type: str = "gauss"):
    _ = sem_type
    linear_data, weights = my_simulate_linear_gaussian(dag, K, total_n, sem_type)
    samples_per_client = total_n // K
    transformed = np.zeros_like(linear_data)
    topo_order = _topological_order(dag)

    for client_id in range(K):
        start = client_id * samples_per_client
        end = start + samples_per_client
        for node in topo_order:
            parents = np.where(dag[:, node] != 0)[0]
            noise = np.random.normal(loc=0.0, scale=0.3 + 0.05 * client_id, size=samples_per_client)
            if len(parents) == 0:
                transformed[start:end, node] = linear_data[start:end, node] + noise
            else:
                signal = np.tanh(linear_data[start:end][:, parents] @ weights[parents, node])
                transformed[start:end, node] = signal + noise
    return transformed, weights


def _build_client_index(K: int, n: int) -> np.ndarray:
    client_index = np.repeat(np.arange(K), n)
    return client_index.reshape(-1, 1)


def run_synthetic_instance(seed: int, n: int, K: int, d: int, s0: int, model: str, target: int):
    set_random_seed(seed)
    print(f"Running synthetic instance {seed} with K={K}.")

    c_indx = _build_client_index(K, n)
    true_dag = simulate_dag(d, s0, "ER")

    if model == "linear":
        data, _ = my_simulate_linear_gaussian(true_dag, K, n * K, "gauss")
    else:
        data, _ = my_simulate_general_hetero(true_dag, K, n * K, "gauss")

    start = time.time()
    cg = hfedlcs(
        data,
        c_indx,
        target=target,
        K=K,
        alpha=0.05,
        max_hops=2,
        max_cond_size=2,
        n_perm=32,
        random_state=seed,
    )
    end = time.time()

    metrics = evaluate_local_causal_graph(true_dag, cg, num_observed=d)
    metrics["time"] = end - start
    return metrics


def run_dataset_instance(dataset_dir: str, K: int, target: int, true_dag_path=None):
    bundle = load_heterogeneous_dataset(dataset_dir, max_clients=K, true_dag_path=true_dag_path)
    data = bundle["data"]
    c_indx = bundle["c_indx"]

    start = time.time()
    cg = hfedlcs(
        data,
        c_indx,
        target=target,
        K=K,
        alpha=0.05,
        max_hops=2,
        max_cond_size=2,
        n_perm=32,
        random_state=42,
    )
    end = time.time()

    metrics = {"time": end - start}
    if bundle["true_dag"] is not None:
        metrics.update(
            evaluate_local_causal_graph(
                bundle["true_dag"],
                cg,
                num_observed=data.shape[1],
            )
        )
    return metrics


def summarize_results(result_list):
    keys = list(result_list[0].keys())
    values = np.array([[result[key] for key in keys] for result in result_list], dtype=float)
    return keys, values.mean(axis=0), values.std(axis=0)


def evaluate_synthetic(args):
    curve_results = []
    for K in args.K_list:
        run_results = []
        for seed in range(args.N):
            run_results.append(
                run_synthetic_instance(seed, args.n, K, args.d, args.d, args.model, args.target)
            )
        keys, avg, std = summarize_results(run_results)
        summary = {key: float(avg[idx]) for idx, key in enumerate(keys)}
        summary["num_clients"] = int(K)
        curve_results.append(summary)
        print("########## Synthetic K =", K)
        print("########## Measurement:", keys)
        print("########## Average:    ", avg)
        print("########## Std:        ", std)

    if args.plot_path:
        plot_client_scaling(curve_results, args.plot_path, title="H-FedLCS Synthetic Scaling")


def evaluate_dataset(args):
    dataset_dir = os.path.abspath(os.path.join(args.data_root, args.dataset))
    if args.true_dag:
        true_dag_path = os.path.abspath(args.true_dag)
        _ = load_numeric_adjacency(true_dag_path)
    else:
        true_dag_path = None

    curve_results = []
    for K in args.K_list:
        metrics = run_dataset_instance(dataset_dir, K, args.target, true_dag_path=true_dag_path)
        metrics["num_clients"] = int(K)
        curve_results.append(metrics)
        print("########## Dataset K =", K)
        print(metrics)

    if args.plot_path:
        plot_client_scaling(curve_results, args.plot_path, title=f"H-FedLCS on {args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-FedLCS evaluation")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "dataset"])
    parser.add_argument("--N", default=5, type=int, help="number of synthetic runs")
    parser.add_argument("--d", default=6, type=int, help="number of variables")
    parser.add_argument("--n", default=100, type=int, help="samples per client")
    parser.add_argument("--model", default="linear", type=str, help="linear or general")
    parser.add_argument("--target", default=0, type=int, help="target variable index")
    parser.add_argument(
        "--K-list",
        nargs="+",
        default=[5, 10, 15, 20],
        type=int,
        help="client counts for scaling evaluation",
    )
    parser.add_argument("--data-root", default=".", type=str, help="root path of benchmark datasets")
    parser.add_argument(
        "--dataset",
        default="child_data",
        type=str,
        help="dataset directory name, e.g. child_data or alarm_data",
    )
    parser.add_argument("--true-dag", default="", type=str, help="optional path to the ground-truth DAG")
    parser.add_argument(
        "--plot-path",
        default=os.path.join("tests", "hfedlcs_metrics.png"),
        type=str,
        help="where to save the metric figure",
    )
    arguments = parser.parse_args()

    if arguments.mode == "synthetic":
        evaluate_synthetic(arguments)
    else:
        evaluate_dataset(arguments)
