import hashlib
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import Meek
from causallearn.utils.PCUtils.Helper import append_value


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[1] == 0:
        return features
    centered = features - features.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return centered / scale


def _augment_condition_set(
    conditioning_set: Iterable[int], x: int, y: int, domain_index: int
) -> Tuple[int, ...]:
    condition = set(int(node) for node in conditioning_set)
    condition.discard(int(x))
    condition.discard(int(y))
    if x != domain_index and y != domain_index:
        condition.add(int(domain_index))
    return tuple(sorted(condition))


class MixedFeatureCIT:
    """
    Feature-based conditional independence test for H-FedLCS.

    Continuous variables are mapped with random Fourier features.
    The domain variable is mapped with a one-hot encoding, which is
    an exact feature representation of the Delta kernel.
    """

    def __init__(
        self,
        data: np.ndarray,
        domain_index: int,
        n_components: int = 64,
        gamma: float = 1.0,
        ridge: float = 1e-6,
        n_perm: int = 64,
        random_state: int = 42,
        discrete_indices: Optional[Sequence[int]] = None,
    ):
        self.data = np.asarray(data, dtype=float)
        self.domain_index = int(domain_index)
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.ridge = float(ridge)
        self.n_perm = int(n_perm)
        self.random_state = int(random_state)
        self.discrete_indices = set(discrete_indices or [])
        self.discrete_indices.add(self.domain_index)
        self.method = "hfedlcs_kci"
        self.feature_cache: Dict[int, np.ndarray] = {}
        self.pvalue_cache: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}

    def _feature_seed(self, index: int) -> int:
        digest = hashlib.md5(f"{self.random_state}:{index}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _permutation_seed(self, x: int, y: int, condition_set: Tuple[int, ...]) -> int:
        key = f"{self.random_state}:{x}:{y}:{','.join(map(str, condition_set))}"
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _delta_feature_map(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=int).reshape(-1)
        unique_values, inverse = np.unique(values, return_inverse=True)
        features = np.eye(len(unique_values), dtype=float)[inverse]
        return _normalize_features(features)

    def _rff_feature_map(self, values: np.ndarray, index: int) -> np.ndarray:
        values = _normalize_features(values.reshape(-1, 1))
        sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self._feature_seed(index),
        )
        features = sampler.fit_transform(values)
        return _normalize_features(features)

    def _map_variable(self, index: int) -> np.ndarray:
        index = int(index)
        if index not in self.feature_cache:
            column = self.data[:, index]
            if index in self.discrete_indices:
                self.feature_cache[index] = self._delta_feature_map(column)
            else:
                self.feature_cache[index] = self._rff_feature_map(column, index)
        return self.feature_cache[index]

    def _stack_conditioning_features(self, condition_set: Sequence[int]) -> np.ndarray:
        if len(condition_set) == 0:
            return np.zeros((self.data.shape[0], 0))
        blocks = [self._map_variable(index) for index in condition_set]
        return _normalize_features(np.concatenate(blocks, axis=1))

    def _residualize(self, features: np.ndarray, conditioning_features: np.ndarray) -> np.ndarray:
        features = _normalize_features(features)
        if conditioning_features.shape[1] == 0:
            return features
        z = _normalize_features(conditioning_features)
        gram = z.T @ z + self.ridge * np.eye(z.shape[1])
        coef = np.linalg.solve(gram, z.T @ features)
        residual = features - z @ coef
        return _normalize_features(residual)

    @staticmethod
    def _statistic(x_features: np.ndarray, y_features: np.ndarray) -> float:
        covariance = x_features.T @ y_features / float(x_features.shape[0])
        return float(np.sum(covariance * covariance))

    def __call__(self, X: int, Y: int, condition_set=None, gmm: int = 0, K: int = 0) -> float:
        _ = gmm
        _ = K
        x = int(X)
        y = int(Y)
        if condition_set is None:
            condition_set = ()
        condition = tuple(sorted(set(int(node) for node in condition_set)))
        key = (min(x, y), max(x, y), condition)
        if key in self.pvalue_cache:
            return self.pvalue_cache[key]

        x_features = self._map_variable(x)
        y_features = self._map_variable(y)
        z_features = self._stack_conditioning_features(condition)

        residual_x = self._residualize(x_features, z_features)
        residual_y = self._residualize(y_features, z_features)
        observed = self._statistic(residual_x, residual_y)

        rng = np.random.default_rng(self._permutation_seed(x, y, condition))
        count = 0
        for _ in range(self.n_perm):
            permuted = residual_y[rng.permutation(residual_y.shape[0])]
            permuted_stat = self._statistic(residual_x, permuted)
            if permuted_stat >= observed - 1e-12:
                count += 1
        p_value = float(count + 1) / float(self.n_perm + 1)
        self.pvalue_cache[key] = p_value
        return p_value


def _make_empty_causal_graph(no_of_var: int, node_names: Optional[List[str]] = None) -> CausalGraph:
    cg = CausalGraph(no_of_var, node_names)
    edges = []
    for i in range(no_of_var):
        for j in range(i + 1, no_of_var):
            edge = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge is not None:
                edges.append(edge)
    cg.G.remove_edges(edges)
    cg.sepset = np.empty((no_of_var + 1, no_of_var + 1), object)
    return cg


def _has_edge(cg: CausalGraph, x: int, y: int) -> bool:
    return cg.G.graph[x, y] != 0 or cg.G.graph[y, x] != 0


def _remove_edge(cg: CausalGraph, x: int, y: int) -> None:
    edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
    if edge is not None:
        cg.G.remove_edge(edge)
        return
    edge = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
    if edge is not None:
        cg.G.remove_edge(edge)


def _add_undirected_edge(cg: CausalGraph, x: int, y: int) -> None:
    if x == y or _has_edge(cg, x, y):
        return
    cg.G.add_edge(Edge(cg.G.nodes[x], cg.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))


def _orient_edge(cg: CausalGraph, parent: int, child: int) -> bool:
    if parent == child:
        return False
    if cg.is_fully_directed(parent, child):
        return True
    if cg.is_fully_directed(child, parent):
        return False
    if cg.G.is_ancestor_of(cg.G.nodes[child], cg.G.nodes[parent]):
        return False
    _remove_edge(cg, parent, child)
    cg.G.add_edge(
        Edge(cg.G.nodes[parent], cg.G.nodes[child], Endpoint.TAIL, Endpoint.ARROW)
    )
    return True


def _find_separating_set(
    x: int,
    y: int,
    candidate_nodes: Sequence[int],
    depth: int,
    ci_test: MixedFeatureCIT,
    alpha: float,
    domain_index: int,
) -> Tuple[bool, Tuple[int, ...]]:
    candidates = [node for node in sorted(set(candidate_nodes)) if node not in (x, y, domain_index)]
    if depth > len(candidates):
        return False, ()
    for condition_set in combinations(candidates, depth):
        augmented = _augment_condition_set(condition_set, x, y, domain_index)
        p_value = ci_test(x, y, augmented)
        if p_value > alpha:
            return True, tuple(sorted(condition_set))
    return False, ()


def _discover_local_region(
    no_of_var: int,
    target: int,
    domain_index: int,
    ci_test: MixedFeatureCIT,
    alpha: float,
    max_hops: int,
) -> List[int]:
    local_nodes = {int(target)}
    frontier = {int(target)}
    observed_nodes = [node for node in range(no_of_var) if node != domain_index]
    hop = 0
    while frontier and (max_hops < 0 or hop <= max_hops):
        next_frontier = set()
        conditioning_pool = sorted(local_nodes)
        for center in sorted(frontier):
            for candidate in observed_nodes:
                if candidate == center or candidate in local_nodes:
                    continue
                separated, _ = _find_separating_set(
                    center,
                    candidate,
                    [node for node in conditioning_pool if node != center],
                    hop,
                    ci_test,
                    alpha,
                    domain_index,
                )
                if not separated:
                    next_frontier.add(candidate)
        if not next_frontier:
            break
        local_nodes.update(next_frontier)
        frontier = next_frontier
        hop += 1
    return sorted(local_nodes)


def _restricted_local_skeleton(
    no_of_var: int,
    local_nodes: Sequence[int],
    domain_index: int,
    ci_test: MixedFeatureCIT,
    alpha: float,
    max_cond_size: int,
    node_names: Optional[List[str]] = None,
) -> CausalGraph:
    cg = _make_empty_causal_graph(no_of_var, node_names=node_names)
    local_nodes = sorted(set(int(node) for node in local_nodes))
    for i, x in enumerate(local_nodes):
        for y in local_nodes[i + 1 :]:
            _add_undirected_edge(cg, x, y)

    depth = 0
    while True:
        if max_cond_size >= 0 and depth > max_cond_size:
            break
        adjacency_snapshot = {
            node: [adj for adj in cg.neighbors(node) if adj in local_nodes]
            for node in local_nodes
        }
        edge_removal = []
        has_testable_pair = False
        for i, x in enumerate(local_nodes):
            for y in local_nodes[i + 1 :]:
                if not _has_edge(cg, x, y):
                    continue
                neighbors_x = [node for node in adjacency_snapshot[x] if node != y]
                neighbors_y = [node for node in adjacency_snapshot[y] if node != x]
                candidate_nodes = sorted(set(neighbors_x + neighbors_y))
                candidate_nodes = [node for node in candidate_nodes if node != domain_index]
                if len(candidate_nodes) < depth:
                    continue
                has_testable_pair = True
                separated, sepset = _find_separating_set(
                    x,
                    y,
                    candidate_nodes,
                    depth,
                    ci_test,
                    alpha,
                    domain_index,
                )
                if separated:
                    edge_removal.append((x, y, sepset))
        if not has_testable_pair:
            break
        for x, y, sepset in edge_removal:
            if _has_edge(cg, x, y):
                _remove_edge(cg, x, y)
                append_value(cg.sepset, x, y, sepset)
                append_value(cg.sepset, y, x, sepset)
        depth += 1
    return cg


def _attach_domain_edges(
    cg: CausalGraph,
    local_nodes: Sequence[int],
    domain_index: int,
    ci_test: MixedFeatureCIT,
    alpha: float,
    max_cond_size: int,
) -> None:
    observed_nodes = [node for node in sorted(set(local_nodes)) if node != domain_index]
    for node in observed_nodes:
        candidate_nodes = [candidate for candidate in observed_nodes if candidate != node]
        max_depth = len(candidate_nodes) if max_cond_size < 0 else min(max_cond_size, len(candidate_nodes))
        separated = False
        for depth in range(max_depth + 1):
            found, sepset = _find_separating_set(
                domain_index,
                node,
                candidate_nodes,
                depth,
                ci_test,
                alpha,
                domain_index,
            )
            if found:
                append_value(cg.sepset, domain_index, node, sepset)
                append_value(cg.sepset, node, domain_index, sepset)
                separated = True
                break
        if not separated:
            _add_undirected_edge(cg, domain_index, node)

    for node in observed_nodes:
        if _has_edge(cg, domain_index, node):
            _orient_edge(cg, domain_index, node)


def _orient_v_structures_from_sepsets(
    cg: CausalGraph, skip_nodes: Optional[Sequence[int]] = None
) -> CausalGraph:
    skip = set(skip_nodes or [])
    for x, y, z in cg.find_unshielded_triples():
        if x >= z:
            continue
        if x in skip or y in skip or z in skip:
            continue
        sepsets = cg.sepset[x, z]
        middle_in_sepset = False
        if sepsets is not None:
            middle_in_sepset = any(y in sepset for sepset in sepsets)
        if middle_in_sepset:
            continue
        _orient_edge(cg, x, y)
        _orient_edge(cg, z, y)
    return cg


def _linear_residual(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float).reshape(-1)
    if design.size == 0:
        return target - target.mean()
    design = np.asarray(design, dtype=float)
    intercept = np.ones((design.shape[0], 1))
    design_matrix = np.concatenate((intercept, design), axis=1)
    coef, _, _, _ = np.linalg.lstsq(design_matrix, target, rcond=None)
    return target - design_matrix @ coef


def _variance_divergence(residual: np.ndarray, domain_index: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    domain_values = np.asarray(domain_index).reshape(-1)
    unique_domains = np.unique(domain_values)
    variances = []
    for value in unique_domains:
        local_residual = residual[domain_values == value]
        if local_residual.size < 2:
            continue
        centered = local_residual - local_residual.mean()
        variances.append(float(np.mean(centered * centered)))
    if len(variances) < 2:
        return float("inf")
    pairwise_gap = [abs(left - right) for left, right in combinations(variances, 2)]
    return float(np.mean(pairwise_gap))


def _ficp_score(
    data: np.ndarray,
    domain_index: np.ndarray,
    source: int,
    target: int,
    conditioning_set: Sequence[int],
) -> float:
    predictors = [int(source)] + [int(node) for node in conditioning_set]
    design = data[:, predictors] if len(predictors) > 0 else np.zeros((data.shape[0], 0))
    residual = _linear_residual(data[:, target], design)
    return _variance_divergence(residual, domain_index)


def _orient_remaining_edges_with_ficp(
    cg: CausalGraph, data: np.ndarray, c_indx: np.ndarray, domain_index: int
) -> CausalGraph:
    undirected_edges = sorted({tuple(sorted(edge)) for edge in cg.find_undirected()})
    for x, y in undirected_edges:
        if domain_index in (x, y):
            continue
        if not cg.is_undirected(x, y):
            continue
        forward_condition = [
            node
            for node in cg.neighbors(y)
            if node not in (x, y, domain_index) and cg.is_fully_directed(node, y)
        ]
        backward_condition = [
            node
            for node in cg.neighbors(x)
            if node not in (x, y, domain_index) and cg.is_fully_directed(node, x)
        ]
        score_xy = _ficp_score(data, c_indx, x, y, forward_condition)
        score_yx = _ficp_score(data, c_indx, y, x, backward_condition)
        if score_xy == float("inf") and score_yx == float("inf"):
            continue
        if score_xy <= score_yx:
            changed = _orient_edge(cg, x, y)
        else:
            changed = _orient_edge(cg, y, x)
        if changed:
            cg = Meek.meek(cg)
    return cg


def hfedlcs(
    data: np.ndarray,
    c_indx: np.ndarray,
    target: int,
    K: Optional[int] = None,
    alpha: float = 0.05,
    indep_test: str = "kci",
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    max_hops: int = 2,
    max_cond_size: int = 2,
    n_components: int = 64,
    gamma: float = 1.0,
    ridge: float = 1e-6,
    n_perm: int = 64,
    random_state: int = 42,
    node_names: Optional[List[str]] = None,
    **kwargs,
) -> CausalGraph:
    """
    H-FedLCS: Local causal structure learning under heterogeneous data.

    Parameters largely follow the FedCDH/CDNOD entry style, but the algorithm
    is target-centric and uses a mixed Delta/RFF feature conditional
    independence test.
    """

    _ = K
    _ = indep_test
    _ = stable
    _ = uc_rule
    _ = uc_priority
    _ = kwargs

    data = np.asarray(data, dtype=float)
    c_indx = np.asarray(c_indx).reshape(-1, 1)
    if data.shape[0] != c_indx.shape[0]:
        raise ValueError("data and c_indx must contain the same number of samples.")
    if target < 0 or target >= data.shape[1]:
        raise ValueError("target must be a valid observed variable index.")

    data_aug = np.concatenate((data, c_indx), axis=1)
    domain_index = data_aug.shape[1] - 1

    ci_test = MixedFeatureCIT(
        data_aug,
        domain_index=domain_index,
        n_components=n_components,
        gamma=gamma,
        ridge=ridge,
        n_perm=n_perm,
        random_state=random_state,
    )

    local_nodes = _discover_local_region(
        data_aug.shape[1],
        target,
        domain_index,
        ci_test,
        alpha,
        max_hops=max_hops,
    )

    cg = _restricted_local_skeleton(
        data_aug.shape[1],
        local_nodes,
        domain_index,
        ci_test,
        alpha,
        max_cond_size=max_cond_size,
        node_names=node_names,
    )

    _attach_domain_edges(cg, local_nodes, domain_index, ci_test, alpha, max_cond_size)
    cg = _orient_v_structures_from_sepsets(cg, skip_nodes=[domain_index])
    cg = Meek.meek(cg)
    cg = _orient_remaining_edges_with_ficp(cg, data, c_indx, domain_index)

    cg.target = int(target)
    cg.domain_index = int(domain_index)
    cg.local_nodes = sorted(local_nodes)
    cg.local_nodes_with_domain = sorted(local_nodes + [domain_index])
    cg.ci_test_mixed = ci_test
    return cg
