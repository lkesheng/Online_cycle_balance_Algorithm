# -*- coding: utf-8 -*-
"""
Online cycle-balance simulation for a switched linear system on a strongly connected digraph.

What it does (aligned with your Problem 2 + Algorithm):
1) Build a strongly connected directed graph G.
2) Assign edge weights w_ij = log || P_j^{-1} P_i ||, and set dwell time = 1 for all switches.
3) Define a cycle score map Phi(C) (same structural form as your (16), with a minimal runnable surrogate).
4) Ensure the proportion of contracting/expanding simple cycles (len<=6) is ~50%.
5) Select the most contracting cycle C_comp (Karp-equivalent on small graphs via cycle enumeration).
6) Precompute a multi-source shortest-path policy to the vertex set of C_comp (unit weights -> multi-source BFS;
   replace with multi-source Dijkstra if you later introduce non-unit edge costs).
7) Run an online execution: random switching, online walk decomposition via a stack, dynamic cycle statistics,
   and compensation scheduling with hysteresis (enter/exit thresholds).
8) Save all figures in one run (no blocking windows).

Dependencies:
  pip install networkx matplotlib numpy
"""

import math
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# 全局统一字体设置（核心：所有字体大小完全一致）
# ----------------------------
plt.rcParams.update({
    'font.size': 9,               # 基础字体大小（统一基准）
    'axes.labelsize': 9,          # 坐标轴标签
    'axes.titlesize': 9,          # 图表标题
    'xtick.labelsize': 9,         # x轴刻度
    'ytick.labelsize': 9,         # y轴刻度
    'legend.fontsize': 9,         # 图例（如果有）
    'font.family': 'Arial',       # 统一字体类型（论文常用）
})

# ----------------------------
# Utilities
# ----------------------------
rng = np.random.default_rng(7)


def random_invertible_matrix(n: int, scale: float = 1.0) -> np.ndarray:
    """Generate a reasonably-conditioned invertible matrix."""
    while True:
        M = rng.normal(size=(n, n)) * scale
        if abs(np.linalg.det(M)) > 0.2:
            return M


def make_subsystems(n_nodes: int) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Create 2D linear subsystems xdot = A_i x with a mix of stable/unstable max real eigenvalues.
    """
    As = []
    lam_max = []
    for i in range(n_nodes):
        if i < n_nodes // 2:
            target = -0.25 - 0.15 * rng.random()
        else:
            target = 0.12 + 0.18 * rng.random()

        a = target + 0.05 * rng.normal()
        d = target + 0.05 * rng.normal()
        b = 0.25 * rng.normal()
        c = 0.25 * rng.normal()
        A = np.array([[a, b], [c, d]], dtype=float)

        As.append(A)
        lam_max.append(float(np.max(np.real(np.linalg.eigvals(A)))))

    return As, np.array(lam_max, dtype=float)


def build_strongly_connected_digraph(n: int = 6, extra_edges: int = 6) -> nx.DiGraph:
    """Create a strongly connected directed graph with n nodes."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # base directed cycle ensures strong connectivity
    for i in range(n):
        G.add_edge(i, (i + 1) % n)

    # add random extra edges
    while G.number_of_edges() < n + extra_edges:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v:
            G.add_edge(u, v)

    assert nx.is_strongly_connected(G)
    return G


def edge_weights_from_P(G: nx.DiGraph, Ps: list[np.ndarray]) -> dict[tuple[int, int], float]:
    """
    Edge weights: w_ij = log || P_j^{-1} P_i ||_2
    """
    W = {}
    for u, v in G.edges():
        val = np.linalg.norm(np.linalg.inv(Ps[v]) @ Ps[u], ord=2)
        W[(u, v)] = float(np.log(val + 1e-12))
    return W


def phi_cycle(
    cycle_nodes: list[int],
    W: dict[tuple[int, int], float],
    lam_max: np.ndarray,
    stable_set: set[int],
    theta: float = 1.0,
) -> float:
    """
    Minimal runnable surrogate with the same structural form as (16):
      Phi(C) = xi(C) N0^C + (xi(C)/tau_bar + lam_bar_max,C) T_C - theta_C beta_C
    """
    edges = [(cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]) for i in range(len(cycle_nodes))]
    xi = np.mean([W[e] for e in edges])
    T_C = float(len(cycle_nodes))  # dwell time = 1
    tau_bar = 1.0
    lam_bar = float(np.mean(lam_max[cycle_nodes]))
    beta = float(sum(1.0 for v in cycle_nodes if v in stable_set))
    N0 = 0.0
    Phi = xi * N0 + (xi / tau_bar + lam_bar) * T_C - theta * beta
    return float(Phi)


def all_simple_cycles(G: nx.DiGraph, max_len: int = 6) -> list[list[int]]:
    """Enumerate simple directed cycles up to a length cap."""
    cycles = []
    for c in nx.simple_cycles(G):
        if 2 <= len(c) <= max_len:
            cycles.append(c)
    return cycles


def select_most_contracting_cycle(
    G: nx.DiGraph,
    W: dict[tuple[int, int], float],
    lam_max: np.ndarray,
    stable_set: set[int],
    max_len: int = 6,
) -> tuple[list[int], float, list[list[int]], np.ndarray]:
    """
    On small graphs, brute-force enumerate cycles (len<=max_len) and pick the minimum Phi.
    This is Karp-equivalent for minimum mean cycle selection in this toy setup.
    """
    cycles = all_simple_cycles(G, max_len=max_len)
    if not cycles:
        raise RuntimeError("No cycles found (unexpected for strongly connected graph).")
    phis = np.array([phi_cycle(c, W, lam_max, stable_set) for c in cycles], dtype=float)
    idx = int(np.argmin(phis))
    return cycles[idx], float(phis[idx]), cycles, phis


def expm_2x2(A: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Matrix exponential for 2x2 via eigen-decomposition."""
    vals, vecs = np.linalg.eig(A)
    return np.real_if_close(vecs @ np.diag(np.exp(vals * dt)) @ np.linalg.inv(vecs))


def monitored_ratio(N_minus: int, N_plus: int, kmin: float, kmax: float) -> float:
    """r(t) = N^+ kappa^+ / (N^- kappa^-)."""
    if N_minus == 0 or (not np.isfinite(kmin)) or kmin <= 0:
        return 0.0
    return float((N_plus * kmax) / (N_minus * kmin + 1e-12))


def tune_graph_for_balance(target: float = 0.5, trials: int = 250):
    """Try multiple random graphs/subsystems to obtain ~50% contracting cycles (len<=6)."""
    best = None
    for _ in range(trials):
        G = build_strongly_connected_digraph(n=6, extra_edges=6)
        Ps = [random_invertible_matrix(2, scale=1.0) for _ in range(G.number_of_nodes())]
        As, lam_max = make_subsystems(G.number_of_nodes())
        stable_set = {i for i, l in enumerate(lam_max) if l < 0}
        W = edge_weights_from_P(G, Ps)

        cycles = all_simple_cycles(G, max_len=6)
        if len(cycles) < 8:
            continue
        phis = np.array([phi_cycle(c, W, lam_max, stable_set) for c in cycles], dtype=float)
        frac = float(np.mean(phis < 0))

        score = abs(frac - target) + 0.01 * abs(len(cycles) - 20)
        if best is None or score < best[0]:
            best = (score, G, Ps, As, lam_max, stable_set, W, cycles, phis, frac)

    if best is None:
        raise RuntimeError("Failed to find a suitable graph configuration.")
    return best


def main():
    # 1) Build graph with ~50% contracting cycles (Phi<0)
    score, G, Ps, As, lam_max, stable_set, W, cycles, phis, frac = tune_graph_for_balance(target=0.5, trials=250)
    C_comp, Phi_comp, _, _ = select_most_contracting_cycle(G, W, lam_max, stable_set, max_len=6)

    print("=== Graph / cycle stats ===")
    print("Nodes:", list(G.nodes()))
    print("Edges:", G.number_of_edges())
    print("Simple cycles counted (len<=6):", len(cycles))
    print("Contracting fraction (Phi<0):", frac)
    print("Most contracting cycle C_comp:", C_comp, "Phi(C_comp):", Phi_comp)

    # 2) Multi-source shortest-path policy to reach vertex set of C_comp (unit weights -> multi-source BFS)
    targets = set(C_comp)
    Grev = G.reverse(copy=True)

    dist = {v: math.inf for v in G.nodes()}
    next_hop = {v: None for v in G.nodes()}

    dq = deque()
    for t in targets:
        dist[t] = 0
        dq.append(t)

    while dq:
        u = dq.popleft()
        for pred in Grev.successors(u):  # predecessors in original G
            if dist[pred] == math.inf:
                dist[pred] = dist[u] + 1
                next_hop[pred] = u
                dq.append(pred)

    def next_on_cycle(v: int) -> int:
        i = C_comp.index(v)
        return C_comp[(i + 1) % len(C_comp)]

    # 3) Online execution with hysteresis compensation scheduling
    T_steps = 110  # 对应100步输出
    x = np.array([1.0, -0.6], dtype=float)

    x_hist = [x.copy()]
    node_hist = []
    ratio_hist = []
    comp_hist = []

    stack = []
    stack_pos = {}

    N_minus = 0
    N_plus = 0
    kappa_minus_hat = np.inf
    kappa_plus_hat = 0.0

    eps_on = 0.10
    eps_off = 0.25
    r_on = 1 - eps_on
    r_off = 1 - eps_off
    comp_mode = False

    current = int(rng.integers(0, G.number_of_nodes()))
    stack = [current]
    stack_pos = {current: 0}

    for t in range(T_steps):
        node_hist.append(current)

        r = monitored_ratio(N_minus, N_plus, kappa_minus_hat, kappa_plus_hat)
        ratio_hist.append(r)

        if (not comp_mode) and (N_minus > 0) and np.isfinite(kappa_minus_hat) and (kappa_minus_hat > 0) and (r >= r_on):
            comp_mode = True
        if comp_mode and (r <= r_off):
            comp_mode = False

        comp_hist.append(comp_mode)

        if comp_mode:
            if current not in targets:
                nh = next_hop[current]
                nxt = nh if nh is not None else random.choice(list(G.successors(current)))
            else:
                nxt = next_on_cycle(current)
        else:
            nxt = random.choice(list(G.successors(current)))

        expA = expm_2x2(As[current], dt=1.0)
        x = np.real_if_close(expA @ x)
        x_hist.append(x.copy())

        if nxt not in stack_pos:
            stack_pos[nxt] = len(stack)
            stack.append(nxt)
        else:
            k = stack_pos[nxt]
            cycle_nodes = stack[k:]
            Phi = phi_cycle(cycle_nodes, W, lam_max, stable_set)

            if Phi < 0:
                N_minus += 1
                kappa_minus_hat = min(kappa_minus_hat, -Phi)
            else:
                N_plus += 1
                kappa_plus_hat = max(kappa_plus_hat, Phi)

            for v in stack[k + 1 :]:
                stack_pos.pop(v, None)
            stack = stack[: k + 1]

        current = nxt

    x_hist = np.array(x_hist, dtype=float)
    ratio_hist = np.array(ratio_hist, dtype=float)
    comp_hist = np.array(comp_hist, dtype=bool)

    print("\n=== Online results ===")
    print("Final N-:", N_minus, "Final N+:", N_plus)
    print("Final ratio:", monitored_ratio(N_minus, N_plus, kappa_minus_hat, kappa_plus_hat))
    print("Compensation steps:", int(np.sum(comp_hist)))
    print("Total time steps executed:", T_steps)

    # 4) 生成紧凑图片（统一字体+最小空白）
    t_arr = np.arange(len(x_hist))
    norm = np.linalg.norm(x_hist, axis=1)

    # 状态范数图（4x3英寸）
    plt.figure(figsize=(4, 3))
    plt.plot(t_arr, norm)
    plt.xlabel("time step")
    plt.ylabel(r"$\|x(t)\|_2$")
    plt.title("State norm under random switching with compensation")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
    plt.savefig("fig_state_norm.png", dpi=300, bbox_inches='tight')

    # 比率图（4x3英寸）
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(len(ratio_hist)), ratio_hist)
    plt.axhline(r_on, linestyle="--", color='gray')
    plt.axhline(r_off, linestyle="--", color='gray')
    plt.xlabel("time step")
    plt.ylabel("monitored ratio")
    plt.title("Cycle-balance ratio (hysteresis thresholds)")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
    plt.savefig("fig_ratio.png", dpi=300, bbox_inches='tight')

    # 切换执行图（4x3英寸）
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(len(node_hist)), node_hist)
    comp_idx = np.where(comp_hist)[0]
    plt.plot(comp_idx, np.array(node_hist)[comp_idx], linestyle="None", marker="o", markersize=3)
    plt.xlabel("time step")
    plt.ylabel("active node")
    plt.title("Switching execution (compensation mode marked)")
    plt.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
    plt.savefig("fig_switching.png", dpi=300, bbox_inches='tight')

    # 网络图（核心修改：降低高度，从4x4改为4x3英寸）
    pos = nx.spring_layout(G, seed=2)
    plt.figure(figsize=(4, 3))  # 高度从4降到3，和其他图保持一致
    # 节点和标签（统一9号字体）
    nx.draw_networkx_nodes(G, pos, node_size=250)  # 节点稍小，适配高度
    nx.draw_networkx_labels(G, pos, font_size=9)
    # 边和边标签（统一9号字体）
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=8)
    edge_labels = {(u, v): f"{W[(u, v)]:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    # 高亮核心循环
    cycle_edges = [(C_comp[i], C_comp[(i + 1) % len(C_comp)]) for i in range(len(C_comp))]
    nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, width=2, arrows=True, arrowsize=8)
    plt.title(f"Switching graph (bold: C_comp={C_comp})")
    plt.axis("off")
    # 进一步压缩上下空白，适配新高度
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
    plt.savefig("fig_graph.png", dpi=300, bbox_inches='tight')

    plt.close("all")
    print("\nSaved figures (compact, unified font, 100 time steps):")
    print("  fig_state_norm.png")
    print("  fig_ratio.png")
    print("  fig_switching.png")
    print("  fig_graph.png")


if __name__ == "__main__":
    main()