"""
Online cycle-balance simulation for a switched linear system on a strongly connected digraph.

What it does (aligned with Problem 2 + Algorithm):
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
