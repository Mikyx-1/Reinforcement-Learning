"""
NetworkRoutingEnv — packet routing over a fixed network topology.

A modern, graph-observation take on Q-routing (Boyan & Littman, 1993): nodes
are routers, edges are duplex links with finite capacity and transit delay.
Packets are spawned between random (src, dst) pairs; every time a packet
lands on a non-destination node it needs a next-hop decision, which is what
`step()` asks the agent for. Congestion is entirely self-induced — there is
no separate "background traffic" process — so a policy that routes greedily
down the same hub links will learn to see (and cause) its own bottlenecks.

The topology is generated once at construction time and stays fixed across
episodes; `reset()` only clears the dynamic state (in-flight packets, queues,
tick counter). This keeps `action_space`/`observation_space` stable, which a
GNN policy consuming `edge_index` needs anyway.
"""

from collections import deque

import gymnasium as gym
import matplotlib
import networkx as nx
import numpy as np
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

NODE_FEATURE_DIM = 3   # [utilization, is_current, is_destination]
EDGE_FEATURE_DIM = 3   # [utilization, capacity_norm, delay_norm]


class NetworkRoutingEnv(gym.Env):
    """
    Observation: Dict(
        node_features: (num_nodes, 3) float32,
        edge_index:    (2, num_directed_edges) int64  — static, PyG COO format,
        edge_features: (num_directed_edges, 3) float32,
        action_mask:   (max_degree,) int8 — 1 for valid neighbor indices,
    )
    Action: Discrete(max_degree) — index into the current node's sorted
        neighbor list. Indices >= that node's degree are invalid and are
        penalized like a drop.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_nodes: int = 15,
        topology: str = "barabasi_albert",
        topology_m: int = 2,
        topology_seed: int = 42,
        edge_capacity_range: tuple[int, int] = (3, 8),
        edge_delay_range: tuple[int, int] = (1, 5),
        arrival_rate: float = 0.4,
        max_ticks: int = 500,
        max_hops_per_packet: int = 30,
        congestion_penalty_weight: float = 2.0,
        delivery_bonus: float = 5.0,
        drop_penalty: float = 10.0,
        invalid_action_penalty: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        if topology != "barabasi_albert":
            raise ValueError(f"Unsupported topology: {topology!r}")

        self.num_nodes = num_nodes
        self.arrival_rate = arrival_rate
        self.max_ticks = max_ticks
        self.max_hops_per_packet = max_hops_per_packet
        self.congestion_penalty_weight = congestion_penalty_weight
        self.delivery_bonus = delivery_bonus
        self.drop_penalty = drop_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.render_mode = render_mode

        self._max_capacity = edge_capacity_range[1]
        self._max_delay = edge_delay_range[1]

        # ── Static topology (fixed for the env's lifetime) ─────────────────
        self.graph = nx.barabasi_albert_graph(num_nodes, topology_m, seed=topology_seed)
        rng = np.random.default_rng(topology_seed)
        self.directed = nx.DiGraph()
        self.directed.add_nodes_from(self.graph.nodes)
        self._edge_list: list[tuple[int, int]] = []
        for u, v in self.graph.edges():
            capacity = int(rng.integers(edge_capacity_range[0], edge_capacity_range[1] + 1))
            delay = int(rng.integers(edge_delay_range[0], edge_delay_range[1] + 1))
            self.directed.add_edge(u, v, capacity=capacity, delay=delay, in_transit=[])
            self.directed.add_edge(v, u, capacity=capacity, delay=delay, in_transit=[])
            self._edge_list.append((u, v))
            self._edge_list.append((v, u))

        self.neighbors = {n: sorted(self.directed.successors(n)) for n in self.graph.nodes}
        self.max_degree = max(len(ns) for ns in self.neighbors.values())
        self._pos = nx.spring_layout(self.graph, seed=topology_seed)

        num_directed_edges = len(self._edge_list)
        edge_index = np.array(self._edge_list, dtype=np.int64).T  # (2, E)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0, high=1.0, shape=(num_nodes, NODE_FEATURE_DIM), dtype=np.float32
                ),
                "edge_index": spaces.Box(
                    low=0, high=num_nodes - 1, shape=(2, num_directed_edges), dtype=np.int64
                ),
                "edge_features": spaces.Box(
                    low=0.0, high=1.0, shape=(num_directed_edges, EDGE_FEATURE_DIM), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(self.max_degree),
            }
        )
        self.action_space = spaces.Discrete(self.max_degree)
        self.edge_index = edge_index

        # ── Dynamic state, populated by reset() ─────────────────────────────
        self.tick = 0
        self.decision_queue: deque = deque()
        self._current_packet: dict | None = None
        self._packet_id_counter = 0
        self.stats = {"delivered": 0, "dropped": 0, "total_latency": 0, "total_hops": 0}

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        for _, _, data in self.directed.edges(data=True):
            data["in_transit"] = []

        self.tick = 0
        self.decision_queue = deque()
        self._packet_id_counter = 0
        self.stats = {"delivered": 0, "dropped": 0, "total_latency": 0, "total_hops": 0}

        self._spawn_packet()
        self._advance_until_decision()
        self._current_packet = self.decision_queue[0] if self.decision_queue else None

        return self._build_obs(), self._build_info(reward_reason=None)

    def step(self, action: int):
        if not self.decision_queue:
            # Network went quiet before max_ticks (arrival_rate too low). Treat
            # as a truncated episode rather than raising.
            return self._build_obs(), 0.0, False, True, self._build_info(reward_reason="empty")

        pkt = self.decision_queue.popleft()
        current = pkt["current_node"]
        neighbors = self.neighbors[current]

        if action >= len(neighbors):
            reward = -self.invalid_action_penalty
            self.stats["dropped"] += 1
        else:
            nxt = neighbors[action]
            edge_data = self.directed[current][nxt]
            if len(edge_data["in_transit"]) >= edge_data["capacity"]:
                reward = -self.drop_penalty
                self.stats["dropped"] += 1
            else:
                utilization = len(edge_data["in_transit"]) / edge_data["capacity"]
                reward = -1.0 - self.congestion_penalty_weight * utilization
                if nxt == pkt["dest"]:
                    reward += self.delivery_bonus
                pkt["hops"] += 1
                edge_data["in_transit"].append([pkt, edge_data["delay"]])

        self._advance_until_decision()
        self._current_packet = self.decision_queue[0] if self.decision_queue else None

        truncated = self.tick >= self.max_ticks
        return self._build_obs(), reward, False, truncated, self._build_info(reward_reason=None)

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        fig = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        node_util = self._node_utilization()
        node_colors = matplotlib.colormaps["YlOrRd"](node_util)
        nx.draw_networkx_nodes(
            self.graph, self._pos, ax=ax, node_color=node_colors, node_size=400, edgecolors="black"
        )

        pkt = self._current_packet
        if pkt is not None:
            nx.draw_networkx_nodes(
                self.graph, self._pos, nodelist=[pkt["current_node"]], ax=ax,
                node_color="lime", node_size=550, edgecolors="black",
            )
            nx.draw_networkx_nodes(
                self.graph, self._pos, nodelist=[pkt["dest"]], ax=ax,
                node_color="deepskyblue", node_shape="*", node_size=700, edgecolors="black",
            )

        edge_util = [self._edge_utilization(u, v) for u, v in self.graph.edges()]
        nx.draw_networkx_edges(
            self.graph, self._pos, ax=ax, edge_color=edge_util, edge_cmap=matplotlib.colormaps["Reds"],
            edge_vmin=0.0, edge_vmax=1.0, width=2.5,
        )
        nx.draw_networkx_labels(self.graph, self._pos, ax=ax, font_size=8)

        delivered = self.stats["delivered"]
        dropped = self.stats["dropped"]
        avg_latency = self.stats["total_latency"] / delivered if delivered else 0.0
        ax.set_title(
            f"tick {self.tick}/{self.max_ticks}   delivered={delivered}   "
            f"dropped={dropped}   avg latency={avg_latency:.1f}",
            fontsize=10,
        )

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def close(self):
        pass

    # ── Simulation internals ─────────────────────────────────────────────────

    def _spawn_packet(self) -> None:
        src, dest = self.np_random.choice(self.num_nodes, size=2, replace=False)
        pkt = {
            "id": self._packet_id_counter,
            "src": int(src),
            "dest": int(dest),
            "current_node": int(src),
            "hops": 0,
            "created_tick": self.tick,
        }
        self._packet_id_counter += 1
        self.decision_queue.append(pkt)

    def _arrive(self, pkt: dict, node: int) -> None:
        pkt["current_node"] = node
        if node == pkt["dest"]:
            self.stats["delivered"] += 1
            self.stats["total_latency"] += self.tick - pkt["created_tick"]
            self.stats["total_hops"] += pkt["hops"]
        elif pkt["hops"] >= self.max_hops_per_packet:
            self.stats["dropped"] += 1
        else:
            self.decision_queue.append(pkt)

    def _tick(self) -> None:
        for u, v, data in self.directed.edges(data=True):
            if not data["in_transit"]:
                continue
            still_in_transit = []
            for entry in data["in_transit"]:
                entry[1] -= 1
                if entry[1] <= 0:
                    self._arrive(entry[0], v)
                else:
                    still_in_transit.append(entry)
            data["in_transit"] = still_in_transit

        # Poisson arrivals: mean `arrival_rate` packets/tick, unlike a Bernoulli
        # trial this doesn't saturate at 1/tick once arrival_rate >= 1.0.
        for _ in range(self.np_random.poisson(self.arrival_rate)):
            self._spawn_packet()

        self.tick += 1

    def _advance_until_decision(self) -> None:
        while not self.decision_queue and self.tick < self.max_ticks:
            self._tick()

    def _node_utilization(self) -> np.ndarray:
        util = np.zeros(self.num_nodes, dtype=np.float32)
        for n in self.graph.nodes:
            out_edges = [self.directed[n][v] for v in self.neighbors[n]]
            if out_edges:
                util[n] = np.mean([len(e["in_transit"]) / e["capacity"] for e in out_edges])
        return util

    def _edge_utilization(self, u: int, v: int) -> float:
        fwd = self.directed[u][v]
        bwd = self.directed[v][u]
        return max(
            len(fwd["in_transit"]) / fwd["capacity"],
            len(bwd["in_transit"]) / bwd["capacity"],
        )

    def _build_obs(self) -> dict:
        node_features = np.zeros((self.num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        node_features[:, 0] = self._node_utilization()
        if self._current_packet is not None:
            node_features[self._current_packet["current_node"], 1] = 1.0
            node_features[self._current_packet["dest"], 2] = 1.0

        edge_features = np.zeros((len(self._edge_list), EDGE_FEATURE_DIM), dtype=np.float32)
        for i, (u, v) in enumerate(self._edge_list):
            data = self.directed[u][v]
            edge_features[i, 0] = len(data["in_transit"]) / data["capacity"]
            edge_features[i, 1] = data["capacity"] / self._max_capacity
            edge_features[i, 2] = data["delay"] / self._max_delay

        action_mask = np.zeros(self.max_degree, dtype=np.int8)
        if self._current_packet is not None:
            degree = len(self.neighbors[self._current_packet["current_node"]])
            action_mask[:degree] = 1

        return {
            "node_features": node_features,
            "edge_index": self.edge_index,
            "edge_features": edge_features,
            "action_mask": action_mask,
        }

    def _build_info(self, reward_reason: str | None) -> dict:
        info = {"stats": dict(self.stats), "tick": self.tick}
        if self._current_packet is not None:
            info["current_node"] = self._current_packet["current_node"]
            info["destination_node"] = self._current_packet["dest"]
        if reward_reason is not None:
            info["reward_reason"] = reward_reason
        return info


try:
    gym.register(id="NetworkRouting-v0", entry_point="envs.network_routing:NetworkRoutingEnv")
except gym.error.Error:
    pass  # already registered (e.g. module reloaded)
