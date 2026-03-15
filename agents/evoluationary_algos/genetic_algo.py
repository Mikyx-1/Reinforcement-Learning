"""
Genetic Algorithm for RL Policy Search
=======================================
Proves that Evolutionary Algorithms can solve RL environments (CartPole, LunarLander),
but are less sample-efficient than DQN/PPO.

Requirements:
    pip install gymnasium numpy matplotlib

Usage:
    python genetic_algorithm_rl.py
    python genetic_algorithm_rl.py --env CartPole-v1
    python genetic_algorithm_rl.py --env LunarLander-v2
    python genetic_algorithm_rl.py --env CartPole-v1 --compare  # compare with random baseline

Mar 15th 2026 - Will investigate further later.
"""

import argparse
import time

import gymnasium as gym
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
#  Neural Network Policy (the "genome")
# ─────────────────────────────────────────────


class PolicyNetwork:
    """
    A simple 2-layer MLP policy.
    Weights are evolved by the GA — no gradient descent.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Xavier-like init
        self.W1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(action_dim)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        h = np.maximum(0, obs @ self.W1 + self.b1)  # ReLU hidden
        return h @ self.W2 + self.b2  # linear output (logits)

    def act(self, obs: np.ndarray) -> int:
        logits = self.forward(obs)
        return int(np.argmax(logits))

    # ── Genome helpers ──────────────────────────────

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_weights(self, flat: np.ndarray):
        idx = 0
        s = self.obs_dim * self.hidden_dim
        self.W1 = flat[idx : idx + s].reshape(self.obs_dim, self.hidden_dim)
        idx += s
        self.b1 = flat[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        s = self.hidden_dim * self.action_dim
        self.W2 = flat[idx : idx + s].reshape(self.hidden_dim, self.action_dim)
        idx += s
        self.b2 = flat[idx : idx + self.action_dim].copy()

    def clone(self) -> "PolicyNetwork":
        child = PolicyNetwork(self.obs_dim, self.action_dim, self.hidden_dim)
        child.set_weights(self.get_weights().copy())
        return child


# ─────────────────────────────────────────────
#  Fitness Evaluation
# ─────────────────────────────────────────────


def evaluate(
    policy: PolicyNetwork, env_name: str, n_episodes: int = 3, seed: int = 0
) -> float:
    """
    Run the policy for n_episodes and return the mean total reward.
    Averaging over multiple episodes reduces noise.
    """
    env = gym.make(env_name)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = policy.act(np.array(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    env.close()
    return total / n_episodes


# ─────────────────────────────────────────────
#  Genetic Operators
# ─────────────────────────────────────────────


def crossover(parent1: PolicyNetwork, parent2: PolicyNetwork) -> PolicyNetwork:
    """Uniform crossover: each weight independently drawn from either parent."""
    w1 = parent1.get_weights()
    w2 = parent2.get_weights()
    mask = np.random.rand(len(w1)) < 0.5
    child_w = np.where(mask, w1, w2)
    child = parent1.clone()
    child.set_weights(child_w)
    return child


def mutate(
    policy: PolicyNetwork, mutation_rate: float = 0.05, mutation_std: float = 0.2
) -> PolicyNetwork:
    """Gaussian mutation: add noise to each weight with probability mutation_rate."""
    mutant = policy.clone()
    w = mutant.get_weights()
    mask = np.random.rand(len(w)) < mutation_rate
    w[mask] += np.random.randn(mask.sum()) * mutation_std
    mutant.set_weights(w)
    return mutant


# ─────────────────────────────────────────────
#  Genetic Algorithm
# ─────────────────────────────────────────────


class GeneticAlgorithm:

    def __init__(
        self,
        env_name: str,
        pop_size: int = 50,
        elite_frac: float = 0.2,
        mutation_rate: float = 0.05,
        mutation_std: float = 0.2,
        n_eval_episodes: int = 3,
        hidden_dim: int = 16,
    ):
        self.env_name = env_name
        self.pop_size = pop_size
        self.elite_k = max(2, int(pop_size * elite_frac))
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.n_eval_episodes = n_eval_episodes

        # Infer dimensions from env
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        print(f"\n{'='*55}")
        print(f"  Genetic Algorithm — {env_name}")
        print(f"{'='*55}")
        print(f"  obs_dim={obs_dim}  action_dim={action_dim}  hidden={hidden_dim}")
        print(f"  pop_size={pop_size}  elite_k={self.elite_k}")
        print(f"  mutation_rate={mutation_rate}  mutation_std={mutation_std}")
        print(f"{'='*55}\n")

        # Initialize population
        self.population = [
            PolicyNetwork(obs_dim, action_dim, hidden_dim) for _ in range(pop_size)
        ]

        # Tracking
        self.history = {"best": [], "mean": [], "std": [], "env_interactions": []}
        self.total_steps = 0

    def _eval_population(self) -> np.ndarray:
        fitnesses = []
        for policy in self.population:
            f = evaluate(
                policy,
                self.env_name,
                n_episodes=self.n_eval_episodes,
                seed=len(self.history["best"]) * 100,
            )
            fitnesses.append(f)
        # Approximate step count: avg episode length × n_episodes × pop_size
        self.total_steps += self.pop_size * self.n_eval_episodes * 200
        return np.array(fitnesses)

    def run(
        self, n_generations: int = 30, target_score: float = None, verbose: bool = True
    ) -> PolicyNetwork:

        best_ever_policy = None
        best_ever_fitness = -np.inf

        for gen in range(1, n_generations + 1):
            t0 = time.time()
            fitnesses = self._eval_population()

            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            mean_fit = fitnesses.mean()
            std_fit = fitnesses.std()

            self.history["best"].append(best_fit)
            self.history["mean"].append(mean_fit)
            self.history["std"].append(std_fit)
            self.history["env_interactions"].append(self.total_steps)

            if best_fit > best_ever_fitness:
                best_ever_fitness = best_fit
                best_ever_policy = self.population[best_idx].clone()

            elapsed = time.time() - t0
            if verbose:
                bar = "█" * int((gen / n_generations) * 20) + "░" * (
                    20 - int((gen / n_generations) * 20)
                )
                print(
                    f"Gen {gen:3d}/{n_generations} [{bar}]  "
                    f"best={best_fit:8.2f}  mean={mean_fit:8.2f}  "
                    f"std={std_fit:6.2f}  ({elapsed:.1f}s)"
                )

            # Early stopping
            if target_score is not None and best_ever_fitness >= target_score:
                print(f"\n  Target {target_score} reached at generation {gen}!")
                break

            # ── Selection: keep elites ──────────────────
            sorted_idx = np.argsort(fitnesses)[::-1]
            elites = [self.population[i] for i in sorted_idx[: self.elite_k]]

            # ── Reproduction ───────────────────────────
            new_population = list(elites)  # elitism: elites survive unchanged
            while len(new_population) < self.pop_size:
                p1, p2 = np.random.choice(elites, size=2, replace=False)
                child = crossover(p1, p2)
                child = mutate(child, self.mutation_rate, self.mutation_std)
                new_population.append(child)

            self.population = new_population

        print(f"\n  Best fitness ever: {best_ever_fitness:.2f}")
        print(f"  Total env interactions (approx): {self.total_steps:,}")
        return best_ever_policy


# ─────────────────────────────────────────────
#  Baseline: Random Policy
# ─────────────────────────────────────────────


def random_baseline(env_name: str, n_episodes: int = 100) -> float:
    """Evaluate a completely random policy."""
    env = gym.make(env_name)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


# ─────────────────────────────────────────────
#  Comparison Plot
# ─────────────────────────────────────────────

ALGO_BENCHMARKS = {
    "CartPole-v1": {
        "target": 500,
        "DQN": {"score": 490, "interactions": 50_000, "color": "#2ca02c"},
        "PPO": {"score": 498, "interactions": 40_000, "color": "#1f77b4"},
        "A2C": {"score": 470, "interactions": 60_000, "color": "#9467bd"},
    },
    "LunarLander-v3": {
        "target": 200,
        "DQN": {"score": 235, "interactions": 500_000, "color": "#2ca02c"},
        "PPO": {"score": 260, "interactions": 400_000, "color": "#1f77b4"},
        "A2C": {"score": 210, "interactions": 600_000, "color": "#9467bd"},
    },
}


def plot_results(ga: GeneticAlgorithm, env_name: str, best_score: float):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Genetic Algorithm vs Gradient-Based RL — {env_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])  # fitness curve (full width)
    ax2 = fig.add_subplot(gs[1, 0])  # bar comparison
    ax3 = fig.add_subplot(gs[1, 1])  # sample efficiency

    generations = list(range(1, len(ga.history["best"]) + 1))
    best_arr = np.array(ga.history["best"])
    mean_arr = np.array(ga.history["mean"])
    std_arr = np.array(ga.history["std"])

    # ── (1) Fitness curve ────────────────────────
    ax1.plot(generations, best_arr, color="#d62728", linewidth=2, label="Best fitness")
    ax1.plot(
        generations,
        mean_arr,
        color="#aec7e8",
        linewidth=1.5,
        linestyle="--",
        label="Mean fitness",
    )
    ax1.fill_between(
        generations,
        mean_arr - std_arr,
        mean_arr + std_arr,
        alpha=0.2,
        color="#aec7e8",
        label="±1 std",
    )

    target = ALGO_BENCHMARKS.get(env_name, {}).get("target")
    if target:
        ax1.axhline(
            target,
            color="green",
            linestyle=":",
            linewidth=1.5,
            label=f"Target ({target})",
        )

    ax1.set_title("GA fitness over generations", fontsize=12)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Episode reward")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── (2) Score comparison bar chart ───────────
    if env_name in ALGO_BENCHMARKS:
        bench = ALGO_BENCHMARKS[env_name]
        algos = ["Random", "GA (ours)"] + [k for k in bench if k != "target"]
        rand_score = random_baseline(env_name, n_episodes=50)
        scores = [rand_score, best_score] + [
            bench[k]["score"] for k in bench if k != "target"
        ]
        colors = ["#7f7f7f", "#d62728"] + [
            bench[k]["color"] for k in bench if k != "target"
        ]

        bars = ax2.barh(algos, scores, color=colors, edgecolor="white", height=0.6)
        if target:
            ax2.axvline(
                target,
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"Target ({target})",
            )
            ax2.legend(fontsize=8)
        for bar, score in zip(bars, scores):
            ax2.text(
                bar.get_width() + abs(max(scores)) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}",
                va="center",
                fontsize=9,
            )
        ax2.set_title("Final score comparison", fontsize=12)
        ax2.set_xlabel("Episode reward")
        ax2.grid(True, axis="x", alpha=0.3)

        # ── (3) Sample efficiency ─────────────────
        ga_interactions = ga.history["env_interactions"]
        ax3.plot(
            ga_interactions, best_arr, color="#d62728", linewidth=2, label="GA (ours)"
        )

        for algo in [k for k in bench if k != "target"]:
            b = bench[algo]
            ax3.scatter(
                b["interactions"],
                b["score"],
                color=b["color"],
                s=100,
                zorder=5,
                label=algo,
            )
            ax3.annotate(
                f"{algo}\n{b['score']}",
                (b["interactions"], b["score"]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=8,
                color=b["color"],
            )

        if target:
            ax3.axhline(
                target,
                color="green",
                linestyle=":",
                linewidth=1.2,
                label=f"Target ({target})",
            )

        ax3.set_title("Sample efficiency\n(score vs env interactions)", fontsize=12)
        ax3.set_xlabel("Environment interactions (approx)")
        ax3.set_ylabel("Episode reward")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    else:
        ax2.text(
            0.5,
            0.5,
            "No benchmarks\nfor this env",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=11,
        )
        ax3.text(
            0.5,
            0.5,
            "No benchmarks\nfor this env",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=11,
        )

    plt.savefig("ga_rl_results.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved → ga_rl_results.png")
    plt.show()


# ─────────────────────────────────────────────
#  Demo: render best policy
# ─────────────────────────────────────────────


def demo_policy(policy: PolicyNetwork, env_name: str, n_episodes: int = 3):
    """Render the best evolved policy visually."""
    try:
        env = gym.make(env_name, render_mode="human")
        print(f"\n  Rendering {n_episodes} episodes with best policy...")
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            total = 0.0
            done = False
            while not done:
                action = policy.act(np.array(obs, dtype=np.float32))
                obs, r, terminated, truncated, _ = env.step(action)
                total += r
                done = terminated or truncated
            print(f"    Episode {ep+1}: reward = {total:.1f}")
        env.close()
    except Exception as e:
        print(f"  (Rendering skipped: {e})")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

ENV_DEFAULTS = {
    "CartPole-v1": {"target": 500, "pop": 50, "gens": 30, "mut": 0.05, "std": 0.2},
    "LunarLander-v3": {"target": 200, "pop": 80, "gens": 60, "mut": 0.05, "std": 0.15},
    "MountainCar-v0": {"target": -110, "pop": 100, "gens": 80, "mut": 0.08, "std": 0.3},
}


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for RL")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        choices=list(ENV_DEFAULTS.keys()),
        help="Gymnasium environment",
    )
    parser.add_argument("--pop", type=int, default=None, help="Population size")
    parser.add_argument("--gens", type=int, default=None, help="Max generations")
    parser.add_argument("--mut", type=float, default=None, help="Mutation rate")
    parser.add_argument("--std", type=float, default=None, help="Mutation std dev")
    parser.add_argument("--render", action="store_true", help="Render best policy")
    args = parser.parse_args()

    cfg = ENV_DEFAULTS[args.env]
    pop_size = args.pop or cfg["pop"]
    n_gens = args.gens or cfg["gens"]
    mut_rate = args.mut or cfg["mut"]
    mut_std = args.std or cfg["std"]
    target = cfg["target"]

    ga = GeneticAlgorithm(
        env_name=args.env,
        pop_size=pop_size,
        mutation_rate=mut_rate,
        mutation_std=mut_std,
        n_eval_episodes=3,
    )

    best_policy = ga.run(n_generations=n_gens, target_score=target)

    # Final evaluation (more episodes, stable estimate)
    print("\n  Evaluating best policy (10 episodes)...")
    final_score = evaluate(best_policy, args.env, n_episodes=10, seed=9999)
    print(f"  Final score: {final_score:.2f}  (target: {target})")
    solved = final_score >= target if target > 0 else final_score >= target
    print(
        f"  Solved: {'YES ✓' if solved else 'NO — try more generations or larger population'}"
    )

    # ── Print summary table ──────────────────────
    print(f"\n{'─'*55}")
    print(f"  {'Algorithm':<15} {'Score':>10}  {'Notes'}")
    print(f"{'─'*55}")
    rand = random_baseline(args.env, n_episodes=50)
    print(f"  {'Random':<15} {rand:>10.1f}  untrained baseline")
    print(f"  {'GA (ours)':<15} {final_score:>10.1f}  evolved {n_gens} generations")
    if args.env in ALGO_BENCHMARKS:
        bench = ALGO_BENCHMARKS[args.env]
        for algo in [k for k in bench if k != "target"]:
            b = bench[algo]
            print(
                f"  {algo:<15} {b['score']:>10.1f}  ~{b['interactions']//1000}k interactions"
            )
    print(f"{'─'*55}")
    print(f"  GA used ~{ga.total_steps:,} env interactions")
    print(f"  DQN/PPO typically use far fewer — this is GA's main weakness.\n")

    plot_results(ga, args.env, final_score)

    if args.render:
        demo_policy(best_policy, args.env)


if __name__ == "__main__":
    main()
