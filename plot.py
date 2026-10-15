import argparse
import os
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

dataset = args.dataset
model = args.model

FILES = {
    "SCAFFOLD": f"results/SCAFFOLD_{dataset}_{model}.pkl",
    "FedAvg": f"results/FedAvg_{dataset}_{model}.pkl",
    "FedProx": f"results/FedProx_{dataset}_{model}.pkl",
    "FedAdam": f"results/FedAdam_{dataset}_{model}.pkl",
    "FDAOPT": f"results/FDAOPT_{dataset}_{model}.pkl",
}

COLORS = {
    "SCAFFOLD": "#24b2d1",
    "FedAvg": "#2ca02c",
    "FedProx": "#1b0db5",
    "FedAdam": "#ff520e",
    "FDAOPT": "#b338a5",
}

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(7, 4))
max_round = 0
for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"[SKIP] {name}: file not found ({path})")
        continue

    print(f"[LOAD] {name}: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    rounds = data["rounds"]
    acc = data["accuracy"]
    down_per_round = data["down_per_round"]
    up_per_round = data["up_per_round"]
    total_down = data["total_down"]
    total_up = data["total_up"]
    max_round = max(max_round, max(rounds))

    down_str = ("N/A" if down_per_round is None
                else f"{down_per_round / 1e6:.2f} MB")
    up_str = ("N/A" if up_per_round is None
              else f"{up_per_round / 1e6:.2f} MB")

    for r, accu in zip(rounds, acc):
        print(f"Round {r}: accuracy={accu:.2f}% | down/round={down_str} | up/round={up_str}")

    print(f"TOTAL communication cost: {(total_down + total_up) / 1e6:.2f} MB")
    print(f"points={len(acc)}, final acc={acc[-1]:.2f}%\n")

    plt.plot(rounds, acc, label=name, color=COLORS.get(name), linewidth=2)

plt.xlabel("Communication Round")
plt.xticks(range(2, max_round + 1, 2))
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

output = f"plots/{dataset}_{model}_accuracy.png"

if os.path.exists(output):
    os.remove(output)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved plot to: {output}")