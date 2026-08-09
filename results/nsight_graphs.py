import os
import matplotlib.pyplot as plt

OUTPUT_DIR = ".\graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ACTUAL NSIGHT COMPUTE RESULTS
# ============================================================

occupancy_names = [
    "Theoretical Occupancy",
    "Achieved Occupancy"
]

occupancy_values = [
    100.0,
    98.97
]

throughput_names = [
    "Compute",
    "Memory",
    "L1/TEX",
    "DRAM"
]

throughput_values = [
    98.49,
    98.49,
    98.55,
    57.31
]

# ============================================================
# GRAPH 1 — OCCUPANCY
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    occupancy_names,
    occupancy_values
)

plt.ylabel("Occupancy (%)")
plt.title("GPU-X Nsight Compute Occupancy")

plt.ylim(0, 110)

for bar, value in zip(bars, occupancy_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "nsight_occupancy.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2 — THROUGHPUT
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    throughput_names,
    throughput_values
)

plt.ylabel("Throughput (%)")
plt.title("GPU-X Nsight Compute Throughput")

plt.ylim(0, 110)

for bar, value in zip(bars, throughput_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "nsight_throughput.png"),
    dpi=300
)

plt.close()

print()
print("========================================")
print("      GPU-X NSIGHT GRAPH GENERATION")
print("========================================")
print()
print("Graphs generated successfully!")
print()
print("1. nsight_occupancy.png")
print("2. nsight_throughput.png")
print()