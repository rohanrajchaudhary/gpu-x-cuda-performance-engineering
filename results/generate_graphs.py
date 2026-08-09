import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA
# ============================================================

cpu_time = 5119.9697
naive_gpu = 5.8519
optimized_gpu = 4.4236

tile_sizes = ["8x8", "16x16", "32x32"]
tile_times = [6.3988, 4.4687, 4.9898]

optimization_names = [
    "Naive GPU",
    "Tiled GPU",
    "Phase 7B",
    "Fast Math",
]

optimization_times = [
    5.8519,
    4.4687,
    4.4257,
    4.4145,
]

speedup_names = [
    "CPU → Naive GPU",
    "CPU → Optimized GPU",
]

speedup_values = [
    874.9190,
    1157.4257,
]

# ============================================================
# GRAPH 1 — CPU VS GPU
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    ["CPU", "Naive GPU", "Optimized GPU"],
    [cpu_time, naive_gpu, optimized_gpu]
)

plt.ylabel("Execution Time (ms)")
plt.title("CPU vs GPU Matrix Multiplication")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cpu_vs_gpu.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 2 — TILE SIZE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    tile_sizes,
    tile_times
)

plt.xlabel("Tile Size")
plt.ylabel("Kernel Time (ms)")
plt.title("CUDA Tile Size Performance")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "tile_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 3 — OPTIMIZATION PROGRESSION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    optimization_names,
    optimization_times,
    marker="o"
)

plt.ylabel("Kernel Time (ms)")
plt.title("GPU-X Optimization Progression")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "optimization_progress.png"),
    dpi=300
)

plt.close()

# ============================================================
# GRAPH 4 — SPEEDUP
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    speedup_names,
    speedup_values
)

plt.ylabel("Speedup (x)")
plt.title("GPU Speedup Over CPU")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "speedup_comparison.png"),
    dpi=300
)

plt.close()

print()
print("========================================")
print("      GPU-X GRAPH GENERATION")
print("========================================")
print()
print("Graphs generated successfully!")
print()
print("Output directory:")
print("./results/graphs/")
print()
print("Files:")
print("1. cpu_vs_gpu.png")
print("2. tile_comparison.png")
print("3. optimization_progress.png")
print("4. speedup_comparison.png")
print()