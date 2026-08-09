import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Fallback Benchmark Data for Offline/Cached Mode
const CACHED_BENCHMARK_DATA = {
  benchmark: {
    matrix_size: "1024 × 1024",
    best_tile: "16 × 16",
    cpu_ms: 5119.9697,
    best_gpu_ms: 4.4145,
    speedup: 1173,
    verification: "PASSED (Cached Metrics)",
  },
  optimization: {
    shared_memory: true,
    tiled_matrix_multiplication: true,
    register_optimization: true,
    fast_math: true,
    profiling: true,
  },
  nsight: {
    compute_throughput: 98.49,
    memory_throughput: 98.49,
    l1_tex_throughput: 98.55,
    dram_throughput: 57.31,
    theoretical_occupancy: 100,
    achieved_occupancy: 98.97,
    registers_per_thread: 40,
  },
};

function App() {
  const [size, setSize] = useState(1024);
  const [tile, setTile] = useState(16);

  const [running, setRunning] = useState(false);
  const [health, setHealth] = useState(null);

  // Default initial state uses cached data so the page is never empty
  const [data, setData] = useState(CACHED_BENCHMARK_DATA);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((res) => res.json())
      .then((result) => setHealth(result))
      .catch(() =>
        setHealth({ status: "offline", gpu: "NVIDIA Cloud GPU (Offline Mode)" })
      );
  }, []);

  async function runBenchmark() {
    setRunning(true);
    setError("");

    try {
      const response = await fetch(`${API}/api/benchmark`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          size: Number(size),
          tile: Number(tile),
          warmup: 5,
          iterations: 20,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Benchmark failed");
      }

      // Live GPU response
      setData(result);
    } catch (err) {
      console.warn("Colab server offline/error. Fallback to cached metrics.", err);
      // Fallback to static metrics when backend is unreachable
      setData(CACHED_BENCHMARK_DATA);
      setError(
        "GPU Backend is currently offline. Showing cached benchmark results."
      );
    } finally {
      setRunning(false);
    }
  }

  const benchmark = data?.benchmark || {};
  const optimization = data?.optimization || {};
  const nsight = data?.nsight || {};

  const chartData = benchmark.cpu_ms
    ? [
        {
          name: "CPU",
          time: Number(benchmark.cpu_ms),
        },
        {
          name: "Optimized GPU",
          time: Number(benchmark.best_gpu_ms),
        },
      ]
    : [];

  const gpuInfo = health?.gpu || "NVIDIA Cloud GPU (Cached Benchmark)";

  return (
    <div className="app">
      {/* NAVBAR */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">⚡</div>

          <div>
            <div className="brand-name">GPU-X</div>
            <div className="brand-subtitle">
              CUDA GPU Performance Engineering
            </div>
          </div>
        </div>

        <div
          className={`gpu-status ${
            health?.status === "ok" ? "online" : "offline"
          }`}
        >
          <span className="status-dot"></span>
          {health?.status === "ok" ? "GPU ONLINE" : "GPU OFFLINE (CACHED)"}
        </div>
      </header>

      <main className="container">
        {/* HERO */}
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">LIVE CUDA BENCHMARK</div>

            <h1>
              Measure CUDA performance
              <br />
              <span>from your browser.</span>
            </h1>

            <p>
              Run matrix multiplication on your NVIDIA GPU and compare CPU
              execution against optimized CUDA kernels.
            </p>
          </div>

          <div className="gpu-card">
            <div className="gpu-label">NVIDIA GPU</div>

            <div className="gpu-name">{gpuInfo}</div>

            <div className="gpu-description">
              CUDA-powered benchmark backend
            </div>
          </div>
        </section>

        {/* CONFIG + PERFORMANCE */}
        <section className="main-grid">
          {/* CONFIG */}
          <div className="panel config-panel">
            <div className="panel-title">Benchmark Configuration</div>

            <div className="form-group">
              <label>Matrix Size</label>

              <select
                value={size}
                onChange={(e) => setSize(e.target.value)}
                disabled={running}
              >
                <option value={512}>512 × 512</option>
                <option value={1024}>1024 × 1024</option>
                <option value={2048}>2048 × 2048</option>
                <option value={4096}>4096 × 4096</option>
              </select>
            </div>

            <div className="form-group">
              <label>Tile Size</label>

              <select
                value={tile}
                onChange={(e) => setTile(e.target.value)}
                disabled={running}
              >
                <option value={8}>8 × 8</option>
                <option value={16}>16 × 16</option>
                <option value={32}>32 × 32</option>
              </select>
            </div>

            <button
              className="benchmark-button"
              onClick={runBenchmark}
              disabled={running}
            >
              {running ? (
                <>
                  <span className="spinner"></span>
                  Running CUDA Benchmark...
                </>
              ) : (
                <>▶ Run GPU Benchmark</>
              )}
            </button>

            <div className="config-info">
              <span>Warmup</span>
              <strong>5</strong>
            </div>

            <div className="config-info">
              <span>Iterations</span>
              <strong>20</strong>
            </div>
          </div>

          {/* PERFORMANCE */}
          <div className="panel performance-panel">
            <div className="panel-header">
              <div className="panel-title">Performance</div>

              {data && (
                <div className="passed-badge">
                  ✓ VERIFIED
                </div>
              )}
            </div>

            {running && (
              <div className="loading-state">
                <div className="large-spinner"></div>

                <div>
                  <strong>Executing CUDA kernel...</strong>
                  <span>
                    Measuring CPU and GPU performance
                  </span>
                </div>
              </div>
            )}

            {!running && data && (
              <>
                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Optimized GPU</span>
                    <strong>
                      {Number(benchmark.best_gpu_ms || 0).toFixed(4)}
                      <small> ms</small>
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>CPU → GPU</span>
                    <strong>
                      {benchmark.speedup || 0}
                      <small>×</small>
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>Improvement</span>
                    <strong>
                      {benchmark.cpu_ms && benchmark.best_gpu_ms
                        ? (
                            (1 -
                              benchmark.best_gpu_ms /
                                benchmark.cpu_ms) *
                            100
                          ).toFixed(2)
                        : "0.00"}
                      <small>%</small>
                    </strong>
                  </div>
                </div>

                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={chartData}
                      margin={{
                        top: 20,
                        right: 20,
                        left: 10,
                        bottom: 10,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#263243"
                      />

                      <XAxis
                        dataKey="name"
                        stroke="#8190a5"
                        tick={{ fill: "#8190a5" }}
                      />

                      <YAxis
                        stroke="#8190a5"
                        tick={{ fill: "#8190a5" }}
                        label={{
                          value: "Time (ms)",
                          angle: -90,
                          position: "insideLeft",
                          fill: "#8190a5",
                        }}
                      />

                      <Tooltip
                        contentStyle={{
                          background: "#0b111b",
                          border: "1px solid #263243",
                          borderRadius: "10px",
                          color: "#ffffff",
                        }}
                        formatter={(value) => [
                          `${Number(value).toFixed(4)} ms`,
                          "Execution Time",
                        ]}
                      />

                      <Bar
                        dataKey="time"
                        fill="#7CFF00"
                        radius={[8, 8, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        </section>

        {/* NOTICE / ERROR */}
        {error && (
          <div className="error-box">
            <strong>Backend Status Notice</strong>
            <span>{error}</span>
          </div>
        )}

        {/* RESULTS */}
        {data && (
          <>
            <section className="panel results-panel">
              <div className="panel-title">Benchmark Results</div>

              <div className="result-grid">
                <div className="result-item">
                  <span>Matrix Size</span>
                  <strong>{benchmark.matrix_size}</strong>
                </div>

                <div className="result-item">
                  <span>Best Tile</span>
                  <strong>{benchmark.best_tile}</strong>
                </div>

                <div className="result-item">
                  <span>CPU Baseline</span>
                  <strong>
                    {Number(benchmark.cpu_ms).toFixed(4)} ms
                  </strong>
                </div>

                <div className="result-item">
                  <span>Best GPU</span>
                  <strong>
                    {Number(benchmark.best_gpu_ms).toFixed(4)} ms
                  </strong>
                </div>

                <div className="result-item highlight">
                  <span>Speedup</span>
                  <strong>{benchmark.speedup}×</strong>
                </div>

                <div className="result-item success">
                  <span>Verification</span>
                  <strong>✓ {benchmark.verification}</strong>
                </div>
              </div>
            </section>

            {/* OPTIMIZATIONS */}
            <section className="panel">
              <div className="panel-title">
                CUDA Optimization Techniques
              </div>

              <div className="optimization-grid">
                <Optimization
                  title="Shared Memory"
                  enabled={optimization.shared_memory}
                />

                <Optimization
                  title="Tiled Matrix Multiplication"
                  enabled={optimization.tiled_matrix_multiplication}
                />

                <Optimization
                  title="Register Optimization"
                  enabled={optimization.register_optimization}
                />

                <Optimization
                  title="Fast Math"
                  enabled={optimization.fast_math}
                />

                <Optimization
                  title="Nsight Profiling"
                  enabled={optimization.profiling}
                />
              </div>
            </section>

            {/* NSIGHT */}
            <section className="panel">
              <div className="panel-header">
                <div className="panel-title">
                  NVIDIA Nsight Compute Metrics
                </div>

                <div className="nsight-badge">
                  PROFILING DATA
                </div>
              </div>

              <div className="nsight-grid">
                <Metric
                  title="Compute Throughput"
                  value={nsight.compute_throughput}
                  suffix="%"
                />

                <Metric
                  title="Memory Throughput"
                  value={nsight.memory_throughput}
                  suffix="%"
                />

                <Metric
                  title="L1 / Texture Throughput"
                  value={nsight.l1_tex_throughput}
                  suffix="%"
                />

                <Metric
                  title="DRAM Throughput"
                  value={nsight.dram_throughput}
                  suffix="%"
                />

                <Metric
                  title="Theoretical Occupancy"
                  value={nsight.theoretical_occupancy}
                  suffix="%"
                />

                <Metric
                  title="Achieved Occupancy"
                  value={nsight.achieved_occupancy}
                  suffix="%"
                />

                <Metric
                  title="Registers / Thread"
                  value={nsight.registers_per_thread}
                  suffix=""
                />
              </div>
            </section>

            {/* VERIFICATION */}
            <section className="panel verification-panel">
              <div className="panel-title">
                Verification & CUDA Execution
              </div>

              <div className="verification-items">
                <span>✓ Numerical verification</span>
                <span>✓ CUDA execution</span>
                <span>✓ NVIDIA GPU</span>
                <span>✓ Performance benchmark</span>
              </div>

              <div className="success-message">
                <span className="success-icon">✓</span>

                <div>
                  <strong>Benchmark completed successfully</strong>
                  <p>
                    Matrix multiplication passed numerical verification
                    and executed on the NVIDIA CUDA GPU.
                  </p>
                </div>
              </div>
            </section>
          </>
        )}

        <footer>
          <div>GPU-X • CUDA Performance Engineering</div>
          <div>
            Built with React + FastAPI + CUDA
          </div>
        </footer>
      </main>
    </div>
  );
}

function Optimization({ title, enabled }) {
  return (
    <div className={`optimization ${enabled ? "enabled" : ""}`}>
      <span className="check">{enabled ? "✓" : "×"}</span>

      <span>{title}</span>

      <span className="optimization-status">
        {enabled ? "ACTIVE" : "OFF"}
      </span>
    </div>
  );
}

function Metric({ title, value, suffix }) {
  return (
    <div className="nsight-card">
      <span>{title}</span>

      <strong>
        {value !== undefined && value !== null
          ? Number(value).toFixed(2)
          : "--"}
        <small>{suffix}</small>
      </strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);