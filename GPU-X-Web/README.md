# GPU-X Web Dashboard + Cloud GPU Backend

A web interface for running the GPU-X CUDA matrix-multiplication benchmark on a remote NVIDIA GPU.

## Architecture

Browser (React/Vite) → FastAPI → NVIDIA GPU → CUDA benchmark → JSON metrics → Dashboard

## Local GPU test

Requirements:
- NVIDIA GPU
- NVIDIA driver
- Docker with NVIDIA Container Toolkit
- Docker Compose

Run:

```bash
docker compose up --build
```

API: http://localhost:8000
Dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Cloud deployment

Deploy the `backend/` container on any GPU VM/container platform that provides:
1. NVIDIA GPU
2. NVIDIA container runtime
3. inbound HTTPS access to port 8000

Set the dashboard environment variable:

```bash
VITE_API_URL=https://YOUR_GPU_BACKEND_DOMAIN
```

Then build the frontend:

```bash
npm run build
```

Deploy `frontend/dist` to a static host such as Vercel or Netlify.

## Important

The included `backend/cuda/benchmark.cu` is a self-contained web-demo benchmark based on the GPU-X tiled matrix multiplication workflow. To expose the exact final GPU-X kernel, replace this file with the project's `benchmark/final_benchmark.cu` and adjust the output parser in `backend/main.py` if its output format differs.
