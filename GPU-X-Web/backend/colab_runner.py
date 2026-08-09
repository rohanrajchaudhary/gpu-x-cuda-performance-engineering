# Colab Automated Deployment & Tunnel Script
import os, subprocess, time, re

# 1. Setup Cloudflare Binary
subprocess.run(["wget", "-q", "-O", "cloudflared", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
subprocess.run(["chmod", "+x", "cloudflared"])

# 2. Start Uvicorn Server in Background
subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(2)

# 3. Start Cloudflare Tunnel and Print URL
tunnel_proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stderr=subprocess.PIPE, text=True)

url = None
for line in iter(tunnel_proc.stderr.readline, ""):
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        url = match.group(0)
        break

print("\n" + "="*60)
print(f"🚀 LIVE BACKEND URL: {url}")
print("="*60 + "\n")