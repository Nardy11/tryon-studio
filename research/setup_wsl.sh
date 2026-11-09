#!/usr/bin/env bash
# R0 environment for StreamDiffusionV2 inside WSL Ubuntu (no sudo needed).
#   wsl -d Ubuntu -- bash /mnt/c/Users/<you>/Desktop/tryon-studio/research/setup_wsl.sh [uv|env|models|all]
# Everything goes under ~/sdv2 in the Linux filesystem (fast, and keeps C:\ projects clean).
set -euo pipefail
step="${1:-all}"
ROOT="$HOME/sdv2"
mkdir -p "$ROOT" "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

install_uv() {
  # uv: a standalone Python/package manager. Official GitHub release, checksum verified.
  ver=$(curl -sL https://api.github.com/repos/astral-sh/uv/releases/latest | python3 -c 'import sys,json; print(json.load(sys.stdin)["tag_name"])')
  echo "uv $ver"
  mkdir -p "$ROOT/dl" && cd "$ROOT/dl"
  f=uv-x86_64-unknown-linux-gnu.tar.gz
  curl -sLO "https://github.com/astral-sh/uv/releases/download/$ver/$f"
  curl -sLO "https://github.com/astral-sh/uv/releases/download/$ver/$f.sha256"
  sha256sum -c "$f.sha256"
  tar xzf "$f"
  cp uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/"
  uv --version
}

make_env() {
  # Python 3.10 managed by uv; StreamDiffusionV2 from PyPI (Apache 2.0). No flash-attn: it needs a
  # compiler toolchain, and the RTX 4060 runs fine with PyTorch's own attention.
  cd "$ROOT"
  uv python install 3.10
  uv venv --python 3.10 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install streamdiffusionv2 "huggingface_hub[cli]"
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available() and torch.cuda.get_device_name(0))
PY
}

get_models() {
  cd "$ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
  hf download jerryfeng/StreamDiffusionV2 --local-dir ckpts --include "wan_causal_dmd_v2v/*"
  du -sh wan_models ckpts
}

case "$step" in
  uv) install_uv ;;
  env) make_env ;;
  models) get_models ;;
  all) install_uv; make_env; get_models ;;
esac
