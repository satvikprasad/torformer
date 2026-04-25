#!/bin/bash
# Ablation: baseline vs toroidal embedding vs toroidal + re-toroidalization.
# Auto-detects GPU count and scales depth/iterations to match available compute.
#
# Run from torformer/nanochat/:
#   bash runs/torus_probe.sh
# With wandb logging:
#   WANDB_RUN=torus-probe bash runs/torus_probe.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# ── PYTHONPATH: make the torus module importable from the nanochat venv ───────
# torus/ lives one directory above nanochat/, so:
#   SCRIPT_DIR  = torformer/nanochat/runs/
#   NANOCHAT    = torformer/nanochat/
#   TORFORMER   = torformer/         <- needs to be on PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORFORMER_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="${TORFORMER_ROOT}:${PYTHONPATH:-}"

# ── uv / venv setup ──────────────────────────────────────────────────────────
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# ── Dataset + tokenizer ───────────────────────────────────────────────────────
# Download only if not already present (safe to re-run)
if [ ! -f "$NANOCHAT_BASE_DIR/tok/tokenizer.model" ]; then
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train
fi

# ── Auto-detect GPU count and scale model ────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    NGPU=$(nvidia-smi --list-gpus | wc -l)
else
    NGPU=0
fi
echo "Detected $NGPU GPU(s)"

# Scale depth and iterations to available compute.
# Head-dim=128 throughout (FA3-optimal). model_dim = depth * 64.
if   [ "$NGPU" -ge 4 ]; then
    DEPTH=12; ASPECT=88; ITERS=3000; DEV_BS=32  # ~125M params, ~1.5h on 4xA100
elif [ "$NGPU" -eq 2 ]; then
    DEPTH=10; ASPECT=77; ITERS=2000; DEV_BS=32  # ~60M  params, ~1h   on 2xA100
elif [ "$NGPU" -eq 1 ]; then
    DEPTH=8;  ASPECT=64; ITERS=1500; DEV_BS=32  # ~25M  params, ~1h   on 1xA100
else
    # CPU fallback (smoke-test only)
    DEPTH=4;  ASPECT=32; ITERS=50;   DEV_BS=1
    echo "WARNING: No GPU detected, running CPU smoke-test (results not meaningful)"
fi

# Multi-GPU launcher
if [ "$NGPU" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nproc_per_node=$NGPU -m"
else
    LAUNCHER="python -m"
fi

# ── Shared hyperparameters ────────────────────────────────────────────────────
COMMON=(
    --depth="$DEPTH"
    --aspect-ratio="$ASPECT"
    --head-dim=128
    --window-pattern=L        # full context, no sliding window
    --max-seq-len=2048
    --device-batch-size="$DEV_BS"
    --num-iterations="$ITERS"
    --eval-every=200
    --eval-tokens=1048576
    --core-metric-every=-1
    --sample-every=500
)

echo "Model: depth=$DEPTH, aspect-ratio=$ASPECT, model_dim=$(( DEPTH * ASPECT )), iters=$ITERS"
echo ""

# ── Run 1: Baseline (standard nn.Embedding) ───────────────────────────────────
echo "============================================================"
echo " Run 1/3: Baseline (nn.Embedding)"
echo "============================================================"
$LAUNCHER scripts.base_train \
    "${COMMON[@]}" \
    --model-tag=probe-baseline \
    --run="${WANDB_RUN}-baseline"

# ── Run 2: Toroidal embedding (B=2, flat torus) ───────────────────────────────
echo "============================================================"
echo " Run 2/3: Toroidal embedding (B=2, flat torus)"
echo "============================================================"
$LAUNCHER scripts.base_train \
    "${COMMON[@]}" \
    --use-toroidal-embed \
    --torus-block-size=2 \
    --model-tag=probe-torus-b2 \
    --run="${WANDB_RUN}-torus-b2"

# ── Run 3: Toroidal embedding + re-toroidalization at every-other layer ────────
# Apply B=4 re-toroidalization at even-indexed layers (0, 2, 4, ...) plus the
# last layer, giving periodic re-projection back onto the manifold.
echo "============================================================"
echo " Run 3/3: Toroidal embedding (B=2) + ReToroidalization (B=4, every other layer)"
echo "============================================================"

RETORUS_LAYERS=$(python3 -c "
depth = $DEPTH
layers = list(range(0, depth, 2))
if (depth - 1) not in layers:
    layers.append(depth - 1)
print(','.join(str(l) for l in layers))
")
echo "  Re-toroidalization at layers: $RETORUS_LAYERS"

$LAUNCHER scripts.base_train \
    "${COMMON[@]}" \
    --use-toroidal-embed \
    --torus-block-size=2 \
    --retorus-layers="$RETORUS_LAYERS" \
    --retorus-block-size=4 \
    --model-tag=probe-torus-b2-retorus \
    --run="${WANDB_RUN}-torus-b2-retorus"

echo "============================================================"
echo " Done. Compare val BPB:"
echo "   probe-baseline  vs  probe-torus-b2  vs  probe-torus-b2-retorus"
echo " Checkpoints: \$NANOCHAT_BASE_DIR/base_checkpoints/"
echo "============================================================"
