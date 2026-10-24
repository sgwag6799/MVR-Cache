#!/usr/bin/env bash
# Colab (Python 3.13) setup. Run once per runtime:  source colab_setup.sh
# Keeps Colab's torch/numpy/transformers and installs only what the offline benchmarks need.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip freeze | grep -iE "^(torch|numpy|transformers|pandas|pyarrow)==" > /tmp/mvr_constraints.txt
pip install -q -c /tmp/mvr_constraints.txt \
  langchain-community langchain-openai langchain-anthropic langchain-huggingface \
  langchain-google-genai faiss-cpu chromadb rl4co torchrl tensordict
# chromadb bumps these; google-colab pins them
pip install -q "google-auth==2.49.0" "requests==2.32.4"
# hnswlib pins numpy<2 in metadata only; build against the installed numpy
pip install -q pybind11 setuptools wheel
pip install -q --no-deps --no-build-isolation -e "$REPO/mvr-cache/vcache/vcache_core/cache/embedding_store/hnswlib"

export PYTHONPATH="$REPO/mvr-cache"
export HF_ENDPOINT=https://huggingface.co
export HF_CACHE_BASE=/root/hf
export HF_HOME=/root/hf/home
echo "ready: PYTHONPATH=$PYTHONPATH"
