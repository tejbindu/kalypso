### Building vLLM scratch
- Reference: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#build-wheel-from-source (with a few modifications)
```
uv pip install -r requirements/cpu-build.txt --torch-backend cpu --index-strategy unsafe-best-match

uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
```
- You have to update transfomers!: `uv pip install --upgrade transformers`

### Fix torchaudio library loading (2026-10-22)

- Symptom: importing torchaudio raised `OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so`. The underlying error was `undefined symbol: torch_library_impl`.
- Cause: the environment had `torch==2.9.1+cpu` and incompatible `torchaudio==2.11.0+cpu`. The CPU requirements pinned torch but left torchaudio unpinned.
- Installed the matching CPU torchaudio wheel, preserving the existing torch and torchvision packages with `--no-deps`:

```bash
.venv/bin/python -m pip install --no-deps torchaudio==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

- Changed `requirements/cpu.txt` to require `torchaudio==2.9.1` (keeping the existing platform condition) so future CPU installs select the matching release. Continue using the CPU backend/index when installing requirements.
- Verified successful imports of torch, torchaudio, and torchvision. Installed versions are `2.9.1+cpu`, `2.9.1+cpu`, and `0.24.1+cpu`, respectively. A small torchaudio resampling operation also passed:

```bash
.venv/bin/python -c 'import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__); print(torchaudio.functional.resample(torch.zeros(1, 160), 16000, 8000).shape)'
# 2.9.1+cpu 2.9.1+cpu 0.24.1+cpu
# torch.Size([1, 80])
```

This verifies the library-loading fix; a full Kalypso workload has not been rerun.
Compatibility reference: https://pytorch.org/get-started/previous-versions/

### Fix NumPy / Numba compatibility (2026-10-22)

- Symptom: `ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.5.`
- Installed Numba 0.61.2 declares `numpy>=1.24,<2.3`; the environment contained NumPy 2.5.4.
- Downgraded NumPy and added `numpy < 2.3` with the same platform condition as Numba in `requirements/cpu.txt`:

```bash
uv pip install --python .venv/bin/python 'numpy>=1.24,<2.3'
.venv/bin/python -c 'import numpy, numba; print(numpy.__version__, numba.__version__)'
# 2.2.6 0.61.2
```

### Gemma server preflight (2026-10-22)

- Verified that `import vllm`, `import transformers`, and `.venv/bin/vllm serve --help` succeed. Installed Transformers version: 5.20.2.
- CLI reports no active Triton driver and disables Triton; this environment is a CPU build.
- The pasted command contains terminal decoration after the first continuation character. Use a backslash immediately followed by a newline, without the `─╯` decoration.

### CPU model server startup findings (2026-10-22)

- Removed `--gpu-memory-utilization` for CPU serving; vLLM did not include that GPU-only setting in its parsed non-default arguments.
- The initial local server attempt could not bind port 8003 inside the command sandbox (`PermissionError: [Errno 1] Operation not permitted`). The Slurm run had permission to bind.
- The original `--tensor-parallel-size 4` failed CPU auto-binding because this process could see two NUMA nodes for four workers. `VLLM_CPU_OMP_THREADS_BIND=nobind OMP_NUM_THREADS=4` passed worker initialization.
- vLLM initially reserved 15.72 GiB for the CPU KV cache. Set `VLLM_CPU_KVCACHE_SPACE=4` for this test; this leaves room for model weights and respects the 32K context request.
- `google/gemma-4-E2B-it` progressed to the Transformers fallback but failed when vLLM's generic Transformers backend requested one global `head_dim`. Transformers correctly reported that Gemma 4 has per-layer head dimensions (256 for most layers and 512 for selected full-attention layers). Do not suppress this guard: the generic backend currently constructs attention instances with one shared head size. Supporting Gemma 4 requires per-layer head-dimension support in that backend or a vLLM-native Gemma 4 implementation.
- Switched the test to `meta-llama/Llama-3.2-1B` as requested. Its server invocation reached Hugging Face and failed with `401 Client Error: You are trying to access a gated repo`. The checkpoint is not cached locally, and no Hugging Face token is configured in this environment. The Hugging Face model page requires agreeing to Meta's access terms and authenticating.
- After accepting Meta's access terms, the Hugging Face token in the local `hf_token` file was passed through `HF_TOKEN` without printing it. The file is mode `600` and root-ignored by Git.
- The server started successfully in Slurm job `65978995` on `uri-cpu007` with 16 CPUs and 48 GiB RAM. Model weights and tokenizer loaded with the Transformers fallback disabled for this model (vLLM's native Llama implementation was selected).
- Verified `GET /health` succeeded and a short `POST /v1/completions` returned generated text (`" jumps over the lazy dog.\nThe quick"`, 8 tokens). The server remains running in the allocation.
- To repeat the run, keep the token out of command output and load it from the local file first:

```bash
read -r HF_TOKEN < hf_token
export HF_TOKEN
export HF_HUB_CACHE=/scratch4/workspace/ypathak_umass_edu-acads/hf-cache/hub
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_CPU_OMP_THREADS_BIND=nobind \
OMP_NUM_THREADS=4 \
VLLM_CPU_KVCACHE_SPACE=4 \
.venv/bin/vllm serve meta-llama/Llama-3.2-1B \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8003
```
