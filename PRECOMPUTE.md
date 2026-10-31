### Useful commands for building
- Follow READMe.md to install dependencies
- Then use: `VLLM_USE_PRECOMPILED=1 uv pip install --editable . --no-build-isolation`

### Useful commands for running
VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
          --model meta-llama/Llama-3.2-1B-Instruct \
          --tensor-parallel-size 1 \
          --gpu-memory-utilization 0.8 \
          --enable-prefix-caching \
          --max-model-len 4096 \
          --port 8003

### Running benchmark
python vllm/kalypso/benchmark/client_medec_filter_map_map.py --model meta-llama/Llama-3.2-1B-Instruct
