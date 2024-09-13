# End-to-end benchmark with Triton FIL backend

## Benchmark results on AWS EC2
We ran the benchmark (see below) on AWS EC2 and measured the throughput
and latency for end-to-end inference. See `aws_ec2_benchmark_results/summary.ipynb`.

## How to run benchmark

First, train an XGBoost model by running `train.py`. The script will
save the model file to `model_repository/xgb_model/1/xgboost.json`.

Then invokte the Triton model analyzer using the following commands:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.08-py3
docker pull nvcr.io/nvidia/tritonserver:24.08-py3-sdk

# GPU
docker run -it --gpus '"device=0"' \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):$(pwd) --net=host \
    nvcr.io/nvidia/tritonserver:24.08-py3-sdk \
    model-analyzer profile \
    --model-repository $(pwd)/model_repository/ \
    --triton-launch-mode=docker \
    --output-model-repository-path tmp \
    --export-path $(pwd)/profile_results_gpu \
    -f $(pwd)/search_config_gpu.yaml

# CPU
docker run -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):$(pwd) --net=host \
    nvcr.io/nvidia/tritonserver:24.08-py3-sdk \
    model-analyzer profile \
    --model-repository $(pwd)/model_repository/ \
    --triton-launch-mode=docker \
    --output-model-repository-path tmp \
    --export-path $(pwd)/profile_results_cpu \
    -f $(pwd)/search_config_cpu.yaml
```
