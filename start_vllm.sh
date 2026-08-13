#!/bin/bash
source /home/cqphuc/nemotron_env/bin/activate
echo "Starting vLLM API Server for NVIDIA-Nemotron-Parse-2.0..."
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0/vllm_tied_patch${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm.entrypoints.openai.api_server \
    --model /home/cqphuc/NVIDIA-Nemotron-Parse-2.0 \
    --dtype bfloat16 \
    --max-num-seqs 8 \
    --trust-remote-code \
    --chat-template /home/cqphuc/NVIDIA-Nemotron-Parse-2.0/chat_template.jinja \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt '{"image":1}' \
    --max-model-len 9000 \
    --gpu-memory-utilization 0.9
