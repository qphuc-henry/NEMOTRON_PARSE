#!/bin/bash
source nemotron_env/bin/activate
echo "Downloading NVIDIA-Nemotron-Parse-2.0..."
hf download nvidia/NVIDIA-Nemotron-Parse-2.0 --local-dir /home/cqphuc/NVIDIA-Nemotron-Parse-2.0
echo "Download complete!"
