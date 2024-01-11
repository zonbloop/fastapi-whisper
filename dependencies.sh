#!/bin/bash

# Update package list
apt-get update -y

# Install required packages
apt-get install sox ffmpeg libcairo2 libcairo2-dev -y
apt-get install texlive -y

# Install Python dependencies using pip
pip install -r /workspace/requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000