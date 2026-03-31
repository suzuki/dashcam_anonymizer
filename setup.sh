#!/bin/bash
set -e
uv sync --extra dev
mkdir -p model
echo "Downloading the YOLO model..."
uv run gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe -O model/best.pt
echo "Setup complete!"
