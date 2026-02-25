docker run -it --rm \
  -p 7860:7860 \
  -v ./data:/data \
  -v ./envdir:/envdir \
  ghcr.io/meyeroppelt/cellatria:latest cellatria \
  --env_path /envdir