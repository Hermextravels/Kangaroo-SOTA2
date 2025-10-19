# RCKangaroo CUDA build container (configurable base image)
# Usage:
#   # default BASE_IMAGE is set below; override with --build-arg BASE_IMAGE=...
#   docker build -t rckangaroo:cuda12.4 RCKangaroo
#   docker run --rm --gpus all -v $(pwd):/ws -w /ws/RCKangaroo rckangaroo:cuda12.4 make -j
#   # Example run (benchmark mode):
#   docker run --rm --gpus all -v $(pwd):/ws -w /ws/RCKangaroo rckangaroo:cuda12.4 ./rckangaroo -dp 18 -range 84 -max 1.0

# You can override BASE_IMAGE at build time if a tag is not available in your region/registry.
# Known good examples:
#   nvidia/cuda:12.4.0-devel-ubuntu22.04
#   nvidia/cuda:12.6.2-devel-ubuntu24.04
#   nvidia/cuda:12.2.0-devel-ubuntu22.04
ARG BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python3 \
    bc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/RCKangaroo
COPY . /ws/RCKangaroo

# Build binary
RUN make -j4

CMD ["bash", "-lc", "./rckangaroo -max 0.1"]
