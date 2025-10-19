# RCKangaroo CUDA build container (Ubuntu 24.04 + CUDA 12.4)
# Usage:
#   docker build -t rckangaroo:cuda12.4 RCKangaroo
#   docker run --rm --gpus all -v $(pwd):/ws -w /ws/RCKangaroo rckangaroo:cuda12.4 make -j
#   # Example run (benchmark mode):
#   docker run --rm --gpus all -v $(pwd):/ws -w /ws/RCKangaroo rckangaroo:cuda12.4 ./rckangaroo -dp 18 -range 84 -max 1.0

FROM nvidia/cuda:12.4.1-devel-ubuntu24.04

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
