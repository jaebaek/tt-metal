# TT-METAL UBUNTU 20.04 AMD64 WORMHOLE_B0 DOCKERFILE
FROM ghcr.io/tenstorrent/tt-metal/ubuntu-20.04-amd64:latest

ARG GITHUB_BRANCH=main

ENV ARCH_NAME=wormhole_b0

RUN git clone https://github.com/tenstorrent/tt-metal.git --depth 1 -b ${GITHUB_BRANCH} /opt/tt-metal --recurse-submodules
RUN cd tt-metal \
    && pip install -e . \
    && pip install -e ttnn

CMD ["tail", "-f", "/dev/null"]
