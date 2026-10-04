# Built on unsloth's own maintained image rather than a bare nvidia/cuda base:
# hand-pinning torch/triton/CUDA toolkit ourselves for Blackwell (RTX 50-series,
# sm_120) repeatedly hit native segfaults deep in Triton's JIT compiler
# (unsloth's fused rms_layernorm kernel), across several root-caused-but-wrong
# fixes (xformers ABI mismatch, CUDA 12.4 vs 12.8 ptxas, UNSLOTH_COMPILE_DISABLE).
# unsloth publishes this image specifically so consumers don't have to solve
# that dependency maze themselves -- see unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth.
FROM unsloth/unsloth:latest

ENV PYTHONUNBUFFERED=1

# the base image defaults to a non-root user; apt-get needs root
USER root

WORKDIR /workspace

RUN apt-get update && apt-get install -y git cmake build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp for GGUF export, built once at image build time.
# -j3 (not unbounded -j): full parallelism OOM-killed the 16GB WSL2 VM and
# crashed Docker Desktop's backend mid-compile, reproduced twice.
# requirements.txt is intentionally NOT pip-installed here: it pins a bare
# torch==2.11.0 (resolves to a CPU-only PyPI wheel) and would clobber the
# base image's already-correct Blackwell-compatible torch/triton pairing --
# only the couple of packages convert_hf_to_gguf.py actually needs beyond
# what the base image already ships are installed, with --no-deps so pip's
# resolver can't cascade back into torch.
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /workspace/llama.cpp \
    && pip install --no-cache-dir --no-deps gguf mistral-common sentencepiece protobuf \
    && cmake -B /workspace/llama.cpp/build -S /workspace/llama.cpp -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /workspace/llama.cpp/build --config Release -j3

RUN python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

COPY train.py /workspace/train.py
COPY infer.py /workspace/infer.py
COPY distill.py /workspace/distill.py

ENTRYPOINT ["python"]
CMD ["/workspace/train.py"]
