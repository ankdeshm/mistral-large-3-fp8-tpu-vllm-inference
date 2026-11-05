# Reference image build - not used for the measured runs.
# The measured runs used the image referenced in k8s/v7x16-ml3-suite.yaml (vllm 0.32.0 + vllm-torchtpu, with the
# vllm-torchtpu source tree at /workspace/vllm-torchtpu). scripts/apply_ml3_patches.py edits files under
# /workspace/vllm-torchtpu/src/vllm_torchtpu, so the base image must provide that tree.
ARG BASE_IMAGE=vllm/vllm-tpu:nightly-20261029-75749f8-2fb320d
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl jq ca-certificates procps && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "huggingface_hub[hf_transfer]" \
      hf_transfer \
      google-cloud-storage \
      aiohttp \
      requests && \
    (pip install --no-cache-dir "runai-model-streamer>=0.11.0" || true)

RUN git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
    cd /workspace/inferencex && \
    git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd

COPY scripts/apply_ml3_patches.py /workspace/apply_ml3_patches.py
COPY scripts/run_tpu_benchmark_sweep.py /workspace/run_tpu_benchmark_sweep.py
COPY scripts/stage_hf_to_gcs.py /workspace/stage_hf_to_gcs.py
RUN python3 /workspace/apply_ml3_patches.py && \
    chmod +x /workspace/apply_ml3_patches.py /workspace/run_tpu_benchmark_sweep.py /workspace/stage_hf_to_gcs.py
