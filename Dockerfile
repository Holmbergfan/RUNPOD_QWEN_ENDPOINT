FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Redirect HuggingFace cache to the network volume so model downloads
# don't fill up the small container disk (/app).
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
    DTYPE="float16" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface/transformers \
    HF_HUB_DISABLE_XET=1 \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /runpod-volume/.cache/huggingface /runpod-volume/tmp || true

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
