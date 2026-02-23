FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Redirect HuggingFace cache to the network volume so model downloads
# don't fill up the small container disk (/app).
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
    DTYPE="float16" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
