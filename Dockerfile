FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
    DTYPE="float16" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
