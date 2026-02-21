FROM runpod/base:0.6.2-cuda12.4.1

ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
    DTYPE="float16" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
