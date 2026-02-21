# RUNPOD_QWEN_ENDPOINT

A RunPod serverless endpoint worker for running [Qwen](https://huggingface.co/Qwen) language models via the [RunPod](https://runpod.io) serverless platform.

## Overview

This worker loads a Qwen model at startup and exposes it through RunPod's serverless handler interface. Each job receives a prompt (or a messages list) along with optional generation parameters and returns the model's response.

## Files

| File | Description |
|------|-------------|
| `handler.py` | RunPod serverless worker — loads the model and processes inference jobs |
| `Dockerfile` | Container image definition based on `runpod/base` with CUDA support |
| `requirements.txt` | Python dependencies |
| `test_input.json` | Sample job input for local testing |

## Configuration

The following environment variables control the worker's behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID to load |
| `DTYPE` | `float16` | Model precision (`float16`, `bfloat16`, `float32`) |

## Input Schema

Each job sent to the endpoint must include an `input` object:

```json
{
  "input": {
    "prompt": "What is the capital of France?",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

Alternatively, you can supply a `messages` list (chat template format):

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.7
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | — | Raw text prompt (required unless `messages` is provided) |
| `messages` | array | — | Chat messages list; takes precedence over `prompt` |
| `max_new_tokens` | int | `512` | Maximum number of tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature (set to `0` for greedy decoding) |
| `top_p` | float | `0.9` | Nucleus sampling probability |

## Output Schema

```json
{
  "response": "The capital of France is Paris."
}
```

## Local Testing

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the handler locally with the sample input:
   ```bash
   python handler.py --test_input test_input.json
   ```

## Building the Docker Image

```bash
docker build -t qwen-serverless-worker .
```

## Deploying to RunPod

1. Push the image to a container registry (e.g. Docker Hub, GitHub Container Registry).
2. In the [RunPod console](https://www.runpod.io/console/serverless), create a new **Serverless Endpoint**.
3. Set the container image to your pushed image.
4. Configure the desired GPU type, min/max workers, and environment variables (`MODEL_NAME`, `DTYPE`, etc.).
5. Deploy and test using the **Run** tab or the RunPod API.
