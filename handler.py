import os
import traceback

import runpod

WORKER_KIND = "qwen-text"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DTYPE = os.environ.get("DTYPE", "float16").lower()
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")

# Prefer a locally downloaded copy on the network volume over HuggingFace.
# Checks /workspace/models/<basename> first (e.g. Qwen2.5-7B-Instruct).
_local_candidate = os.path.join("/workspace/models", MODEL_NAME.split("/")[-1])
LOAD_PATH = _local_candidate if os.path.isdir(_local_candidate) else MODEL_NAME

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
tokenizer = None
model = None
model_init_error = None


def initialize_model():
    global torch
    global AutoModelForCausalLM
    global AutoTokenizer
    global tokenizer
    global model
    global model_init_error

    print(f"Worker booting. kind={WORKER_KIND}, MODEL_NAME={MODEL_NAME}, DTYPE={DTYPE}, DEVICE_MAP={DEVICE_MAP}")

    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        torch = _torch
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
    except Exception:
        model_init_error = traceback.format_exc()
        print("Dependency import failed with traceback:")
        print(model_init_error)
        return

    try:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if DTYPE not in dtype_map:
            print(f"Warning: unrecognized DTYPE '{DTYPE}', falling back to float16.")
        torch_dtype = dtype_map.get(DTYPE, torch.float16)

        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            print("Warning: torch.cuda.is_available() failed, assuming CPU.")
            print(traceback.format_exc())

        if not cuda_available and torch_dtype == torch.float16:
            print("CUDA not available and DTYPE=float16; falling back to float32 for CPU compatibility.")
            torch_dtype = torch.float32

        print(f"Startup config -> MODEL_NAME={MODEL_NAME}, DTYPE={torch_dtype}, DEVICE_MAP={DEVICE_MAP}")
        print(f"Torch version: {torch.__version__}, CUDA available: {cuda_available}")

        if cuda_available:
            try:
                device_count = torch.cuda.device_count()
                print(f"CUDA device count: {device_count}")
                for device_id in range(device_count):
                    print(f"GPU[{device_id}]: {torch.cuda.get_device_name(device_id)}")
            except Exception:
                print("Warning: CUDA device introspection failed.")
                print(traceback.format_exc())

        print(f"Loading model: {MODEL_NAME} (path={LOAD_PATH})")
        tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH, trust_remote_code=True)

        # Some models do not define a pad token; using EOS avoids generation warnings/failures.
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            LOAD_PATH,
            torch_dtype=torch_dtype,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception:
        model_init_error = traceback.format_exc()
        print("Model initialization failed with traceback:")
        print(model_init_error)


def _last_line(text):
    if not text:
        return "Unknown error"
    lines = text.strip().splitlines()
    return lines[-1] if lines else "Unknown error"


def _get_model_device():
    if torch is None or model is None:
        return None
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def handler(job):
    if model_init_error:
        return {
            "error": "Model failed to initialize. Check worker logs for traceback.",
            "details": _last_line(model_init_error),
        }

    if tokenizer is None or model is None or torch is None:
        return {
            "error": "Worker is not ready.",
            "details": "Model/tokenizer dependencies are unavailable.",
        }

    job_input = job.get("input", {})

    if not isinstance(job_input, dict):
        return {"error": "input must be an object."}

    # Guard against sending WAN/video pipeline payloads to this text-generation endpoint.
    unsupported_fields = [
        key for key in ("model_id", "task", "image", "video", "num_inference_steps")
        if key in job_input
    ]
    if unsupported_fields:
        return {
            "error": "Unsupported payload for this endpoint.",
            "details": (
                "This worker is qwen-text only. Remove WAN/video fields "
                f"{unsupported_fields} and send 'prompt' or 'messages'."
            ),
        }

    messages = job_input.get("messages")
    prompt = job_input.get("prompt", "")

    if messages is not None and not isinstance(messages, list):
        return {"error": "messages must be a list when provided."}

    if prompt and not isinstance(prompt, str):
        return {"error": "prompt must be a string when provided."}

    if messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif prompt:
        text = prompt
    else:
        return {"error": "No prompt or messages provided."}

    try:
        max_new_tokens = int(job_input.get("max_new_tokens", 512))
    except (TypeError, ValueError):
        return {"error": "max_new_tokens must be an integer."}

    if max_new_tokens < 1:
        return {"error": "max_new_tokens must be at least 1."}

    try:
        temperature = float(job_input.get("temperature", 0.7))
    except (TypeError, ValueError):
        return {"error": "temperature must be a number."}

    if temperature < 0:
        return {"error": "temperature must be non-negative."}

    try:
        top_p = float(job_input.get("top_p", 0.9))
    except (TypeError, ValueError):
        return {"error": "top_p must be a number."}

    if not 0 < top_p <= 1:
        return {"error": "top_p must be in the range (0, 1]."}

    try:
        model_inputs = tokenizer([text], return_tensors="pt")
        model_device = _get_model_device()
        if model_device is not None:
            model_inputs = model_inputs.to(model_device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"response": response}
    except Exception:
        generation_error = traceback.format_exc()
        print("Generation failed with traceback:")
        print(generation_error)
        return {
            "error": "Generation failed. Check worker logs for traceback.",
            "details": _last_line(generation_error),
        }


initialize_model()
runpod.serverless.start({"handler": handler})
