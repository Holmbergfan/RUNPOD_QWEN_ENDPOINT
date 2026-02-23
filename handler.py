import os
import shutil
import traceback

import runpod
from huggingface_hub import snapshot_download

WORKER_KIND = "qwen-text"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DTYPE = os.environ.get("DTYPE", "float16").lower()
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0").lower() in ("1", "true", "yes", "on")

# Conservative storage requirements with headroom.
MODEL_REQUIRED_GB = {
    "Qwen/Qwen2.5-7B-Instruct": 18,
    "Qwen/Qwen2.5-VL-7B-Instruct": 20,
}

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
tokenizer = None
model = None
model_init_error = None
model_init_hint = None


def _local_model_dir(model_name):
    return os.path.join("/runpod-volume/models", model_name.split("/")[-1])


def _bytes_to_gb(value):
    return value / (1024 ** 3)


def _model_required_gb(model_name):
    return MODEL_REQUIRED_GB.get(model_name, 15)


def _model_is_ready(path):
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    known_weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any(os.path.isfile(os.path.join(path, f)) for f in known_weight_files):
        return True
    try:
        return any(
            name.startswith("model-") and name.endswith(".safetensors")
            for name in os.listdir(path)
        )
    except Exception:
        return False


def _resolve_load_path():
    local_candidate = _local_model_dir(MODEL_NAME)
    if _model_is_ready(local_candidate):
        return local_candidate, True
    return MODEL_NAME, False


def _download_preflight(required_gb_hint, path_for_usage):
    try:
        required_gb = float(required_gb_hint) if required_gb_hint is not None else None
    except (TypeError, ValueError):
        required_gb = None
    if required_gb is None:
        required_gb = _model_required_gb(MODEL_NAME)

    check_path = os.path.abspath(path_for_usage or HF_HOME or "/runpod-volume")
    if os.path.isfile(check_path):
        check_path = os.path.dirname(check_path)
    os.makedirs(check_path, exist_ok=True)

    usage = shutil.disk_usage(check_path)
    free_gb = _bytes_to_gb(usage.free)
    return {
        "ok": free_gb >= required_gb,
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "path": check_path,
    }


def _reset_model_state():
    global tokenizer
    global model
    global model_init_error
    global model_init_hint

    if model is not None:
        try:
            del model
        except Exception:
            pass
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    tokenizer = None
    model = None
    model_init_error = None
    model_init_hint = None


def _handle_download_models(job_input):
    model_config = job_input.get("model_config", {})
    if not isinstance(model_config, dict):
        return {"error": "download_models requires model_config to be an object."}

    hf_repo = model_config.get("hf_repo")
    output_dir = model_config.get("output_dir")
    required_gb = model_config.get("required_gb", _model_required_gb(hf_repo or MODEL_NAME))

    if not hf_repo or not output_dir:
        return {"error": "download_models requires model_config with hf_repo and output_dir"}

    if os.path.isdir(output_dir) and _model_is_ready(output_dir):
        return {"status": "already_downloaded", "hf_repo": hf_repo, "output_dir": output_dir}

    preflight = _download_preflight(required_gb, output_dir)
    if not preflight["ok"]:
        return {
            "error": (
                "Insufficient disk space for model download. "
                f"Need >= {preflight['required_gb']} GB free on {preflight['path']}, "
                f"have {preflight['free_gb']} GB."
            ),
            "hint": (
                "Attach/expand a larger network volume, choose a smaller model, or clear "
                "old artifacts under /runpod-volume/models and /runpod-volume/.cache/huggingface."
            ),
        }

    try:
        print(f"[download] {hf_repo} → {output_dir}")
        snapshot_download(
            repo_id=hf_repo,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[download] done: {output_dir}")
    except Exception as exc:
        err = str(exc)
        if "No space left on device" in err or "os error 28" in err.lower():
            post = _download_preflight(required_gb, output_dir)
            return {
                "error": f"Download failed: {err}",
                "hint": (
                    "Disk is full during download. Free space on /runpod-volume, then retry. "
                    f"Current free space: {post.get('free_gb')} GB."
                ),
            }
        return {"error": f"Download failed: {err}"}

    result = {"status": "downloaded", "hf_repo": hf_repo, "output_dir": output_dir}

    # If this worker's configured model was just downloaded, try to load it now
    # so subsequent inference requests do not require a cold restart.
    if hf_repo == MODEL_NAME and os.path.abspath(output_dir) == os.path.abspath(_local_model_dir(MODEL_NAME)):
        _reset_model_state()
        initialize_model()
        result["model_ready"] = model_init_error is None
        if model_init_error is not None:
            result["warning"] = "Model downloaded but initialization still failed."
            result["details"] = _last_line(model_init_error)
            if model_init_hint:
                result["hint"] = model_init_hint

    return result


def initialize_model():
    global torch
    global AutoModelForCausalLM
    global AutoTokenizer
    global tokenizer
    global model
    global model_init_error
    global model_init_hint

    print(f"Worker booting. kind={WORKER_KIND}, MODEL_NAME={MODEL_NAME}, DTYPE={DTYPE}, DEVICE_MAP={DEVICE_MAP}")
    print(f"HF_HOME={HF_HOME}, LOCAL_FILES_ONLY={LOCAL_FILES_ONLY}")

    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        torch = _torch
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
    except Exception:
        model_init_error = traceback.format_exc()
        model_init_hint = "Failed importing dependencies. Check image build and requirements."
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

        load_path, is_local = _resolve_load_path()
        local_files_only = LOCAL_FILES_ONLY or is_local
        required_gb = _model_required_gb(MODEL_NAME)
        if not is_local:
            preflight = _download_preflight(required_gb, HF_HOME)
            if not preflight["ok"]:
                model_init_error = (
                    "Insufficient disk space before model initialization. "
                    f"Need >= {preflight['required_gb']} GB free on {preflight['path']}, "
                    f"have {preflight['free_gb']} GB."
                )
                model_init_hint = (
                    "Attach/expand a larger network volume, or pre-download the model to "
                    f"{_local_model_dir(MODEL_NAME)} using a setup/download job."
                )
                print(model_init_error)
                return

        print(f"Loading model: {MODEL_NAME} (path={load_path}, local_files_only={local_files_only})")
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # Some models do not define a pad token; using EOS avoids generation warnings/failures.
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch_dtype,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model.eval()
        model_init_hint = None
        print("Model loaded successfully.")
    except Exception:
        model_init_error = traceback.format_exc()
        lower = model_init_error.lower()
        if "no space left on device" in lower or "os error 28" in lower:
            preflight = _download_preflight(_model_required_gb(MODEL_NAME), HF_HOME)
            model_init_hint = (
                "Disk is full while loading/downloading model weights. "
                f"Free space on {preflight['path']} (currently {preflight['free_gb']} GB) "
                f"or use a smaller model. Needed approx {preflight['required_gb']} GB."
            )
        elif (
            "model is not cached locally" in lower
            and "metadata from the hub" in lower
        ):
            model_init_hint = (
                "Model is not available locally and Hub fetch failed. "
                f"Pre-download to {_local_model_dir(MODEL_NAME)} or ensure outbound access "
                "to huggingface.co from the worker."
            )
        else:
            model_init_hint = None
        print("Model initialization failed with traceback:")
        print(model_init_error)
        if model_init_hint:
            print(f"Hint: {model_init_hint}")


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
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        return {"error": "input must be an object."}

    # Setup mode: allow downloads even if startup model init failed.
    if job_input.get("download_models"):
        return _handle_download_models(job_input)

    if model_init_error:
        response = {
            "error": "Model failed to initialize. Check worker logs for traceback.",
            "details": _last_line(model_init_error),
        }
        if model_init_hint:
            response["hint"] = model_init_hint
        return response

    if tokenizer is None or model is None or torch is None:
        return {
            "error": "Worker is not ready.",
            "details": "Model/tokenizer dependencies are unavailable.",
        }

    # Guard against WAN/video payloads accidentally sent to this text endpoint.
    # Only block fields that are purely WAN-specific (video, task+model_id combo).
    # 'image', 'tool', 'steps' etc. are NOT blocked — AI tool requests use them.
    wan_fields = [key for key in ("video",) if key in job_input]
    if not wan_fields and "task" in job_input and "model_id" in job_input:
        wan_fields = ["task", "model_id"]
    if wan_fields:
        return {
            "error": "Unsupported payload for this endpoint.",
            "details": (
                "This worker is qwen-text only. WAN/video fields "
                f"{wan_fields} are not supported here."
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
