import os
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DTYPE = os.environ.get("DTYPE", "float16")

dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
if DTYPE not in dtype_map:
    print(f"Warning: unrecognized DTYPE '{DTYPE}', falling back to float16.")
torch_dtype = dtype_map.get(DTYPE, torch.float16)

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully.")


def handler(job):
    job_input = job.get("input", {})

    messages = job_input.get("messages")
    prompt = job_input.get("prompt", "")

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

    max_new_tokens = int(job_input.get("max_new_tokens", 512))
    if max_new_tokens < 1:
        return {"error": "max_new_tokens must be at least 1."}

    temperature = float(job_input.get("temperature", 0.7))
    if temperature < 0:
        return {"error": "temperature must be non-negative."}

    top_p = float(job_input.get("top_p", 0.9))
    if not 0 < top_p <= 1:
        return {"error": "top_p must be in the range (0, 1]."}

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"response": response}


runpod.serverless.start({"handler": handler})
