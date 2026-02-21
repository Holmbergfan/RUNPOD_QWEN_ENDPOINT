"""RunPod Qwen Endpoint Handler."""

import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
)


def handler(job):
    """Handle incoming inference requests."""
    job_input = job["input"]
    prompt = job_input.get("prompt", "")
    max_tokens = job_input.get("max_tokens", 512)
    temperature = job_input.get("temperature", 0.7)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return {"output": response}


runpod.serverless.start({"handler": handler})
