from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

prefix = 'def remove_non_ascii(s: str) -> str:\n    """ '
suffix = "\n    return result\n"

prompt = f"<PRE> {prefix}<SUF>{suffix} <MID>"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=200,
    do_sample=False,
)
output = output[0].to("cpu")
print(tokenizer.decode(output))