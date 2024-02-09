from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16)

messages=[
    {'role': 'user', 'content': """ Finish the following C code to multiply 2 doubles and return them in string form.
    char* multiply_doubles_str (double d1, double d2){
        }"""}]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

start = datetime.now()

# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(
    inputs, 
    max_new_tokens=512, 
    do_sample=True,
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=1, 
    eos_token_id=tokenizer.eos_token_id)

end = datetime.now()
print("took " + str(end-start) + " to generate response, or "+str((end-start).total_seconds)+" seconds.")

print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))