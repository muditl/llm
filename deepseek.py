from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datetime import datetime

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16) # CUDA


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct", trust_remote_code=True, quantization_config=bnb_config)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-7b-instruct", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config)

messages=[
    {'role': 'user', 'content': """In the following code finish the function to unsubscribe from all subscriptions:
from flask import Flask, request, redirect, Blueprint
import MySQLdb

bp = Blueprint('email_handling',__name__)
def get_db_connection():
    db = MySQLdb.connect(host="localhost",
                         user="python",
                         passwd="python",
                         db="db")
    return db
    
def create_subscriptions_table(db):
    cursor = db.cursor()
    #delete table if it exists
    cursor.execute("DROP TABLE IF EXISTS subscriptions")
    
    cursor.execute(\"""CREATE TABLE subscriptions (
                            email varchar(255) not null,
                            channel varchar(255) not null,
                            PRIMARY KEY (email, channel)
                            )\""")
    db.commit()
    
#URL handler to remove a user from all subscriptions
@bp.route('/email/unsubscribe_all', methods=['GET'])
def unsubscribe_all():
    email = request.args.get('email')
    db = get_db_connection()
    unsubscribe_email_from_all_subscriptions(db, email)
    db.close()
    return redirect('/')
    
#Database function to delete a user from all subscriptions        
def unsubscribe_email_from_all_subscriptions(db, email): """}]

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
print("took " + str(end-start) + " to generate response, or "+str((end-start).total_seconds())+" seconds.")

print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))