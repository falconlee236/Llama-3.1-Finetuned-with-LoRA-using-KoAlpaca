from transformers import pipeline
import dotenv
import os
import huggingface_hub

dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")

huggingface_hub.login(token=os.getenv("HUGGINGFACE_API_KEY"))

pipeline = pipeline(
    task="text-generation",
    model=REPO_NAME,
    tokenizer=REPO_NAME,
    max_length=256,
    device_map="auto",
    truncation=True,
    model_kwargs={"load_in_8bit": True},
)
 
def generate_and_stop(prompt):
    result = pipeline(f"{prompt}")[0]['generated_text']
    return result
 
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
 
양파는 어떤 식물 부위인가요? 그리고 고구마는 뿌리인가요?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
"""
 
print(generate_and_stop(prompt))