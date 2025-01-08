# https://jjaegii.tistory.com/35
# https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy
import dotenv
import os
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import huggingface_hub
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

def format_example(row):
    return {
        'text': f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
        You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
 
        {row['instruction']}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
 `
        {row['output']}<|eot_id|>"""
    }

dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "Llama-3.1-Finetuned-with-LoRA-using-KoAlpaca"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

wandb.login(key=WANDB_API_KEY)

huggingface_hub.login(token=API_KEY)
# https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a
print(load_dataset("beomi/KoAlpaca-v1.1a"))
train_ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train[:5%]") #10%했더니 OOM 나옴
train_dataset = train_ds.map(format_example)

# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
base_model = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto"
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token # use eos token in sequence padding
tokenizer.padding_side = "right" # add padding right side

# Setting LoRA
peft_params = LoraConfig(
    lora_alpha=16, # LoRA의 스케일링 계수 설정
    lora_dropout=0.1, # 드롭아웃을 통해 과적합 방지
    r=8, # LoRA 어댑터 행렬의 Rank 설정
    bias="none", # 편향 사용 여부 설정
    task_type="CAUSAL_LM", # 작업 유형 설정 (Causal LM)
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj'] # 적용 모듈 설정
)

# 모델을 8bit 학습을 위한 상태로 준비. 메모리를 절약하면서도 모델의 성능을 유지할 수 있음
model = prepare_model_for_kbit_training(model, 8)

# PEFT 어댑터 설정을 모델에 적용
model = get_peft_model(model, peft_params)
 
# 학습 파라미터 설정
training_params = TrainingArguments(
    report_to="wandb",                   # enables logging to wandb
    output_dir="./results",              # 결과 저장 경로
    num_train_epochs=1,                 # 학습 에폭 수, 2만개 전부 해서 기존 50번 하면 172시간 걸림 A40 기준
    per_device_train_batch_size=8,       # 배치 사이즈
    learning_rate=2e-4,                  # 학습률 설정
    save_steps=1000,                     # 저장 빈도
    logging_steps=50,                    # 로그 출력 빈도
    fp16=True,                            # 16-bit 부동 소수점 사용 (메모리 절약)
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine",
)
 
# SFTTrainer를 사용해 학습 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_params,
    #dataset_text_field="text",
    #max_seq_length=None,  # 시퀀스 길이 제한
    tokenizer=tokenizer,
    args=training_params,
)
 
trainer.train()


model.push_to_hub(
    REPO_NAME,
    use_temp_dir=True,
    use_auth_token=API_KEY
)
tokenizer.push_to_hub(
    REPO_NAME,
    use_temp_dir=True,
    use_auth_token=API_KEY
)
