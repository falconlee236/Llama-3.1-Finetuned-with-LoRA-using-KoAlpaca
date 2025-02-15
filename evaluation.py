from transformers import pipeline
import dotenv
import os
import huggingface_hub
from datasets import load_dataset

dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")

huggingface_hub.login(
    token=os.getenv("HUGGINGFACE_API_KEY")
)

pipe = pipeline(
    task="text-generation",
    model=REPO_NAME,
    tokenizer=REPO_NAME,
    max_length=256
)
 
def generate_and_stop(instructions: list) -> list:
    prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
    """
    results = []
    for instruction in instructions:
        results.append(pipe(f"{prompt.format(instruction)}")[0]['generated_text'])
    return results
 


ds = load_dataset("HAERAE-HUB/KUDGE")


JUDGE_PROMPT = """
사용자 질문(user_question)과 시스템 답변(system_answer) 쌍이 주어질 것입니다. 귀하의 임무는 시스템 답변이 사용자 질문에서 표현된 사용자의 관심사를 얼마나 잘 답변했는지 '종합 평가'를 제공하는 것입니다. 

0부터 10까지의 척도로 float 형식의 점수를 제시해 주십시오. 
- 0점은 시스템 답변이 전혀 도움이 되지 않음을 의미합니다
- 10점은 답변이 질문을 완벽하고 유용하게 다루었음을 의미합니다

다음 형식으로 피드백을 제공해 주십시오:

피드백:::
종합 평가: (0에서 10 사이의 float 형식 점수)

이제 질문과 답변입니다.

질문: {question}
답변: {answer}

피드백:::
종합 평가:
"""

results = generate_and_stop([x for x in ds["instruction"]])
for x in results:
    print(x)