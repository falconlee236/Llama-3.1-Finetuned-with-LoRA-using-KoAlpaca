import os
import re
import dotenv
from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, InferenceClient
from datasets import load_dataset
import torch


# https://github.com/HeegyuKim/open-korean-instructions?tab=readme-ov-file 
# https://huggingface.co/learn/cookbook/llm_judge
dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")
JUDGE_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return 0


def generate_and_stop(pipe:Pipeline, instructions: list) -> list:
    prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
    """
    results = []
    llm_client = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        load_in_4bit=True,
        device_map="auto"
    )
    # llm_client = InferenceClient(
    #     model=JUDGE_MODEL_ID,
    #     timeout=120,
    # )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)

    for instruction_dict in instructions:
        eval_model_input = instruction_dict["instruction"]

        eval_model_output = pipe(prompt.format(
            question=eval_model_input,
        ))[0]['generated_text']

        test_model_prompt = JUDGE_PROMPT.format(
            question=eval_model_input,
            answer=eval_model_output,
        )
        input_ids = tokenizer.apply_chat_template(
            test_model_prompt,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = llm_client.generate(input_ids, max_new_tokens=20)
        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        results.append({
            **instruction_dict,
            "judge_score": extract_judge_score(answer=llm_answer),
        })
    return results
 

if __name__ == "__main__":
    login(token=os.getenv("HUGGINGFACE_API_KEY"))    

    pipeline = pipeline(
        task="text-generation",
        model=REPO_NAME,
        tokenizer=REPO_NAME,
        max_length=256,
        device_map="auto",
        model_kwargs={"load_in_8bit": True},
    )

    hub_datasets = load_dataset("HAERAE-HUB/KUDGE", "Human Annotations")
    human_datasets = [
    dict(uuid=item["uuid"],
        instruction=item["instruction"],
        response=item["response"],
    ) for item in hub_datasets["test"]]

    ans = generate_and_stop(
        pipe=pipeline,
        instructions=human_datasets,
    )

    print(len(ans))
    for x in ans:
        print(x)