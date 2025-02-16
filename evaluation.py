import os
import re
import dotenv
import pandas as pd
from transformers import pipeline, Pipeline, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset, Dataset
from google import genai
from tqdm import tqdm
import torch


# https://github.com/HeegyuKim/open-korean-instructions?tab=readme-ov-file 
# https://huggingface.co/learn/cookbook/llm_judge
dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")
JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: 
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
    prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
    """
    split_str = "<|start_header_id|>assistant<|end_header_id|>"
    llm_client = genai.Client(api_key=GEMINI_API_KEY)
    dataset = Dataset.from_list(instructions)

    def process_example(example):
        eval_model_output = pipe(
            prompt_template.format(question=example["instruction"])
        )[0]['generated_text'].split(split_str)[1].strip()
        
        test_model_prompt = JUDGE_PROMPT.format(
            question=example["instruction"],
            answer=eval_model_output,
        )

        response = llm_client.models.generate_content(
            model='gemini-2.0-flash', contents=test_model_prompt
        )
        llm_answer = response.text

        # print(f"instruction_output = {example['instruction']}")
        # print(f"eval_model_output = {eval_model_output}")
        # print(f"llm_answer = {llm_answer}")

        return {
            **example,
            "judge_score": extract_judge_score(answer=llm_answer),
        }

    results = list(tqdm(dataset.map(process_example), total=len(dataset))) 
    return results

def calculate_average_judge_score(results):
    """
    judge_score의 평균을 계산하는 함수
    results: generate_and_stop()의 결과 리스트
    """
    scores = [item["judge_score"] for item in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average Judge Score: {avg_score:.4f}")
    return avg_score

def save_results_to_csv(results, filename="judge_scores.csv"):
    """
    결과 리스트를 CSV 파일로 저장하는 함수
    results: generate_and_stop()의 결과 리스트
    filename: 저장할 CSV 파일명 (기본값: judge_scores.csv)
    """
    df = pd.DataFrame(results)  # 리스트를 DataFrame으로 변환
    df.to_csv(filename, index=False, encoding="utf-8")  # CSV로 저장
    print(f"✅ CSV 파일 저장 완료: {filename}")



if __name__ == "__main__":
    login(token=os.getenv("HUGGINGFACE_API_KEY"))    

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    pipeline = pipeline(
        task="text-generation",
        model=REPO_NAME,
        tokenizer=REPO_NAME,
        max_length=512,
        device_map="auto",
        truncation=True,
        model_kwargs={"quantization_config": quantization_config}
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
    calculate_average_judge_score(ans)
    save_results_to_csv(ans)