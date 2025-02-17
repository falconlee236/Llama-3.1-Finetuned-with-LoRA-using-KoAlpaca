import os
import re
import dotenv
import pandas as pd
from transformers import pipeline, Pipeline, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset, Dataset
from groq import Groq
from tqdm import tqdm


# https://github.com/HeegyuKim/open-korean-instructions?tab=readme-ov-file 
# https://huggingface.co/learn/cookbook/llm_judge
dotenv.load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPO_NAME = os.getenv("HUGGINGFACE_REPO")
DATA_SIZE = 1000
JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

If you need, Provide your feedback as follows:

Feedback (optional):::
Total rating (nessesary): (your rating, as a float between 0 and 10)

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
    
    You are a helpful assistant and You always provide summarized answer<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
    """
    split_str = "<|start_header_id|>assistant<|end_header_id|>"
    llm_client = Groq(api_key=GROQ_API_KEY)
    dataset = Dataset.from_list(instructions)

    def process_batch_example(examples: dict[str, list]) -> dict[str, list]:
        """
        examples = {
            "uuid": ["1", "2"],
            "instruction": ["A", "B"],
            "response": ["X", "Y"]
        }

        """
        judge_scores = []
        feedbacks = []

        for instruction in examples["instruction"]:
            eval_model_output = pipe(
                prompt_template.format(question=instruction)
            )[0]['generated_text'].split(split_str)[1].strip()
            
            test_model_prompt = JUDGE_PROMPT.format(
                question=instruction,
                answer=eval_model_output,
            )

            response = llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "you are a helpful llm judge assistant.",
                    },
                    {
                        "role": "user",
                        "content": test_model_prompt,
                    }
                ],
                model="gemma2-9b-it", # https://console.groq.com/settings/limits
                temperature=0.5,
                max_completion_tokens=128,
                top_p=1,
                stop=None,
            )
            llm_answer = response.choices[0].message.content
            judge_score = extract_judge_score(answer=llm_answer)

            judge_scores.append(judge_score)
            feedbacks.append(llm_answer)

        return {
            "uuid": examples["uuid"],
            "instruction": examples["instruction"],
            "response": examples["response"],
            "feedback": feedbacks,
            "judge_score": judge_scores,
        }

    results = list(tqdm(dataset.map(
        process_batch_example,
        batched=True,
        batch_size=8,
    ), total=len(dataset))) 

    final_results = {
        "uuid": [],
        "instruction": [],
        "response": [],
        "feedback": [],
        "judge_score": [],
    }

    for batch in results:
        final_results["uuid"].append(batch["uuid"])
        final_results["instruction"].append(batch["instruction"])
        final_results["response"].append(batch["response"])
        final_results["feedback"].append(batch["feedback"])
        final_results["judge_score"].append(batch["judge_score"])

    return final_results


def calculate_average_judge_score(results):
    """
    judge_score의 평균을 계산하는 함수
    results: 배치 결과를 합친 최종 딕셔너리
    """
    scores = results["judge_score"]  # judge_score 리스트 사용
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average Judge Score: {avg_score:.4f}")
    return avg_score


def save_results_to_csv(results, filename="judge_scores.csv"):
    """
    결과 딕셔너리를 CSV 파일로 저장하는 함수
    results: 배치 결과를 합친 최종 딕셔너리
    filename: 저장할 CSV 파일명 (기본값: judge_scores.csv)
    """
    df = pd.DataFrame.from_dict(results)  # 딕셔너리를 DataFrame으로 변환
    df.to_csv(filename, index=False, encoding="utf-8")  # CSV로 저장
    print(f"✅ CSV 파일 저장 완료: {filename}")


if __name__ == "__main__":
    login(token=os.getenv("HUGGINGFACE_API_KEY"))    

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    pipeline = pipeline(
        task="text-generation",
        model=REPO_NAME,
        tokenizer=REPO_NAME,
        max_length=256,
        device_map="auto",
        truncation=True,
        model_kwargs={"quantization_config": quantization_config}
    )

    hub_datasets = load_dataset("HAERAE-HUB/KUDGE", "Human Annotations")
    human_datasets = [
    dict(uuid=item["uuid"],
        instruction=item["instruction"],
        response=item["response"],
    ) for item in hub_datasets["test"].select(range(DATA_SIZE))]

    ans = generate_and_stop(
        pipe=pipeline,
        instructions=human_datasets,
    )

    print(len(ans))
    calculate_average_judge_score(ans)
    save_results_to_csv(ans)