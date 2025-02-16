import os
import re
import dotenv
from transformers import pipeline, Pipeline
from huggingface_hub import login
from datasets import load_dataset
from google import genai


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
    prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a helpful assistant<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
    """
    results = []
    llm_client = genai.Client(api_key='GEMINI_API_KEY')

    for instruction_dict in instructions:
        eval_model_input = instruction_dict["instruction"]

        eval_model_output = pipe(prompt.format(
            question=eval_model_input,
        ))[0]['generated_text']

        test_model_prompt = JUDGE_PROMPT.format(
            question=eval_model_input,
            answer=eval_model_output,
        )

        response = llm_client.models.generate_content(
            model='gemini-2.0-flash', contents=test_model_prompt
        )
        llm_answer = response.text
    
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