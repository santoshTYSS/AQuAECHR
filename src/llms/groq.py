import os
from time import sleep
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_response(prompt: str, model: str, prompt_size: int = 5000) -> str:
    try:
        messages = [
            {"role": "user", "content": prompt},
        ]
        chat_completion = client.chat.completions.create(
            messages=messages, model=model, temperature=0
        )

        result = chat_completion.choices[0].message.content
        if "70b" in model:
            sleep(4)  # avoid rate limiting
        else:
            sleep(2)
        return result
    except Exception as e:
        print(e)
        # the prompt should be under 8192 tokens => we heuristically truncate it to 5k words and make it smaller each iteration till it fits
        if len(prompt.split()) >= prompt_size:
            prompt = " ".join(prompt.split()[:prompt_size])
        sleep(60)  # avoid rate limiting
        return get_response(prompt, model, prompt_size - 500)


class Llama8B:
    def infer_completion(self, prompt: str) -> str:
        return get_response(prompt, "llama3-8b-8192")


class Llama70B:
    def infer_completion(self, prompt: str) -> str:
        return get_response(prompt, "llama3-70b-8192")


class Mixtral8x7B:
    def infer_completion(self, prompt: str) -> str:
        return get_response(prompt, "mixtral-8x7b-32768")
