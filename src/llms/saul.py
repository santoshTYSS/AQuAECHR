from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constants import CACHE_DIR


class Saul:
    def __init__(self, device: str) -> None:
        self.device = device
        model_name = "Equall/Saul-Instruct-v1"

        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def infer_completion(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            model_inputs, max_new_tokens=1024, do_sample=True
        )
        decoded = self.tokenizer.batch_decode(generated_ids)
        response = decoded[0]
        response_part = response.split("[/INST]", 1)[-1]
        response_part = response_part.split("</s>", 1)[0]
        response_part = response_part.strip()
        return response_part
