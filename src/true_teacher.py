from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch

from src.constants import CACHE_DIR
from src.utils.chunk import chunk as chunk_tokens


class TRUETeacher:
    def __init__(self, device, cache_dir=None, max_length=2048):
        model_path = "google/t5_11b_trueteacher_and_anli"
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, cache_dir=cache_dir, torch_dtype=torch.float16
        )
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length

    def __get_value(self, premise: str, hypothesis: str) -> float:
        # print(f"Premise: {premise[0:60]}")
        # print(f"Hypothesis: {hypothesis[0:60]}")
        # print(f"Tokens premise: {len(self.__get_tokens(premise))}")
        # print(f"Tokens hypothesis: {len(self.__get_tokens(hypothesis))}")

        input_ids = self.tokenizer(
            f"premise: {premise} hypothesis: {hypothesis}",
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids.to(self.device)
        with torch.no_grad():
            decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]]).to(
                self.device
            )
            outputs = self.model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            probs = torch.softmax(logits[0], dim=-1)
            one_token_id = self.tokenizer("1").input_ids[0]
            entailment_prob = probs[0, one_token_id].item()
            return entailment_prob

    def get_value(self, premise: str, hypothesis: str):
        nr_tokens_hypothesis = len(self.__get_tokens(hypothesis))
        if nr_tokens_hypothesis > 512:
            # we chunk the hypothesis if it is too long
            chunks = self.__get_chunks(hypothesis, max_length=500, stride=250)
            results: list[float] = [self.get_value(premise, chunk) for chunk in chunks]
            result = min(results)
            return result

        nr_tokens_premise = len(self.__get_tokens(premise))

        if nr_tokens_premise + nr_tokens_hypothesis < self.max_length - 10:
            return self.__get_value(premise, hypothesis)

        chunk_size = self.max_length - nr_tokens_hypothesis - 10
        chunks = self.__get_chunks(
            premise, max_length=chunk_size, stride=int(chunk_size * 0.5)
        )
        results = [self.__get_value(chunk, hypothesis) for chunk in chunks]
        result = max(results)
        return result

    def __get_tokens(self, text):
        return self.tokenizer(text).input_ids

    def __get_chunks(self, text, max_length, stride):
        tokens = self.__get_tokens(text)

        if len(tokens) <= max_length:
            return [text]

        token_chunks = chunk_tokens(tokens, max_length, stride)
        chunks = [self.tokenizer.decode(chunk) for chunk in token_chunks]
        return chunks
