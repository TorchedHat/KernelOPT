"""TorchServe handler for Qwen3-30B-A3B (BF16) — used by planner + summarizer."""

from __future__ import annotations

import json
import logging

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"


class Qwen3Handler(BaseHandler):

    def initialize(self, context):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.manifest = context.manifest
        properties = context.system_properties
        gpu_id = properties.get("gpu_id", 0)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model %s on %s (BF16)", MODEL_NAME, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def preprocess(self, requests):
        """Parse OpenAI-format chat messages from each request in the batch."""
        batch = []
        for req in requests:
            body = req.get("body") or req.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body)

            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 4096)
            temperature = body.get("temperature", 0.2)

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch.append({
                "text": text,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        return batch

    def inference(self, batch):
        """Tokenize and generate for the whole batch at once."""
        texts = [item["text"] for item in batch]
        max_tokens = max(item["max_tokens"] for item in batch)
        temperature = batch[0]["temperature"]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids[i].shape[0]
            generated = self.tokenizer.decode(
                output[input_len:], skip_special_tokens=True,
            )
            results.append(generated)
        return results

    def postprocess(self, results):
        """Return list of OpenAI-compatible response dicts."""
        return [json.dumps({"generated_text": r}) for r in results]
