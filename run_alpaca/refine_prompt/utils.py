import json
import logging
import os
import random

import pandas as pd
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Refiner:
    def __init__(
        self,
        hf_model_path,
        api_model_name,
        prompt_template_path,
        attack_prompt,
        output_json,
        base_url=None,
        api_key=None,
    ):
        self.model_path = hf_model_path
        self.api_model_name = api_model_name
        self.attack_prompt = attack_prompt
        self.output_json = output_json
        self.base_url = base_url
        self.api_key = api_key
        self.template_str = load_prompt_template(prompt_template_path)
        self.client = None
        self.model = None
        self.tokenizer = None

        if self.base_url and self.api_key:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        elif self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_path, torch_dtype=torch.float16, device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        else:
            raise ValueError("Provide either API settings or --hf_model_path.")

    def apply_prompt_template(self, vanilla_prompt):
        return self.template_str.format(prompt=vanilla_prompt)

    def api_generate(self, prompt, temperature, max_new_tokens=200):
        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    def llm_generate(self, prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": "You are a helpful assistant and strictly follow the instructions."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def load_rows(self):
        return pd.read_csv(self.attack_prompt).fillna("").to_dict("records")

    def build_output_item(self, row, response):
        return {
            "source": row.get("source", "tatsu-lab/alpaca"),
            "source_index": int(row["source_index"]) if str(row.get("source_index", "")).isdigit() else row.get("source_index", ""),
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
            "goal": row["goal"],
            "target": row.get("target", row.get("output", "")),
            "refined goal": response,
        }

    def run_refinement_hf(self, max_new_tokens=200):
        refined_list = []
        with torch.no_grad():
            for row in tqdm(self.load_rows(), desc="Refining Alpaca"):
                prompt = self.apply_prompt_template(row["goal"])
                response = self.llm_generate(prompt, max_new_tokens)
                logging.info(f"[Refined]: {response}")
                refined_list.append(self.build_output_item(row, response))
        self.save(refined_list)

    def run_refinement_api(self, max_new_tokens=200):
        refined_list = []
        for row in tqdm(self.load_rows(), desc="Refining Alpaca"):
            prompt = self.apply_prompt_template(row["goal"])
            response = None
            for i in range(5):
                temperature = 0.3 if i == 0 else random.uniform(0.2, 0.7)
                try:
                    response = self.api_generate(prompt, temperature, max_new_tokens)
                except Exception as exc:
                    logging.error(f"API call failed: {exc}")
                    response = None
                if response and "sorry" not in response.lower():
                    break
            if response is None:
                response = ""
            logging.info(f"[Refined]: {response}")
            refined_list.append(self.build_output_item(row, response))
        self.save(refined_list)

    def save(self, refined_list):
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(refined_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved refined Alpaca prompts to {self.output_json}")
