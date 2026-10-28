import json
import requests
import numpy as np
from config import LLM_CONFIG


class NeuralLayer:
    def __init__(self, config=None):
        self.config = config or LLM_CONFIG
        self._use_mock = not self.config.get("api_key")

    def _build_prompt(self, drug_pair, patient_context, retrieved_evidence):
        return (
            "You are a clinical pharmacologist. Estimate the patient-specific interaction risk "
            "between two drugs on a scale from 0.0 (no risk) to 1.0 (critical risk). "
            "Consider the patient context and retrieved evidence.\n\n"
            f"Drug pair: {drug_pair[0]} + {drug_pair[1]}\n"
            f"Patient context: {json.dumps(patient_context)}\n"
            f"Retrieved evidence: {json.dumps(retrieved_evidence)}\n\n"
            "Respond with ONLY a JSON object: {\"risk\": <float>, \"confidence\": <float 0-1>}"
        )

    def _call_api(self, prompt):
        if self.config["provider"] == "openai":
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"]
            }
            url = self.config.get("base_url") or "https://api.openai.com/v1/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unsupported provider: {self.config['provider']}")

        try:
            parsed = json.loads(content)
            risk = float(np.clip(parsed.get("risk", 0.5), 0.0, 1.0))
            confidence = float(np.clip(parsed.get("confidence", 0.5), 0.0, 1.0))
        except (json.JSONDecodeError, ValueError):
            risk, confidence = 0.5, 0.3
        return risk, confidence

    def _deterministic_fallback(self, drug_pair, patient_context, seed):
        rng = np.random.RandomState(seed)
        base = rng.uniform(0.3, 0.7)
        age = patient_context.get("age", 40)
        egfr = patient_context.get("egfr", 90)
        n_meds = patient_context.get("n_medications", 1)
        adjustment = 0.0
        if age >= 65:
            adjustment += 0.08
        if egfr < 60:
            adjustment += 0.12
        if n_meds >= 5:
            adjustment += 0.05
        risk = float(np.clip(base + adjustment, 0.0, 1.0))
        confidence = float(np.clip(0.6 + rng.uniform(-0.1, 0.15), 0.3, 0.95))
        return risk, confidence

    def predict(self, drug_pair, patient_context, retrieved_evidence=None, seed=None):
        if retrieved_evidence is None:
            retrieved_evidence = {}
        if self._use_mock:
            seed_value = seed if seed is not None else hash((drug_pair[0], drug_pair[1])) % (2**31)
            return self._deterministic_fallback(drug_pair, patient_context, seed_value)
        prompt = self._build_prompt(drug_pair, patient_context, retrieved_evidence)
        return self._call_api(prompt)
