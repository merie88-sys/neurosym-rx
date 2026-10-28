import json
import numpy as np
from config import SYMBOLIC_WEIGHTS, CONTRAINDICATION_ESCALATION, RULES_PATH


class SymbolicLayer:
    def __init__(self, rules_path=RULES_PATH):
        with open(rules_path, "r") as f:
            data = json.load(f)
        self.rules = data["rules"]
        self._index = self._build_index()

    def _build_index(self):
        index = {}
        for rule in self.rules:
            key = tuple(sorted([rule["d1"].lower(), rule["d2"].lower()]))
            index.setdefault(key, []).append(rule)
        return index

    def _normalize_drug(self, name):
        return name.strip().lower().replace(" ", "_")

    def _check_contraindication(self, rule, patient_state):
        for condition in rule.get("contraindications", []):
            if condition == "renal_impairment" and patient_state.get("egfr", 90) < 60:
                return True
            if condition == "elderly" and patient_state.get("age", 40) >= 65:
                return True
            if condition in patient_state.get("conditions", []):
                return True
        return False

    def predict(self, drug_pair, patient_state=None):
        if patient_state is None:
            patient_state = {}
        d1 = self._normalize_drug(drug_pair[0])
        d2 = self._normalize_drug(drug_pair[1])
        key = tuple(sorted([d1, d2]))
        activated_rules = self._index.get(key, [])

        if not activated_rules:
            return {"R_symbolic": 0.0, "activated_rules": [], "max_severity": None}

        max_sev_val = 0.0
        max_severity_label = None
        any_contra = False
        for rule in activated_rules:
            sev = SYMBOLIC_WEIGHTS.get(rule["severity"], 0.2)
            if sev > max_sev_val:
                max_sev_val = sev
                max_severity_label = rule["severity"]
            if self._check_contraindication(rule, patient_state):
                any_contra = True

        r_symbolic = min(1.0, max_sev_val + (CONTRAINDICATION_ESCALATION if any_contra else 0.0))
        return {
            "R_symbolic": float(r_symbolic),
            "activated_rules": [r["id"] for r in activated_rules],
            "max_severity": max_severity_label,
            "contraindication_active": any_contra
        }
