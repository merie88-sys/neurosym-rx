import numpy as np
from config import EMBEDDING_DIM


class TemporalMemory:
    def __init__(self, embedding_dim=EMBEDDING_DIM, lambda_weight=0.4):
        self.embedding_dim = embedding_dim
        self.lambda_weight = lambda_weight
        self.short_term = {}
        self.medium_term = {}
        self.long_term = {}

    def _seeded_vector(self, seed, scale):
        rng = np.random.RandomState(seed)
        return rng.uniform(-scale, scale, size=self.embedding_dim)

    def encode_state(self, patient_state, t):
        pid = patient_state.get("patient_id", "default")
        base_seed = hash((pid, patient_state.get("age", 40),
                          patient_state.get("egfr", 90),
                          tuple(sorted(patient_state.get("medications", []))))) % (2**31)

        self.short_term[pid] = {
            "vector": self._seeded_vector(base_seed + t, 0.3),
            "timestamp": t,
            "medications": list(patient_state.get("medications", []))
        }
        if t >= 1:
            self.medium_term[pid] = {
                "vector": self._seeded_vector(base_seed + t // 2, 0.5),
                "timestamp": t,
                "conditions": list(patient_state.get("conditions", []))
            }
        if t >= 12:
            self.long_term[pid] = {
                "vector": self._seeded_vector(base_seed, 0.7),
                "timestamp": t,
                "history": patient_state.get("history", [])
            }

    def retrieve_context(self, patient_id, current_t):
        context = {"short": None, "medium": None, "long": None}
        if patient_id in self.short_term:
            context["short"] = self.short_term[patient_id]
        if patient_id in self.medium_term:
            context["medium"] = self.medium_term[patient_id]
        if patient_id in self.long_term:
            context["long"] = self.long_term[patient_id]
        return context

    def compute_relevance(self, current_state, historical_state, t_current, t_historical):
        delta_t = abs(t_current - t_historical)
        temporal_decay = np.exp(-delta_t / 12.0)
        curr_vec = self._state_to_vector(current_state)
        hist_vec = self._state_to_vector(historical_state)
        cos_sim = float(np.dot(curr_vec, hist_vec) /
                        (np.linalg.norm(curr_vec) * np.linalg.norm(hist_vec) + 1e-9))
        semantic = (cos_sim + 1.0) / 2.0
        return self.lambda_weight * semantic + (1 - self.lambda_weight) * temporal_decay

    def _state_to_vector(self, state):
        seed = hash((state.get("patient_id", ""),
                     state.get("age", 0),
                     state.get("egfr", 0),
                     tuple(sorted(state.get("medications", []))))) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.uniform(-1, 1, size=self.embedding_dim)
