import numpy as np
from config import THETA_CONFLICT, THETA_FALLBACK, CONFIDENCE_MARGIN, ALPHA_WEIGHTS


class CascadedFusion:
    def __init__(self):
        self.stats = {"complementary": 0, "conflict_resolution": 0, "symbolic_fallback": 0}

    def compute_alpha(self, c_rule, c_context, c_data):
        return (ALPHA_WEIGHTS["rule"] * c_rule +
                ALPHA_WEIGHTS["context"] * c_context +
                ALPHA_WEIGHTS["data"] * c_data)

    def fuse(self, r_symbolic, r_neural, c_rule, c_context, c_data):
        delta = abs(r_symbolic - r_neural)
        alpha = self.compute_alpha(c_rule, c_context, c_data)

        if delta < THETA_CONFLICT:
            r_final = alpha * r_symbolic + (1 - alpha) * r_neural
            strategy = "complementary"
            alert_elevated = False
        elif delta <= THETA_FALLBACK:
            if c_rule > c_context + CONFIDENCE_MARGIN:
                r_final = r_symbolic
            else:
                r_final = r_neural
            strategy = "conflict_resolution"
            alert_elevated = False
        else:
            if c_rule < 0.6:
                r_final = r_symbolic
                alert_elevated = True
            else:
                r_final = r_symbolic if c_rule > c_context else r_neural
                alert_elevated = False
            strategy = "symbolic_fallback"

        self.stats[strategy] += 1
        return {
            "R_final": float(np.clip(r_final, 0.0, 1.0)),
            "strategy": strategy,
            "delta": float(delta),
            "alpha": float(alpha),
            "alert_elevated": alert_elevated
        }

    def reset_stats(self):
        self.stats = {"complementary": 0, "conflict_resolution": 0, "symbolic_fallback": 0}
