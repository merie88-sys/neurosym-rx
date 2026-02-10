#!/usr/bin/env python3
"""Cascaded fusion mechanism with explicit conflict resolution"""

class FusionModule:
    def integrate(self, r_symbolic: float, r_neural: float, 
                  c_rule: float, c_context: float, c_data: float) -> float:
        """
        Cascaded fusion with conflict resolution
        
        Args:
            r_symbolic: Symbolic risk score [0.0, 1.0]
            r_neural: Neural risk score [0.0, 1.0]
            c_rule: Rule confidence [0.0, 1.0]
            c_context: Contextual relevance [0.0, 1.0]
            c_data: Data reliability [0.0, 1.0]
        
        Returns:
            Integrated risk score [0.0, 1.0]
        """
        # Adaptive weighting (Equation 4)
        alpha = 0.4 * c_rule + 0.35 * c_context + 0.25 * c_data
        alpha = max(0.0, min(1.0, alpha))
        
        # Conflict detection
        delta = abs(r_symbolic - r_neural)
        
        if delta > 0.5 and c_rule < 0.6:
            # Safety fallback to symbolic reasoning
            return r_symbolic
        elif delta > 0.3:
            # Conflict resolution
            if c_rule > c_context + 0.2:
                return r_symbolic
            else:
                return r_neural
        
        # Complementary fusion (Equation 3)
        return alpha * r_symbolic + (1 - alpha) * r_neural
