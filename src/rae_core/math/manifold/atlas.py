"""
RAE Theory Atlas - The Registry of Mathematical Success.
Loads and manages hundreds of historical strategies.
"""

from typing import Dict, List, Any
from .modular import ModularManifoldArm

class TheoryAtlas:
    """
    Central registry for all RAE mathematical genomes.
    """
    def __init__(self):
        # Initializing with core historical 'genotypes' found in Git Audit
        self.genomes = {
            "classic_ib": {"name": "Information Bottleneck 1.0", "sharpening": 1.0, "alpha": 0.3, "beta": 0.5, "gamma": 0.2},
            "hyper_resolution_37": {"name": "Hyper-Resolution 37.0", "sharpening": 3.0, "alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "fluid_math_100": {"name": "Fluid Manifold 100.0", "sharpening": 5.0, "alpha": 0.4, "beta": 0.4, "gamma": 0.2},
            "neural_heavy_22": {"name": "Neural Scalpel 22.1", "sharpening": 10.0, "alpha": 0.1, "beta": 0.1, "gamma": 0.8},
            "industrial_robust": {"name": "Industrial Resilience 6.5", "sharpening": 2.0, "alpha": 0.6, "beta": 0.2, "gamma": 0.2},
            "documentation_pro": {"name": "DocScalpel 41.4", "sharpening": 4.0, "alpha": 0.7, "beta": 0.2, "gamma": 0.1},
            "recency_bias": {"name": "Temporal Echo v1", "sharpening": 3.0, "alpha": 0.2, "beta": 0.2, "gamma": 0.6}
        }
        
    def get_arm(self, name: str) -> ModularManifoldArm:
        genome = self.genomes.get(name, self.genomes["hyper_resolution_37"])
        return ModularManifoldArm(genome)

    def list_theories(self) -> List[str]:
        return list(self.genomes.keys())
