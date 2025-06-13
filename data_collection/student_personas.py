#!/usr/bin/env python3
"""
Enhanced Student Persona Generator

This module creates detailed CS student personas based on psychological traits:
- Goal Commitment (High/Low)
- Motivation (High/Low)
- Self-Efficacy (High/Low)
"""

import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StudentTraits:
    """Represents the psychological traits of a CS student"""

    goal_commitment: str  # "high" or "low"
    motivation: str  # "high" or "low"
    self_efficacy: str  # "high" or "low"
    year_level: str  # "freshman", "sophomore", "junior", "senior"


class StudentPersonaGenerator:
    """
    Generates realistic CS student personas with specific psychological traits
    """

    def __init__(self):
        self.trait_combinations = [
            # goal_commitment, motivation, self_efficacy
            ("high", "high", "high"),  # Ideal student
            ("high", "high", "low"),  # Hardworking but lacks confidence
            ("high", "low", "high"),  # Skilled but burned out
            ("high", "low", "low"),  # Determined but struggling
            ("low", "high", "high"),  # Talented but unfocused
            ("low", "high", "low"),  # Enthusiastic but uncertain
            ("low", "low", "high"),  # Capable but disengaged
            ("low", "low", "low"),  # At-risk student
        ]

        self.year_levels = ["freshman", "sophomore", "junior", "senior"]
        # self.year_levels = ["freshman", ]

    def generate_persona_description(self, traits: StudentTraits) -> str:
        """
        Generate a detailed persona description based on traits
        """
        # Base description template
        base_templates = {
            ("high", "high", "high"): [
                "highly motivated CS {year} with strong programming skills and clear career goals",
                "dedicated CS {year} who excels academically and actively pursues coding projects",
                "ambitious CS {year} with excellent technical abilities and strong academic commitment",
            ],
            ("high", "high", "low"): [
                "hardworking CS {year} who is passionate about programming but often doubts their abilities",
                "committed CS {year} with clear goals but struggles with imposter syndrome",
                "determined CS {year} who puts in extra effort but lacks confidence in their skills",
            ],
            ("high", "low", "high"): [
                "skilled CS {year} who is technically competent but feels burned out",
                "capable CS {year} with strong abilities but lacks enthusiasm for coursework",
                "talented CS {year} who finds programming easy but questions their passion for the field",
            ],
            ("high", "low", "low"): [
                "CS {year} who wants to succeed but feels overwhelmed and lacks confidence",
                "determined CS {year} with goals but struggles with both motivation and self-doubt",
                "persistent CS {year} committed to CS despite feeling behind and lacking energy",
            ],
            ("low", "high", "high"): [
                "naturally gifted CS {year} who enjoys programming but hasn't committed to specific career plans",
                "enthusiastic CS {year} with strong skills but prefers to keep options open",
                "curious CS {year} who excels at coding but avoids long-term goal setting",
            ],
            ("low", "high", "low"): [
                "eager CS {year} who is excited about programming but unsure about their abilities and future",
                "enthusiastic CS {year} interested in CS but worried about competitiveness",
                "motivated CS {year} who loves coding but lacks confidence in their technical skills",
            ],
            ("low", "low", "high"): [
                "technically skilled CS {year} who can handle coursework but lacks engagement",
                "competent CS {year} who finds programming routine and shows little academic drive",
                "capable CS {year} with abilities but no clear direction or enthusiasm",
            ],
            ("low", "low", "low"): [
                "CS {year} who questions their fit in computer science and struggles with confidence and direction",
                "uncertain CS {year} who feels lost in CS and lacks both technical confidence and goals",
                "disengaged CS {year} who continues studying CS but doubts their abilities and future",
            ],
        }

        trait_key = (traits.goal_commitment, traits.motivation, traits.self_efficacy)
        templates = base_templates.get(trait_key, ["CS {year} student"])

        template = random.choice(templates)
        return template.format(year=traits.year_level)

    def get_random_persona(self) -> Tuple[StudentTraits, str]:
        """
        Get a random persona with traits and description
        """
        goal_commit, motivation, self_eff = random.choice(self.trait_combinations)
        year = random.choice(self.year_levels)

        traits = StudentTraits(
            goal_commitment=goal_commit,
            motivation=motivation,
            self_efficacy=self_eff,
            year_level=year,
        )

        description = self.generate_persona_description(traits)
        return traits, description


# Example usage and testing
if __name__ == "__main__":
    generator = StudentPersonaGenerator()

    print("🎭 Enhanced CS Student Persona Examples")
    print("=" * 50)

    # Generate 10 example personas
    for i in range(10):
        traits, description = generator.get_random_persona()

        print(f"\n{i+1}. {description}")
        print(
            f"   Traits: Goal Commitment={traits.goal_commitment}, "
            f"Motivation={traits.motivation}, Self-Efficacy={traits.self_efficacy}"
        )
