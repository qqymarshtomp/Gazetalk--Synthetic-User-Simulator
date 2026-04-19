from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # optional for local non-API testing
    OpenAI = None


@lru_cache(maxsize=1)
def get_openai_client() -> Optional["OpenAI"]:
    """Return a cached OpenAI client if the dependency and API key are available."""
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

TOKEN_RE = re.compile(r"[A-Za-z]+|[.,!?]")

# =========================
# 1. Prompt templates
# =========================
HDT_PROMPT_TEMPLATE = """
You are a Human Digital Twin simulating a patient with {severity_label} Broca's Aphasia using the GazeTalk eye-tracking system.
Context/Task: {context}
Patient Identity:
- Country/Culture: {country_or_culture}
- Primary language background: {primary_language}
- Age group: {age_group}
- Gender: {gender}
- Interests: {interests}
- Daily routines: {daily_routines}
- Communication goals: {communication_goals}
- Active training topic: {active_topic}
- High-frequency personal words from recent history: {history_keywords}

Clinical & Linguistic Constraints:
- Telegraphic Speech: You can ONLY use concrete nouns and action verbs. Absolutely NO function words.
- Limit output to 1-4 words per turn. If the rehabilitation week is above 2, you may use up to 5 words.
- Circumlocution: you may forget the exact keyword and use a roundabout description.
- Input errors: introduce occasional spelling errors, repeated letters, or logical jumps to simulate gaze-tracking fatigue.

Identity & Topic Constraints:
- Stay consistent with the patient identity.
- Prefer words and small topics that match this patient's age, culture, routines, and interests.
- Focus mainly on the active training topic, unless the assistant strongly changes the topic.
- Keep the utterance in simple English words so the click simulator remains comparable across patients.

Task:
Respond naturally to the GazeTalk assistant based on the context. Strictly maintain the clinical constraints above. NEVER output fluent full sentences.
"""

GAZETALK_PROMPT_TEMPLATE = """
You are the GazeTalk AI rehabilitation assistant.
Your user has {severity_label} aphasia and uses an eye-tracker.
Patient identity:
- Country/Culture: {country_or_culture}
- Age group: {age_group}
- Interests: {interests}
- Active topic: {active_topic}

Instructions:
- Ask short, simple, concrete questions.
- Keep the conversation close to the active topic.
- Gently suggest words that this patient is likely to use based on identity and recent history.
- Keep responses short and encouraging.
- Ask only one question at a time.
"""

# =========================
# 2. Persona and topic model
# =========================
@dataclass
class PatientPersona:
    patient_id: str
    display_name: str
    country_or_culture: str
    primary_language: str
    age_group: str
    gender: str
    interests: List[str]
    daily_routines: List[str]
    communication_goals: List[str]
    aphasia_severity: str = "severe"
    system_familiarity: float = 0.45
    fatigue_sensitivity: float = 1.0


CULTURE_DEFAULT_LANGUAGE = {
    "China": "Chinese background",
    "Denmark": "Danish background",
    "Italy": "Italian background",
    "Indonesia": "Indonesian background",
    "Generic": "English background",
}

TOPIC_BANK: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "Generic": {
        "family": {
            "nouns": ["family", "daughter", "son", "wife", "husband", "mother", "father", "visit"],
            "verbs": ["call", "see", "visit", "miss", "hug"],
            "questions": [
                "Would you like to talk about family today?",
                "Who did you see at home?",
                "Do you want to tell me about family?",
            ],
        },
        "food": {
            "nouns": ["food", "rice", "soup", "tea", "coffee", "bread", "fish", "fruit"],
            "verbs": ["eat", "cook", "buy", "drink", "want"],
            "questions": [
                "What food do you want today?",
                "What did you eat?",
                "Would you like to talk about food?",
            ],
        },
        "weather": {
            "nouns": ["weather", "rain", "sun", "wind", "cloud", "cold", "heat"],
            "verbs": ["see", "like", "watch"],
            "questions": [
                "How is the weather outside?",
                "Do you see sun or rain?",
                "Would you like to talk about weather?",
            ],
        },
        "health": {
            "nouns": ["head", "pain", "sleep", "medicine", "doctor", "nurse", "body"],
            "verbs": ["need", "hurt", "rest", "sleep", "take"],
            "questions": [
                "How does your body feel today?",
                "Do you need rest or medicine?",
                "Would you like to talk about health?",
            ],
        },
        "shopping": {
            "nouns": ["shop", "market", "bag", "milk", "bread", "vegetable", "fruit"],
            "verbs": ["buy", "go", "carry", "want"],
            "questions": [
                "What do you want from the shop?",
                "Did you go to the market?",
                "Would you like to talk about shopping?",
            ],
        },
        "sports": {
            "nouns": ["ball", "team", "match", "goal", "game", "run", "bike"],
            "verbs": ["watch", "play", "run", "train", "like"],
            "questions": [
                "Did you watch a match?",
                "What sport do you like?",
                "Would you like to talk about sports?",
            ],
        },
        "books": {
            "nouns": ["book", "story", "page", "paper", "poem"],
            "verbs": ["read", "like", "open"],
            "questions": [
                "Did you read a book?",
                "What story do you like?",
                "Would you like to talk about books?",
            ],
        },
        "garden": {
            "nouns": ["garden", "flower", "tree", "grass", "soil", "plant"],
            "verbs": ["water", "grow", "cut", "plant"],
            "questions": [
                "Did you see the garden today?",
                "What plant do you like?",
                "Would you like to talk about the garden?",
            ],
        },
        "friends": {
            "nouns": ["friend", "coffee", "talk", "walk", "weekend"],
            "verbs": ["meet", "call", "see", "talk"],
            "questions": [
                "Did you meet a friend?",
                "Who do you want to call?",
                "Would you like to talk about friends?",
            ],
        },
        "work": {
            "nouns": ["work", "office", "team", "meeting", "job", "project"],
            "verbs": ["go", "work", "meet", "plan"],
            "questions": [
                "Do you want to talk about work?",
                "What job task do you remember?",
                "Would you like to talk about your team?",
            ],
        },
        "travel": {
            "nouns": ["trip", "train", "bus", "road", "hotel", "ticket"],
            "verbs": ["go", "ride", "visit", "plan"],
            "questions": [
                "Do you want to talk about a trip?",
                "Did you ride a train or bus?",
                "Where do you want to go?",
            ],
        },
        "religion": {
            "nouns": ["church", "prayer", "temple", "mosque", "song", "service"],
            "verbs": ["pray", "go", "sing", "join"],
            "questions": [
                "Would you like to talk about prayer or service?",
                "Did you go to church or temple?",
                "Do you want to talk about religion today?",
            ],
        },
    },
    "China": {
        "food": {
            "nouns": ["rice", "noodle", "dumpling", "porridge", "tea", "fish", "vegetable", "market"],
            "verbs": ["cook", "buy", "eat", "drink", "steam"],
            "questions": [
                "Did you eat rice or noodles today?",
                "What do you want from the market?",
                "Would you like to talk about cooking?",
            ],
        },
        "family": {
            "nouns": ["family", "daughter", "son", "grandchild", "mother", "father", "home"],
            "verbs": ["call", "visit", "see", "help"],
            "questions": [
                "Did family visit today?",
                "Do you want to call your daughter or son?",
                "Would you like to talk about your family?",
            ],
        },
        "shopping": {
            "nouns": ["market", "vegetable", "fruit", "bag", "street", "shop"],
            "verbs": ["buy", "go", "carry", "pick"],
            "questions": [
                "Did you go to the vegetable market?",
                "What do you want to buy?",
                "Would you like to talk about shopping?",
            ],
        },
    },
    "Denmark": {
        "weather": {
            "nouns": ["wind", "rain", "cloud", "sun", "bike", "cold"],
            "verbs": ["watch", "bike", "walk", "see"],
            "questions": [
                "How is the wind outside?",
                "Did you go by bike today?",
                "Would you like to talk about the weather?",
            ],
        },
        "sports": {
            "nouns": ["football", "match", "team", "goal", "run", "gym"],
            "verbs": ["watch", "play", "train", "run"],
            "questions": [
                "Did you watch football today?",
                "Do you like the gym or football?",
                "Would you like to talk about sports?",
            ],
        },
        "friends": {
            "nouns": ["friend", "coffee", "weekend", "city", "walk"],
            "verbs": ["meet", "call", "walk", "talk"],
            "questions": [
                "Did you meet friends this week?",
                "Would you like to talk about coffee with friends?",
                "Who do you want to call?",
            ],
        },
    },
    "Italy": {
        "food": {
            "nouns": ["coffee", "bread", "pasta", "market", "tomato", "olive"],
            "verbs": ["cook", "eat", "drink", "buy"],
            "questions": [
                "Did you drink coffee today?",
                "What food do you want to cook?",
                "Would you like to talk about food?",
            ],
        },
        "garden": {
            "nouns": ["garden", "flower", "tree", "soil", "sun", "water"],
            "verbs": ["water", "grow", "cut", "plant"],
            "questions": [
                "Did you work in the garden?",
                "What flower do you like?",
                "Would you like to talk about the garden?",
            ],
        },
        "family": {
            "nouns": ["family", "grandchild", "home", "visit", "table"],
            "verbs": ["visit", "eat", "call", "see"],
            "questions": [
                "Did family visit today?",
                "Do you want to talk about your grandchild?",
                "Would you like to talk about family?",
            ],
        },
    },
    "Indonesia": {
        "food": {
            "nouns": ["rice", "soup", "tea", "fish", "spice", "market"],
            "verbs": ["cook", "buy", "eat", "drink"],
            "questions": [
                "What food do you want today?",
                "Did you go to the market?",
                "Would you like to talk about cooking?",
            ],
        },
        "religion": {
            "nouns": ["prayer", "mosque", "song", "service", "family"],
            "verbs": ["pray", "go", "join", "sing"],
            "questions": [
                "Did you go to prayer today?",
                "Would you like to talk about prayer or family?",
                "Do you want to talk about religious activity?",
            ],
        },
        "family": {
            "nouns": ["family", "child", "mother", "father", "home", "visit"],
            "verbs": ["call", "help", "visit", "see"],
            "questions": [
                "Did you see your family today?",
                "Do you want to talk about your child?",
                "Would you like to talk about family?",
            ],
        },
    },
}

AGE_TOPIC_PRIORS = {
    "young_adult": {"friends": 3, "sports": 3, "work": 2, "travel": 2, "food": 1},
    "middle_aged": {"family": 3, "work": 2, "shopping": 2, "food": 2, "health": 1, "weather": 1},
    "senior": {"family": 3, "health": 3, "garden": 2, "weather": 2, "food": 2},
}

INTEREST_TO_TOPICS = {
    "cooking": ["food", "shopping"],
    "family": ["family"],
    "shopping": ["shopping", "food"],
    "weather": ["weather"],
    "football": ["sports"],
    "sport": ["sports"],
    "sports": ["sports"],
    "gym": ["sports"],
    "gaming": ["friends", "sports"],
    "friends": ["friends"],
    "books": ["books"],
    "reading": ["books"],
    "garden": ["garden"],
    "gardening": ["garden"],
    "religion": ["religion", "family"],
    "coffee": ["food", "friends"],
    "health": ["health"],
    "travel": ["travel"],
    "music": ["friends", "religion"],
    "work": ["work"],
}

AGE_ALIAS = {
    "20-35": "young_adult",
    "18-35": "young_adult",
    "young": "young_adult",
    "young_adult": "young_adult",
    "35-55": "middle_aged",
    "40-60": "middle_aged",
    "middle-aged": "middle_aged",
    "middle_aged": "middle_aged",
    "50+": "senior",
    "60+": "senior",
    "65+": "senior",
    "senior": "senior",
    "older": "senior",
}


@dataclass
class SyntheticUserProfile:
    profile_id: str

    # Selection accuracy
    visible_letter_accuracy: float = 0.95
    hidden_letter_accuracy: float = 0.86
    recommendation_accept_prob: float = 0.90

    # Error behavior
    typo_prob_visible: float = 0.03
    typo_prob_hidden: float = 0.08
    delete_after_error_prob: float = 0.96

    # Base timing
    dwell_time_visible_mean: float = 0.90
    dwell_time_hidden_mean: float = 1.35
    dwell_time_word_mean: float = 0.80
    dwell_time_space_mean: float = 0.65
    dwell_time_delete_mean: float = 0.75
    dwell_time_punctuation_mean: float = 0.85
    reaction_time_base_mean: float = 1.80
    reaction_time_std: float = 0.25

    # Old fatigue baseline kept for backward compatibility
    fatigue_growth_per_event: float = 0.015

    # New fatigue-state parameters
    fatigue_sensitivity: float = 1.0
    learning_phase_events: int = 5
    late_phase_start: int = 18
    base_fatigue_gain: float = 0.020
    recovery_rate: float = 0.012
    stable_recovery_bonus: float = 0.008
    error_fatigue_cost: float = 0.030
    correction_fatigue_cost: float = 0.020
    hesitation_fatigue_cost: float = 0.012
    difficult_action_bonus: float = 0.016
    fatigue_accuracy_drop_visible: float = 0.10
    fatigue_accuracy_drop_hidden: float = 0.16
    fatigue_timing_multiplier: float = 0.65
    variability_growth: float = 0.55
    hesitation_base_prob: float = 0.03
    hesitation_fatigue_boost: float = 0.18

    # Recommendation weighting
    history_weight_scale: float = 1.0
    lm_weight_scale: float = 1.0
    recency_weight_scale: float = 1.0


@dataclass
class FatigueState:
    level: float = 0.0
    event_index: int = 0
    stability_streak: int = 0
    last_phase: str = "learning"


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def age_bucket(age_group: str) -> str:
    return AGE_ALIAS.get(age_group, age_group if age_group in AGE_TOPIC_PRIORS else "middle_aged")


def build_patient_persona(
    patient_id: str,
    display_name: str,
    country_or_culture: str,
    age_group: str,
    gender: str,
    interests: List[str],
    daily_routines: Optional[List[str]] = None,
    communication_goals: Optional[List[str]] = None,
    aphasia_severity: str = "severe",
    primary_language: Optional[str] = None,
    system_familiarity: float = 0.45,
    fatigue_sensitivity: float = 1.0,
) -> PatientPersona:
    return PatientPersona(
        patient_id=patient_id,
        display_name=display_name,
        country_or_culture=country_or_culture,
        primary_language=primary_language or CULTURE_DEFAULT_LANGUAGE.get(country_or_culture, CULTURE_DEFAULT_LANGUAGE["Generic"]),
        age_group=age_group,
        gender=gender,
        interests=interests,
        daily_routines=daily_routines or ["home", "rest"],
        communication_goals=communication_goals or ["daily needs", "family chat"],
        aphasia_severity=aphasia_severity,
        system_familiarity=system_familiarity,
        fatigue_sensitivity=fatigue_sensitivity,
    )


def build_synthetic_user_profile(profile_id: str, severity: str = "severe", seed: int = 42) -> SyntheticUserProfile:
    rng = random.Random(seed)
    presets = {
        "severe": {
            "visible_letter_accuracy": 0.92,
            "hidden_letter_accuracy": 0.80,
            "recommendation_accept_prob": 0.88,
            "typo_prob_visible": 0.05,
            "typo_prob_hidden": 0.11,
            "delete_after_error_prob": 0.97,
            "dwell_time_visible_mean": 1.10,
            "dwell_time_hidden_mean": 1.60,
            "dwell_time_word_mean": 0.95,
            "dwell_time_space_mean": 0.80,
            "dwell_time_delete_mean": 0.90,
            "dwell_time_punctuation_mean": 1.00,
            "reaction_time_base_mean": 2.20,
            "reaction_time_std": 0.30,
            "fatigue_growth_per_event": 0.025,
            "fatigue_sensitivity": 1.20,
            "base_fatigue_gain": 0.026,
            "recovery_rate": 0.010,
            "stable_recovery_bonus": 0.006,
            "error_fatigue_cost": 0.040,
            "correction_fatigue_cost": 0.025,
            "difficult_action_bonus": 0.020,
            "fatigue_accuracy_drop_visible": 0.13,
            "fatigue_accuracy_drop_hidden": 0.18,
            "fatigue_timing_multiplier": 0.75,
            "variability_growth": 0.70,
        },
        "moderate": {
            "visible_letter_accuracy": 0.95,
            "hidden_letter_accuracy": 0.86,
            "recommendation_accept_prob": 0.90,
            "typo_prob_visible": 0.03,
            "typo_prob_hidden": 0.08,
            "delete_after_error_prob": 0.96,
            "dwell_time_visible_mean": 0.90,
            "dwell_time_hidden_mean": 1.35,
            "dwell_time_word_mean": 0.80,
            "dwell_time_space_mean": 0.65,
            "dwell_time_delete_mean": 0.75,
            "dwell_time_punctuation_mean": 0.85,
            "reaction_time_base_mean": 1.80,
            "reaction_time_std": 0.25,
            "fatigue_growth_per_event": 0.015,
            "fatigue_sensitivity": 1.00,
            "base_fatigue_gain": 0.020,
            "recovery_rate": 0.012,
            "stable_recovery_bonus": 0.008,
            "error_fatigue_cost": 0.030,
            "correction_fatigue_cost": 0.020,
            "difficult_action_bonus": 0.016,
            "fatigue_accuracy_drop_visible": 0.10,
            "fatigue_accuracy_drop_hidden": 0.16,
            "fatigue_timing_multiplier": 0.65,
            "variability_growth": 0.55,
        },
        "mild": {
            "visible_letter_accuracy": 0.97,
            "hidden_letter_accuracy": 0.91,
            "recommendation_accept_prob": 0.93,
            "typo_prob_visible": 0.02,
            "typo_prob_hidden": 0.05,
            "delete_after_error_prob": 0.94,
            "dwell_time_visible_mean": 0.75,
            "dwell_time_hidden_mean": 1.10,
            "dwell_time_word_mean": 0.70,
            "dwell_time_space_mean": 0.55,
            "dwell_time_delete_mean": 0.65,
            "dwell_time_punctuation_mean": 0.75,
            "reaction_time_base_mean": 1.50,
            "reaction_time_std": 0.20,
            "fatigue_growth_per_event": 0.010,
            "fatigue_sensitivity": 0.85,
            "base_fatigue_gain": 0.015,
            "recovery_rate": 0.016,
            "stable_recovery_bonus": 0.010,
            "error_fatigue_cost": 0.022,
            "correction_fatigue_cost": 0.015,
            "difficult_action_bonus": 0.010,
            "fatigue_accuracy_drop_visible": 0.07,
            "fatigue_accuracy_drop_hidden": 0.12,
            "fatigue_timing_multiplier": 0.50,
            "variability_growth": 0.40,
        },
    }
    if severity not in presets:
        raise ValueError(f"Unknown severity: {severity}")
    cfg = presets[severity].copy()

    # Add small user-level variation.
    for key, sigma in [
        ("visible_letter_accuracy", 0.015),
        ("hidden_letter_accuracy", 0.020),
        ("recommendation_accept_prob", 0.020),
        ("typo_prob_visible", 0.010),
        ("typo_prob_hidden", 0.015),
        ("delete_after_error_prob", 0.020),
        ("dwell_time_visible_mean", 0.080),
        ("dwell_time_hidden_mean", 0.100),
        ("dwell_time_word_mean", 0.070),
        ("dwell_time_space_mean", 0.060),
        ("dwell_time_delete_mean", 0.060),
        ("dwell_time_punctuation_mean", 0.060),
        ("reaction_time_base_mean", 0.100),
        ("reaction_time_std", 0.030),
        ("fatigue_growth_per_event", 0.003),
        ("fatigue_sensitivity", 0.050),
        ("base_fatigue_gain", 0.003),
        ("recovery_rate", 0.002),
        ("stable_recovery_bonus", 0.002),
        ("error_fatigue_cost", 0.004),
        ("correction_fatigue_cost", 0.003),
        ("difficult_action_bonus", 0.003),
        ("fatigue_accuracy_drop_visible", 0.015),
        ("fatigue_accuracy_drop_hidden", 0.020),
        ("fatigue_timing_multiplier", 0.040),
        ("variability_growth", 0.050),
    ]:
        cfg[key] = rng.gauss(cfg[key], sigma)

    # Clamp relevant ranges.
    cfg["visible_letter_accuracy"] = clamp(cfg["visible_letter_accuracy"], 0.60, 0.99)
    cfg["hidden_letter_accuracy"] = clamp(cfg["hidden_letter_accuracy"], 0.45, 0.98)
    cfg["recommendation_accept_prob"] = clamp(cfg["recommendation_accept_prob"], 0.60, 0.99)
    cfg["typo_prob_visible"] = clamp(cfg["typo_prob_visible"], 0.0, 0.25)
    cfg["typo_prob_hidden"] = clamp(cfg["typo_prob_hidden"], 0.0, 0.35)
    cfg["delete_after_error_prob"] = clamp(cfg["delete_after_error_prob"], 0.70, 1.0)
    cfg["reaction_time_std"] = clamp(cfg["reaction_time_std"], 0.05, 0.60)
    cfg["fatigue_growth_per_event"] = clamp(cfg["fatigue_growth_per_event"], 0.0, 0.08)
    cfg["fatigue_sensitivity"] = clamp(cfg["fatigue_sensitivity"], 0.5, 1.6)
    cfg["base_fatigue_gain"] = clamp(cfg["base_fatigue_gain"], 0.005, 0.08)
    cfg["recovery_rate"] = clamp(cfg["recovery_rate"], 0.0, 0.05)
    cfg["stable_recovery_bonus"] = clamp(cfg["stable_recovery_bonus"], 0.0, 0.03)
    cfg["error_fatigue_cost"] = clamp(cfg["error_fatigue_cost"], 0.0, 0.08)
    cfg["correction_fatigue_cost"] = clamp(cfg["correction_fatigue_cost"], 0.0, 0.06)
    cfg["difficult_action_bonus"] = clamp(cfg["difficult_action_bonus"], 0.0, 0.05)
    cfg["fatigue_accuracy_drop_visible"] = clamp(cfg["fatigue_accuracy_drop_visible"], 0.0, 0.30)
    cfg["fatigue_accuracy_drop_hidden"] = clamp(cfg["fatigue_accuracy_drop_hidden"], 0.0, 0.35)
    cfg["fatigue_timing_multiplier"] = clamp(cfg["fatigue_timing_multiplier"], 0.1, 1.2)
    cfg["variability_growth"] = clamp(cfg["variability_growth"], 0.0, 1.5)

    return SyntheticUserProfile(profile_id=profile_id, **cfg)


# =========================
# 3. Click and time simulator
# =========================
DEFAULT_CORPUS = [
    "good morning",
    "how are you",
    "i need help",
    "i want water",
    "i want coffee",
    "i want food",
    "i am tired",
    "i am okay",
    "yes",
    "no",
    "stop please",
    "more practice",
    "call family",
    "go home",
    "pain here",
    "need nurse",
    "thank you",
    "bathroom now",
    "head pain",
    "drink water",
    "family visit",
    "want rest",
    "stop session",
]


def personalize_user_profile(base_profile: SyntheticUserProfile, persona: PatientPersona) -> SyntheticUserProfile:
    data = asdict(base_profile)
    age_key = age_bucket(persona.age_group)

    # Low familiarity increases errors and slows selections.
    familiarity_penalty = 1.0 - clamp(persona.system_familiarity, 0.0, 1.0)
    data["visible_letter_accuracy"] = clamp(data["visible_letter_accuracy"] - 0.05 * familiarity_penalty, 0.60, 0.99)
    data["hidden_letter_accuracy"] = clamp(data["hidden_letter_accuracy"] - 0.08 * familiarity_penalty, 0.45, 0.98)
    data["typo_prob_visible"] = clamp(data["typo_prob_visible"] + 0.03 * familiarity_penalty, 0.0, 0.30)
    data["typo_prob_hidden"] = clamp(data["typo_prob_hidden"] + 0.05 * familiarity_penalty, 0.0, 0.40)
    data["reaction_time_base_mean"] = max(0.5, data["reaction_time_base_mean"] * (1.0 + 0.20 * familiarity_penalty))

    # Age group changes speed and fatigue profile.
    if age_key == "senior":
        data["reaction_time_base_mean"] *= 1.10
        data["dwell_time_visible_mean"] *= 1.08
        data["dwell_time_hidden_mean"] *= 1.10
        data["fatigue_growth_per_event"] *= 1.10
        data["fatigue_sensitivity"] *= 1.08
        data["base_fatigue_gain"] *= 1.08
        data["variability_growth"] *= 1.08
    elif age_key == "young_adult":
        data["reaction_time_base_mean"] *= 0.95
        data["fatigue_growth_per_event"] *= 0.90
        data["fatigue_sensitivity"] *= 0.95
        data["recovery_rate"] *= 1.05

    # Persona fatigue sensitivity changes the fatigue dynamics directly.
    data["fatigue_sensitivity"] = clamp(data["fatigue_sensitivity"] * persona.fatigue_sensitivity, 0.5, 2.0)
    data["base_fatigue_gain"] = clamp(data["base_fatigue_gain"] * persona.fatigue_sensitivity, 0.0, 0.10)
    data["fatigue_growth_per_event"] = clamp(data["fatigue_growth_per_event"] * persona.fatigue_sensitivity, 0.0, 0.12)
    data["fatigue_timing_multiplier"] = clamp(data["fatigue_timing_multiplier"] * (0.95 + 0.10 * persona.fatigue_sensitivity), 0.2, 1.5)

    # Interests shift recommendation behavior slightly.
    if "family" in [x.lower() for x in persona.interests]:
        data["history_weight_scale"] *= 1.05
    if "books" in [x.lower() for x in persona.interests] or "reading" in [x.lower() for x in persona.interests]:
        data["lm_weight_scale"] *= 1.05

    return SyntheticUserProfile(**data)


def merge_topic_spec(persona: PatientPersona, topic: str) -> Dict[str, List[str]]:
    generic_spec = TOPIC_BANK["Generic"].get(topic, {"nouns": [topic], "verbs": ["want"], "questions": [f"Would you like to talk about {topic}?"]})
    culture_spec = TOPIC_BANK.get(persona.country_or_culture, {}).get(topic, {})
    return {
        "nouns": list(dict.fromkeys(generic_spec.get("nouns", []) + culture_spec.get("nouns", []))),
        "verbs": list(dict.fromkeys(generic_spec.get("verbs", []) + culture_spec.get("verbs", []))),
        "questions": list(dict.fromkeys(generic_spec.get("questions", []) + culture_spec.get("questions", []))),
    }


def get_persona_topic_weights(persona: PatientPersona) -> Counter:
    weights = Counter()
    culture_topics = TOPIC_BANK.get(persona.country_or_culture, {})
    for topic in culture_topics:
        weights[topic] += 2

    for topic, weight in AGE_TOPIC_PRIORS.get(age_bucket(persona.age_group), {}).items():
        weights[topic] += weight

    for interest in persona.interests:
        for topic in INTEREST_TO_TOPICS.get(interest.lower(), []):
            weights[topic] += 4

    for routine in persona.daily_routines:
        r = routine.lower()
        if "shop" in r or "market" in r:
            weights["shopping"] += 2
        if "cook" in r or "kitchen" in r:
            weights["food"] += 2
        if "walk" in r or "bike" in r:
            weights["weather"] += 1
            weights["sports"] += 1
        if "garden" in r:
            weights["garden"] += 2
        if "pray" in r or "church" in r or "mosque" in r or "temple" in r:
            weights["religion"] += 2

    for goal in persona.communication_goals:
        g = goal.lower()
        if "family" in g:
            weights["family"] += 2
        if "need" in g or "daily" in g:
            weights["health"] += 1
            weights["food"] += 1

    if not weights:
        weights.update({"family": 3, "food": 2, "weather": 1})
    return weights


def weighted_topics(counter: Counter, top_k: int = 4) -> List[str]:
    return [topic for topic, _ in counter.most_common(top_k)]


def synthesize_telegraphic_phrase(topic: str, persona: PatientPersona, rng: random.Random) -> str:
    spec = merge_topic_spec(persona, topic)
    nouns = spec["nouns"] or [topic]
    verbs = spec["verbs"] or ["want"]
    pattern = rng.choice(["vn", "nn", "vnn", "vn", "nn"])

    if pattern == "vn":
        phrase = f"{rng.choice(verbs)} {rng.choice(nouns)}"
    elif pattern == "nn":
        phrase = f"{rng.choice(nouns)} {rng.choice(nouns)}"
    else:
        phrase = f"{rng.choice(verbs)} {rng.choice(nouns)} {rng.choice(nouns)}"

    # avoid repeated identical adjacent words when possible
    toks = phrase.split()
    cleaned = [toks[0]]
    for tok in toks[1:]:
        if tok != cleaned[-1]:
            cleaned.append(tok)
    return " ".join(cleaned[:4])


def build_persona_seed_corpus(persona: PatientPersona, n_phrases: int = 60, seed: int = 42) -> Tuple[List[str], Counter]:
    rng = random.Random(seed)
    topic_weights = get_persona_topic_weights(persona)
    topics, weights = zip(*topic_weights.items())
    phrases = []
    topic_counts = Counter()
    for _ in range(n_phrases):
        topic = rng.choices(topics, weights=weights, k=1)[0]
        topic_counts[topic] += 1
        phrases.append(synthesize_telegraphic_phrase(topic, persona, rng))
    return phrases, topic_counts


def generate_daily_history(
    persona: PatientPersona,
    days: int = 7,
    min_utterances_per_day: int = 8,
    max_utterances_per_day: int = 14,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    topic_weights = get_persona_topic_weights(persona)
    topics, weights = zip(*topic_weights.items())

    daily_logs = []
    all_utterances: List[str] = []
    topic_counter = Counter()

    for day_idx in range(1, days + 1):
        n_utterances = rng.randint(min_utterances_per_day, max_utterances_per_day)
        day_utterances = []
        day_topics = Counter()
        for _ in range(n_utterances):
            topic = rng.choices(topics, weights=weights, k=1)[0]
            phrase = synthesize_telegraphic_phrase(topic, persona, rng)
            day_utterances.append(phrase)
            all_utterances.append(phrase)
            topic_counter[topic] += 1
            day_topics[topic] += 1
        daily_logs.append({
            "day_index": day_idx,
            "utterances": day_utterances,
            "topic_counts": dict(day_topics),
        })

    words = [w.lower() for utt in all_utterances for w in TOKEN_RE.findall(utt) if w.isalpha()]
    top_words = [w for w, _ in Counter(words).most_common(12)]
    total_topics = sum(topic_counter.values()) or 1
    topic_distribution = {
        topic: round(count / total_topics, 3)
        for topic, count in topic_counter.most_common()
    }

    return {
        "daily_logs": daily_logs,
        "all_utterances": all_utterances,
        "top_words": top_words,
        "topic_distribution": topic_distribution,
        "top_topics": [topic for topic, _ in topic_counter.most_common(5)],
    }


def make_topic_start_message(persona: PatientPersona, active_topic: str, seed: int = 42) -> str:
    rng = random.Random(seed)
    spec = merge_topic_spec(persona, active_topic)
    questions = spec.get("questions") or [f"Would you like to talk about {active_topic}?"]
    return rng.choice(questions)


def build_hdt_system_prompt(
    context: str,
    persona: PatientPersona,
    active_topic: str,
    history_keywords: List[str],
    weeks_progress: int,
) -> str:
    prompt = HDT_PROMPT_TEMPLATE.format(
        severity_label=persona.aphasia_severity,
        context=context,
        country_or_culture=persona.country_or_culture,
        primary_language=persona.primary_language,
        age_group=persona.age_group,
        gender=persona.gender,
        interests=", ".join(persona.interests),
        daily_routines=", ".join(persona.daily_routines),
        communication_goals=", ".join(persona.communication_goals),
        active_topic=active_topic,
        history_keywords=", ".join(history_keywords[:10]) if history_keywords else "none",
    )
    if weeks_progress > 2:
        prompt += "\n- Progress Note: You have improved slightly. You can now use 3-5 words and make fewer spelling mistakes."
    return prompt


def build_gazetalk_system_prompt(
    persona: PatientPersona,
    active_topic: str,
) -> str:
    return GAZETALK_PROMPT_TEMPLATE.format(
        severity_label=persona.aphasia_severity,
        country_or_culture=persona.country_or_culture,
        age_group=persona.age_group,
        interests=", ".join(persona.interests),
        active_topic=active_topic,
    )


# =========================
# 3. Click and time simulator
# =========================
class GazetalkClickSimulator:
    def __init__(
        self,
        base_corpus: Optional[List[str]] = None,
        visible_letter_slots: int = 9,
        word_prediction_slots: int = 4,
        fallback_letter_clicks: int = 2,
        auto_space_after_word_prediction: bool = True,
        punctuation_clicks: int = 1,
        delete_clicks: int = 1,
        base_lm_weight: float = 0.55,
        base_history_weight: float = 0.35,
        base_recency_weight: float = 0.10,
        seed: int = 42,
    ):
        self.visible_letter_slots = visible_letter_slots
        self.word_prediction_slots = word_prediction_slots
        self.fallback_letter_clicks = fallback_letter_clicks
        self.auto_space_after_word_prediction = auto_space_after_word_prediction
        self.punctuation_clicks = punctuation_clicks
        self.delete_clicks = delete_clicks
        self.base_lm_weight = base_lm_weight
        self.base_history_weight = base_history_weight
        self.base_recency_weight = base_recency_weight
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.user_word_counts = Counter()
        self.user_recency = deque(maxlen=300)
        self.rng = random.Random(seed)
        corpus = DEFAULT_CORPUS if base_corpus is None else base_corpus
        self.fit_corpus(corpus)

    def tokenize(self, text: str) -> List[str]:
        return [tok.lower() for tok in TOKEN_RE.findall(text)]

    def words_only(self, text: str) -> List[str]:
        return [tok for tok in self.tokenize(text) if tok.isalpha()]

    def lexicon(self) -> set:
        return set(self.unigrams.keys()) | set(self.user_word_counts.keys())

    def fit_corpus(self, phrases: List[str]) -> None:
        for phrase in phrases:
            self._update_language_model(phrase)

    def _update_language_model(self, text: str) -> None:
        words = self.words_only(text)
        prev = "<s>"
        for w in words:
            self.unigrams[w] += 1
            self.bigrams[prev][w] += 1
            prev = w
        self.bigrams[prev]["</s>"] += 1

    def update_user_history(self, text: str) -> None:
        words = self.words_only(text)
        for w in words:
            self.user_word_counts[w] += 1
            self.user_recency.append(w)
        self._update_language_model(text)

    def preload_history(self, history_utterances: List[str]) -> None:
        for utt in history_utterances:
            self.update_user_history(utt)

    def _recency_score(self, word: str) -> float:
        for rank, recent_word in enumerate(reversed(self.user_recency), start=1):
            if recent_word == word:
                return 1.0 / rank
        return 0.0

    def _normalized_weights(self, profile: SyntheticUserProfile):
        lm_w = self.base_lm_weight * profile.lm_weight_scale
        hist_w = self.base_history_weight * profile.history_weight_scale
        rec_w = self.base_recency_weight * profile.recency_weight_scale
        s = lm_w + hist_w + rec_w
        return lm_w / s, hist_w / s, rec_w / s

    def _word_score(self, word: str, prev_word: str, profile: SyntheticUserProfile) -> float:
        vocab = max(len(self.lexicon()), 1)
        unigram_total = max(sum(self.unigrams.values()), 1)
        bigram_total = max(sum(self.bigrams[prev_word].values()), 1)
        history_total = max(sum(self.user_word_counts.values()), 1)
        p_uni = (self.unigrams[word] + 1.0) / (unigram_total + vocab)
        p_bi = (self.bigrams[prev_word][word] + 1.0) / (bigram_total + vocab)
        p_hist = self.user_word_counts[word] / history_total
        p_recent = self._recency_score(word)
        lm_score = 0.7 * p_bi + 0.3 * p_uni
        lm_w, hist_w, rec_w = self._normalized_weights(profile)
        return lm_w * lm_score + hist_w * p_hist + rec_w * p_recent

    def recommend_words(self, context_words: List[str], profile: SyntheticUserProfile, prefix: str = "") -> List[str]:
        prev_word = context_words[-1] if context_words else "<s>"
        candidates = []
        for word in self.lexicon():
            if prefix and not word.startswith(prefix):
                continue
            candidates.append((self._word_score(word, prev_word, profile), word))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [w for _, w in candidates[: self.word_prediction_slots]]

    def visible_letters(self, context_words: List[str], profile: SyntheticUserProfile, prefix: str = "") -> List[str]:
        prev_word = context_words[-1] if context_words else "<s>"
        letter_scores = Counter()
        for word in self.lexicon():
            if prefix and not word.startswith(prefix):
                continue
            if len(word) <= len(prefix):
                continue
            next_char = word[len(prefix)]
            letter_scores[next_char] += self._word_score(word, prev_word, profile)
        if not letter_scores:
            for ch in "etaoinshrdlucmfwypvbgkjqxz":
                letter_scores[ch] += 1.0
        return [ch for ch, _ in letter_scores.most_common(self.visible_letter_slots)]

    # ---------- Improved fatigue model ----------
    def init_fatigue_state(self, profile: SyntheticUserProfile) -> FatigueState:
        base_level = 0.04 * profile.fatigue_sensitivity
        return FatigueState(level=clamp(base_level, 0.0, 0.15), event_index=0, stability_streak=0, last_phase="learning")

    def get_phase(self, state: FatigueState, profile: SyntheticUserProfile) -> str:
        if state.event_index < profile.learning_phase_events:
            return "learning"
        if state.event_index < profile.late_phase_start:
            return "middle"
        return "late"

    def phase_fatigue_multiplier(self, phase: str) -> float:
        if phase == "learning":
            return 0.7
        if phase == "middle":
            return 1.0
        return 1.35

    def phase_variability_multiplier(self, phase: str) -> float:
        if phase == "learning":
            return 1.25
        if phase == "middle":
            return 1.0
        return 1.35

    def action_effort(self, action_type: str, hidden: bool = False) -> float:
        effort = {
            "word_prediction": 0.6,
            "visible_letter": 1.0,
            "hidden_letter": 1.35,
            "space": 0.45,
            "delete": 0.65,
            "punctuation": 0.6,
            "hesitation": 0.3,
        }.get(action_type, 0.9)
        if hidden:
            effort += 0.15
        return effort

    def update_fatigue_state(
        self,
        state: FatigueState,
        profile: SyntheticUserProfile,
        action_type: str,
        was_error: bool = False,
        was_correction: bool = False,
        easy_action: bool = False,
    ) -> FatigueState:
        state.event_index += 1
        phase = self.get_phase(state, profile)
        phase_mult = self.phase_fatigue_multiplier(phase)
        effort = self.action_effort(action_type, hidden=(action_type == "hidden_letter"))

        gain = profile.base_fatigue_gain * profile.fatigue_sensitivity * phase_mult * effort

        if action_type in {"hidden_letter", "hesitation"}:
            gain += profile.difficult_action_bonus
        if was_error:
            gain += profile.error_fatigue_cost
        if was_correction:
            gain += profile.correction_fatigue_cost
        if action_type == "hesitation":
            gain += profile.hesitation_fatigue_cost

        recovery = 0.0
        if easy_action and not was_error and action_type in {"word_prediction", "visible_letter", "space"}:
            state.stability_streak += 1
            recovery += profile.recovery_rate
            if state.stability_streak >= 3:
                recovery += profile.stable_recovery_bonus
        else:
            state.stability_streak = 0

        state.level = clamp(state.level + gain - recovery, 0.0, 1.0)
        state.last_phase = phase
        return state

    def _positive_gaussian(self, mean: float, std: float, lower: float = 0.05) -> float:
        return max(lower, self.rng.gauss(mean, std))

    def sample_action_time(
        self,
        action_type: str,
        profile: SyntheticUserProfile,
        fatigue_state: FatigueState,
        weeks_progress: int,
    ) -> float:
        phase = self.get_phase(fatigue_state, profile)
        progress_factor = max(0.65, 1.0 - 0.05 * max(weeks_progress - 1, 0))
        fatigue_timing = 1.0 + profile.fatigue_timing_multiplier * fatigue_state.level
        variability = 1.0 + profile.variability_growth * fatigue_state.level
        variability *= self.phase_variability_multiplier(phase)

        dwell_map = {
            "word_prediction": profile.dwell_time_word_mean,
            "visible_letter": profile.dwell_time_visible_mean,
            "hidden_letter": profile.dwell_time_hidden_mean,
            "space": profile.dwell_time_space_mean,
            "delete": profile.dwell_time_delete_mean,
            "punctuation": profile.dwell_time_punctuation_mean,
            "hesitation": 0.40,
        }
        dwell_mean = dwell_map.get(action_type, 0.80) * fatigue_timing
        reaction_mean = profile.reaction_time_base_mean * progress_factor * fatigue_timing

        if action_type == "word_prediction":
            reaction_mean *= 0.90
        elif action_type == "hidden_letter":
            reaction_mean *= 1.10
        elif action_type == "delete":
            reaction_mean *= 0.80
        elif action_type == "hesitation":
            reaction_mean *= 0.60

        dwell = self._positive_gaussian(dwell_mean, max(0.05, dwell_mean * 0.12 * variability))
        reaction = self._positive_gaussian(reaction_mean, max(0.05, profile.reaction_time_std * variability))
        return reaction + dwell

    def effective_accuracy(self, profile: SyntheticUserProfile, fatigue_state: FatigueState, hidden: bool) -> float:
        if hidden:
            return clamp(profile.hidden_letter_accuracy - profile.fatigue_accuracy_drop_hidden * fatigue_state.level, 0.30, 0.98)
        return clamp(profile.visible_letter_accuracy - profile.fatigue_accuracy_drop_visible * fatigue_state.level, 0.40, 0.99)

    def hesitation_probability(self, profile: SyntheticUserProfile, fatigue_state: FatigueState) -> float:
        return clamp(profile.hesitation_base_prob + profile.hesitation_fatigue_boost * fatigue_state.level, 0.0, 0.60)

    def _random_wrong_letter(self, visible_letters: List[str], correct_char: str) -> str:
        candidates = [c for c in visible_letters if c != correct_char]
        if not candidates:
            candidates = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != correct_char]
        return self.rng.choice(candidates)

    def clicks_for_word(
        self,
        target_word: str,
        context_words: List[str],
        profile: SyntheticUserProfile,
        fatigue_state: FatigueState,
        weeks_progress: int,
    ) -> Dict[str, Any]:
        prefix = ""
        total_clicks = 0
        total_time = 0.0
        actions = []
        finished_by_prediction = False
        error_count = 0
        fatigue_trace = []

        while prefix != target_word:
            predicted_words = self.recommend_words(context_words, profile, prefix)

            # Optional hesitation before action becomes more likely with fatigue.
            if self.rng.random() < self.hesitation_probability(profile, fatigue_state):
                hesitation_time = self.sample_action_time("hesitation", profile, fatigue_state, weeks_progress)
                actions.append({
                    "event_index": fatigue_state.event_index + 1,
                    "action": "hesitation",
                    "prefix_before_action": prefix,
                    "cost_clicks": 0,
                    "cost_time_seconds": round(hesitation_time, 3),
                    "fatigue_before": round(fatigue_state.level, 3),
                    "phase_before": self.get_phase(fatigue_state, profile),
                })
                total_time += hesitation_time
                self.update_fatigue_state(fatigue_state, profile, "hesitation", easy_action=False)
                fatigue_trace.append(round(fatigue_state.level, 3))

            if target_word in predicted_words and self.rng.random() < profile.recommendation_accept_prob:
                action_time = self.sample_action_time("word_prediction", profile, fatigue_state, weeks_progress)
                actions.append({
                    "event_index": fatigue_state.event_index + 1,
                    "action": "select_word_prediction",
                    "prefix_before_action": prefix,
                    "target": target_word,
                    "cost_clicks": 1,
                    "cost_time_seconds": round(action_time, 3),
                    "predicted_words": predicted_words,
                    "success": True,
                    "fatigue_before": round(fatigue_state.level, 3),
                    "phase_before": self.get_phase(fatigue_state, profile),
                })
                total_clicks += 1
                total_time += action_time
                finished_by_prediction = True
                prefix = target_word
                self.update_fatigue_state(fatigue_state, profile, "word_prediction", easy_action=True)
                fatigue_trace.append(round(fatigue_state.level, 3))
                break

            next_char = target_word[len(prefix)]
            visible = self.visible_letters(context_words, profile, prefix)
            hidden = next_char not in visible

            if not hidden:
                cost_clicks = 1
                action_type = "visible_letter"
                typo_prob = profile.typo_prob_visible
                candidate_wrong_letters = visible
            else:
                cost_clicks = self.fallback_letter_clicks
                action_type = "hidden_letter"
                typo_prob = profile.typo_prob_hidden
                candidate_wrong_letters = list("abcdefghijklmnopqrstuvwxyz")

            accuracy = self.effective_accuracy(profile, fatigue_state, hidden=hidden)
            error_prob = clamp(max(1.0 - accuracy, typo_prob), 0.0, 1.0)
            action_time = self.sample_action_time(action_type, profile, fatigue_state, weeks_progress)

            if self.rng.random() < error_prob:
                wrong_char = self._random_wrong_letter(candidate_wrong_letters, next_char)
                actions.append({
                    "event_index": fatigue_state.event_index + 1,
                    "action": f"select_{action_type}",
                    "prefix_before_action": prefix,
                    "target": wrong_char,
                    "intended_target": next_char,
                    "cost_clicks": cost_clicks,
                    "cost_time_seconds": round(action_time, 3),
                    "visible_letters": visible,
                    "success": False,
                    "error_type": "wrong_letter",
                    "fatigue_before": round(fatigue_state.level, 3),
                    "phase_before": self.get_phase(fatigue_state, profile),
                })
                total_clicks += cost_clicks
                total_time += action_time
                error_count += 1
                self.update_fatigue_state(fatigue_state, profile, action_type, was_error=True, easy_action=False)
                fatigue_trace.append(round(fatigue_state.level, 3))

                if self.rng.random() > profile.delete_after_error_prob:
                    hesitation_time = self.sample_action_time("hesitation", profile, fatigue_state, weeks_progress)
                    actions.append({
                        "event_index": fatigue_state.event_index + 1,
                        "action": "hesitation_before_delete",
                        "prefix_before_action": prefix,
                        "cost_clicks": 0,
                        "cost_time_seconds": round(hesitation_time, 3),
                        "success": True,
                        "fatigue_before": round(fatigue_state.level, 3),
                        "phase_before": self.get_phase(fatigue_state, profile),
                    })
                    total_time += hesitation_time
                    self.update_fatigue_state(fatigue_state, profile, "hesitation", easy_action=False)
                    fatigue_trace.append(round(fatigue_state.level, 3))

                delete_time = self.sample_action_time("delete", profile, fatigue_state, weeks_progress)
                actions.append({
                    "event_index": fatigue_state.event_index + 1,
                    "action": "delete",
                    "prefix_before_action": prefix,
                    "target": "DELETE",
                    "cost_clicks": self.delete_clicks,
                    "cost_time_seconds": round(delete_time, 3),
                    "success": True,
                    "fatigue_before": round(fatigue_state.level, 3),
                    "phase_before": self.get_phase(fatigue_state, profile),
                })
                total_clicks += self.delete_clicks
                total_time += delete_time
                self.update_fatigue_state(fatigue_state, profile, "delete", was_correction=True, easy_action=False)
                fatigue_trace.append(round(fatigue_state.level, 3))
                continue

            actions.append({
                "event_index": fatigue_state.event_index + 1,
                "action": f"select_{action_type}",
                "prefix_before_action": prefix,
                "target": next_char,
                "cost_clicks": cost_clicks,
                "cost_time_seconds": round(action_time, 3),
                "visible_letters": visible,
                "success": True,
                "fatigue_before": round(fatigue_state.level, 3),
                "phase_before": self.get_phase(fatigue_state, profile),
            })
            total_clicks += cost_clicks
            total_time += action_time
            prefix += next_char
            self.update_fatigue_state(fatigue_state, profile, action_type, easy_action=(action_type == "visible_letter"))
            fatigue_trace.append(round(fatigue_state.level, 3))

        return {
            "word": target_word,
            "clicks": total_clicks,
            "time_seconds": round(total_time, 3),
            "error_count": error_count,
            "finished_by_prediction": finished_by_prediction,
            "actions": actions,
            "fatigue_trace": fatigue_trace,
            "fatigue_level_after_word": round(fatigue_state.level, 3),
            "fatigue_phase_after_word": self.get_phase(fatigue_state, profile),
        }

    def clicks_for_sentence(
        self,
        sentence: str,
        profile: SyntheticUserProfile,
        weeks_progress: int,
        update_history: bool = True,
    ) -> Dict[str, Any]:
        tokens = self.tokenize(sentence)
        context_words = []
        total_clicks = 0
        total_time = 0.0
        total_errors = 0
        word_reports = []
        actions = []
        fatigue_state = self.init_fatigue_state(profile)
        sentence_fatigue_trace = []

        for i, tok in enumerate(tokens):
            if tok.isalpha():
                word_report = self.clicks_for_word(tok, context_words, profile, fatigue_state, weeks_progress)
                total_clicks += word_report["clicks"]
                total_time += word_report["time_seconds"]
                total_errors += word_report["error_count"]
                word_reports.append({
                    "word": word_report["word"],
                    "clicks": word_report["clicks"],
                    "time_seconds": word_report["time_seconds"],
                    "error_count": word_report["error_count"],
                    "finished_by_prediction": word_report["finished_by_prediction"],
                    "fatigue_level_after_word": word_report["fatigue_level_after_word"],
                    "fatigue_phase_after_word": word_report["fatigue_phase_after_word"],
                })
                actions.extend(word_report["actions"])
                sentence_fatigue_trace.extend(word_report["fatigue_trace"])
                context_words.append(tok)

                next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
                if next_tok is not None and next_tok.isalpha():
                    needs_space = not (word_report["finished_by_prediction"] and self.auto_space_after_word_prediction)
                    if needs_space:
                        space_time = self.sample_action_time("space", profile, fatigue_state, weeks_progress)
                        actions.append({
                            "event_index": fatigue_state.event_index + 1,
                            "action": "space",
                            "target": " ",
                            "cost_clicks": 1,
                            "cost_time_seconds": round(space_time, 3),
                            "success": True,
                            "fatigue_before": round(fatigue_state.level, 3),
                            "phase_before": self.get_phase(fatigue_state, profile),
                        })
                        total_clicks += 1
                        total_time += space_time
                        self.update_fatigue_state(fatigue_state, profile, "space", easy_action=True)
                        sentence_fatigue_trace.append(round(fatigue_state.level, 3))

            elif tok in {".", ",", "!", "?"}:
                punct_time = self.sample_action_time("punctuation", profile, fatigue_state, weeks_progress)
                actions.append({
                    "event_index": fatigue_state.event_index + 1,
                    "action": "punctuation",
                    "target": tok,
                    "cost_clicks": self.punctuation_clicks,
                    "cost_time_seconds": round(punct_time, 3),
                    "success": True,
                    "fatigue_before": round(fatigue_state.level, 3),
                    "phase_before": self.get_phase(fatigue_state, profile),
                })
                total_clicks += self.punctuation_clicks
                total_time += punct_time
                self.update_fatigue_state(fatigue_state, profile, "punctuation", easy_action=False)
                sentence_fatigue_trace.append(round(fatigue_state.level, 3))

        if update_history:
            self.update_user_history(sentence)

        subjective_fatigue = int(clamp(round(1 + 4 * fatigue_state.level), 1, 5))
        return {
            "sentence": sentence,
            "total_clicks": total_clicks,
            "total_time_seconds": round(total_time, 3),
            "total_errors": total_errors,
            "word_reports": word_reports,
            "actions": actions,
            "fatigue_final_level": round(fatigue_state.level, 3),
            "fatigue_final_phase": self.get_phase(fatigue_state, profile),
            "fatigue_trace": sentence_fatigue_trace,
            "subjective_fatigue_rating": subjective_fatigue,
        }


# =========================
# 4. Clinical and fatigue metrics
# =========================
def calculate_metrics(user_messages: List[str], weeks_progress: int) -> Dict[str, float]:
    all_text = " ".join(user_messages).lower()
    words = re.findall(r"\b\w+\b", all_text)
    sentences = max(len(re.split(r"[.!?]+", all_text)) - 1, 1)
    total_words = len(words)

    if total_words == 0:
        return {
            "lix_score": 0,
            "ttr_lexical_diversity": 0,
            "cwr_content_word_ratio": 0,
            "ciu_correct_info_units_rate": 0,
            "circumlocution_rate": 0,
            "avg_reaction_time_seconds": 0
        }

    unique_words = len(set(words))
    long_words = len([w for w in words if len(w) > 6])

    ttr = unique_words / total_words
    lix = (total_words / sentences) + ((long_words * 100) / total_words)

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "at",
        "by", "with", "and", "but", "or", "for", "of", "it", "this", "that",
        "i", "you", "he", "she", "we", "they", "me", "him", "her"
    }
    content_words = [w for w in words if w not in stopwords]
    cwr = len(content_words) / total_words

    ciu_efficiency = min(0.5 + (weeks_progress * 0.1), 0.95)
    ciu_rate = cwr * random.uniform(max(0.0, ciu_efficiency - 0.1), ciu_efficiency)

    circumlocution_markers = {"thing", "stuff", "like", "know", "mean", "something"}
    marker_count = len([w for w in words if w in circumlocution_markers])
    base_circ_rate = max(0.8 - (weeks_progress * 0.15), 0.1)
    circumlocution_rate = min(base_circ_rate + (marker_count / total_words), 1.0)
    simulated_reaction_time = random.uniform(25.0 - (weeks_progress * 2), 45.0 - (weeks_progress * 2))

    return {
        "lix_score": round(lix, 2),
        "ttr_lexical_diversity": round(ttr, 2),
        "cwr_content_word_ratio": round(cwr, 2),
        "ciu_correct_info_units_rate": round(ciu_rate, 2),
        "circumlocution_rate": round(circumlocution_rate, 2),
        "avg_reaction_time_seconds": round(simulated_reaction_time, 1)
    }


def estimate_subjective_fatigue_rating(
    total_time_seconds: float,
    total_errors: int,
    total_clicks: int,
    persona: PatientPersona,
) -> int:
    raw = 1.0
    raw += 0.10 * total_errors
    raw += 0.015 * total_clicks
    raw += 0.03 * max(total_time_seconds - 8.0, 0.0)
    raw *= persona.fatigue_sensitivity
    return int(clamp(round(raw), 1, 5))


# =========================
# 5. Simulation loop
# =========================
def run_session(
    session_id: str,
    context: str,
    active_topic: str,
    weeks_progress: int,
    click_simulator: GazetalkClickSimulator,
    user_profile: SyntheticUserProfile,
    persona: PatientPersona,
    history_keywords: List[str],
    start_message: Optional[str] = None,
    turns: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"Simulating Session [{session_id}] | Topic: {active_topic} | Week: {weeks_progress}")

    client = get_openai_client()
    if client is None:
        raise ImportError("The openai package is required to run sessions. Install it with pip install openai.")

    current_prompt = build_hdt_system_prompt(
        context=context,
        persona=persona,
        active_topic=active_topic,
        history_keywords=history_keywords,
        weeks_progress=weeks_progress,
    )
    gazetalk_prompt = build_gazetalk_system_prompt(persona, active_topic)

    history_hdt = [{"role": "system", "content": current_prompt}]
    history_gazetalk = [{"role": "system", "content": gazetalk_prompt}]

    first_message = start_message or make_topic_start_message(persona, active_topic, seed=seed)
    history_hdt.append({"role": "user", "content": first_message})

    user_utterances = []
    dialogue_log = [{"speaker": "GazeTalk", "text": first_message, "topic": active_topic}]

    total_session_clicks = 0
    total_session_time = 0.0
    total_session_errors = 0
    total_user_turns = 0
    session_fatigue_levels = []
    session_fatigue_ratings = []

    for turn_idx in range(turns):
        hdt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history_hdt,
            temperature=0.7
        ).choices[0].message.content.strip()

        click_report = click_simulator.clicks_for_sentence(
            sentence=hdt_response,
            profile=user_profile,
            weeks_progress=weeks_progress,
            update_history=True
        )

        total_session_clicks += click_report["total_clicks"]
        total_session_time += click_report["total_time_seconds"]
        total_session_errors += click_report["total_errors"]
        total_user_turns += 1
        session_fatigue_levels.append(click_report["fatigue_final_level"])
        session_fatigue_ratings.append(click_report["subjective_fatigue_rating"])

        user_utterances.append(hdt_response)
        dialogue_log.append({
            "speaker": "HDT_Patient",
            "text": hdt_response,
            "topic": active_topic,
            "turn_index": turn_idx + 1,
            "clicks": click_report["total_clicks"],
            "input_time_seconds": click_report["total_time_seconds"],
            "input_errors": click_report["total_errors"],
            "fatigue_final_level": click_report["fatigue_final_level"],
            "fatigue_final_phase": click_report["fatigue_final_phase"],
            "subjective_fatigue_rating": click_report["subjective_fatigue_rating"],
            "fatigue_trace": click_report["fatigue_trace"],
            "click_details": click_report["word_reports"],
            "action_log": click_report["actions"]
        })

        history_gazetalk.append({"role": "user", "content": hdt_response})
        history_hdt.append({"role": "assistant", "content": hdt_response})

        gazetalk_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history_gazetalk,
            temperature=0.5
        ).choices[0].message.content.strip()

        dialogue_log.append({"speaker": "GazeTalk", "text": gazetalk_response, "topic": active_topic, "turn_index": turn_idx + 1})
        history_hdt.append({"role": "user", "content": gazetalk_response})
        history_gazetalk.append({"role": "assistant", "content": gazetalk_response})

    dashboard_data = {
        "session_id": session_id,
        "context": context,
        "active_topic": active_topic,
        "training_week": weeks_progress,
        "persona": asdict(persona),
        "synthetic_user_profile": asdict(user_profile),
        "dashboard_metrics": calculate_metrics(user_utterances, weeks_progress),
        "dialogue_transcript": dialogue_log,
        "session_click_metrics": {
            "total_clicks": total_session_clicks,
            "avg_clicks_per_user_turn": round(total_session_clicks / max(total_user_turns, 1), 2),
            "total_input_time_seconds": round(total_session_time, 3),
            "avg_input_time_per_user_turn_seconds": round(total_session_time / max(total_user_turns, 1), 3),
            "total_input_errors": total_session_errors,
            "avg_input_errors_per_user_turn": round(total_session_errors / max(total_user_turns, 1), 2),
            "avg_final_fatigue_level": round(sum(session_fatigue_levels) / max(len(session_fatigue_levels), 1), 3),
            "max_final_fatigue_level": round(max(session_fatigue_levels) if session_fatigue_levels else 0.0, 3),
            "avg_subjective_fatigue_rating": round(sum(session_fatigue_ratings) / max(len(session_fatigue_ratings), 1), 2),
        }
    }
    return dashboard_data


# =========================
# 6. Helper for multi-patient export
# =========================
def build_patient_package(
    persona: PatientPersona,
    seed: int = 42,
    visible_letter_slots: int = 9,
    word_prediction_slots: int = 4,
    fallback_letter_clicks: int = 2,
) -> Dict[str, Any]:
    history_bundle = generate_daily_history(persona, days=7, seed=seed)
    persona_seed_corpus, _ = build_persona_seed_corpus(persona, n_phrases=60, seed=seed)

    click_simulator = GazetalkClickSimulator(
        base_corpus=DEFAULT_CORPUS + persona_seed_corpus + history_bundle["all_utterances"],
        visible_letter_slots=visible_letter_slots,
        word_prediction_slots=word_prediction_slots,
        fallback_letter_clicks=fallback_letter_clicks,
        auto_space_after_word_prediction=True,
        seed=seed,
    )
    click_simulator.preload_history(history_bundle["all_utterances"])

    base_profile = build_synthetic_user_profile(
        profile_id=persona.patient_id,
        severity=persona.aphasia_severity,
        seed=seed,
    )
    personalized_profile = personalize_user_profile(base_profile, persona)

    active_topics = history_bundle["top_topics"][:3] or weighted_topics(get_persona_topic_weights(persona), top_k=3)
    sessions = []
    for idx, topic in enumerate(active_topics, start=1):
        sessions.append(run_session(
            session_id=f"{persona.patient_id}_session_{idx:02d}",
            context=f"Topic-guided rehabilitation about {topic}.",
            active_topic=topic,
            weeks_progress=min(idx * 2 - 1, 5),
            click_simulator=click_simulator,
            user_profile=personalized_profile,
            persona=persona,
            history_keywords=history_bundle["top_words"],
            start_message=make_topic_start_message(persona, topic, seed=seed + idx),
            turns=3 if idx < 3 else 2,
            seed=seed + idx,
        ))

    return {
        "patient_id": persona.patient_id,
        "persona": asdict(persona),
        "history_summary": {
            "top_words": history_bundle["top_words"],
            "topic_distribution": history_bundle["topic_distribution"],
            "top_topics": active_topics,
        },
        "daily_history": history_bundle["daily_logs"],
        "sessions": sessions,
    }




# =========================
# 7. Package-level helpers
# =========================
def default_personas() -> List[PatientPersona]:
    """Return the built-in demo personas used in the original script."""
    return [
        build_patient_persona(
            patient_id="CN_F_58_001",
            display_name="Ms. Lin",
            country_or_culture="China",
            age_group="50+",
            gender="female",
            interests=["cooking", "family", "shopping", "weather"],
            daily_routines=["cook breakfast", "go market", "call daughter"],
            communication_goals=["daily needs", "family chat"],
            aphasia_severity="severe",
            system_familiarity=0.42,
            fatigue_sensitivity=1.12,
        ),
        build_patient_persona(
            patient_id="DK_M_29_001",
            display_name="Mr. Mads",
            country_or_culture="Denmark",
            age_group="20-35",
            gender="male",
            interests=["football", "friends", "gym", "coffee"],
            daily_routines=["bike commute", "gym", "coffee with friends"],
            communication_goals=["social chat", "daily planning"],
            aphasia_severity="moderate",
            system_familiarity=0.68,
            fatigue_sensitivity=0.92,
        ),
        build_patient_persona(
            patient_id="IT_M_71_001",
            display_name="Mr. Paolo",
            country_or_culture="Italy",
            age_group="65+",
            gender="male",
            interests=["garden", "family", "coffee", "weather"],
            daily_routines=["garden walk", "coffee", "family lunch"],
            communication_goals=["family visit", "health needs"],
            aphasia_severity="severe",
            system_familiarity=0.35,
            fatigue_sensitivity=1.18,
        ),
        build_patient_persona(
            patient_id="ID_F_41_001",
            display_name="Ms. Ayu",
            country_or_culture="Indonesia",
            age_group="35-55",
            gender="female",
            interests=["family", "religion", "cooking"],
            daily_routines=["family meal", "prayer", "market shopping"],
            communication_goals=["family chat", "daily needs"],
            aphasia_severity="moderate",
            system_familiarity=0.50,
            fatigue_sensitivity=1.05,
        ),
    ]


def personas_from_dicts(records: List[Dict[str, Any]]) -> List[PatientPersona]:
    """Build persona objects from JSON-like dictionaries."""
    personas: List[PatientPersona] = []
    for record in records:
        personas.append(build_patient_persona(
            patient_id=record["patient_id"],
            display_name=record.get("display_name", record["patient_id"]),
            country_or_culture=record["country_or_culture"],
            age_group=record["age_group"],
            gender=record["gender"],
            interests=record.get("interests", []),
            daily_routines=record.get("daily_routines", ["home", "rest"]),
            communication_goals=record.get("communication_goals", ["daily needs", "family chat"]),
            aphasia_severity=record.get("aphasia_severity", "severe"),
            primary_language=record.get("primary_language"),
            system_familiarity=record.get("system_familiarity", 0.45),
            fatigue_sensitivity=record.get("fatigue_sensitivity", 1.0),
        ))
    return personas


def load_personas_from_json(path: str | Path) -> List[PatientPersona]:
    """Load persona definitions from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("personas", payload)
    if not isinstance(records, list):
        raise ValueError("Expected a list of personas or a {'personas': [...]} object.")
    return personas_from_dicts(records)


def generate_project_export(
    personas: List[PatientPersona],
    seed_start: int = 100,
    visible_letter_slots: int = 9,
    word_prediction_slots: int = 4,
    fallback_letter_clicks: int = 2,
    project_name: str = "GazeTalk persona-conditioned synthetic user simulator",
) -> Dict[str, Any]:
    """Generate the full export payload for a list of personas."""
    return {
        "project": project_name,
        "patients": [
            build_patient_package(
                persona,
                seed=seed_start + idx,
                visible_letter_slots=visible_letter_slots,
                word_prediction_slots=word_prediction_slots,
                fallback_letter_clicks=fallback_letter_clicks,
            )
            for idx, persona in enumerate(personas)
        ],
    }


def save_project_export(payload: Dict[str, Any], output_path: str | Path) -> Path:
    """Write a generated payload to disk and return the written path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
