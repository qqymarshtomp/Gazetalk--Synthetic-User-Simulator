# GazeTalk Simulator

A persona-conditioned synthetic patient simulator for a GazeTalk-style eye-tracking interface.


## What it does

The simulator combines:

1. **patient personas** (culture, age, gender, interests, routines, communication goals)
2. **topic and daily-history generation**
3. **prompt-based patient and assistant dialogue generation**
4. **click-level interaction simulation**
5. **fatigue-aware timing, error, and correction modeling**

## Repository layout

```text
gazetalk-simulator/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── examples/
│   └── personas.json
├── outputs/
├── src/
│   └── gazetalk_simulator/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       └── core.py
└── tests/
    └── test_smoke.py
```

## Installation

```bash
pip install -e .
```

For generation that calls the OpenAI API, set:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Quick start

Run with the built-in example personas:

```bash
gazetalk-sim --output outputs/persona_conditioned_dashboard_data.json
```

Run with a custom persona file:

```bash
gazetalk-sim --personas-file examples/personas.json --output outputs/custom_run.json
```

## Persona file format

You can provide either a raw list or an object with a `personas` field.

Example:

```json
{
  "personas": [
    {
      "patient_id": "CN_F_58_001",
      "display_name": "Ms. Lin",
      "country_or_culture": "China",
      "age_group": "50+",
      "gender": "female",
      "interests": ["cooking", "family", "shopping", "weather"],
      "daily_routines": ["cook breakfast", "go market", "call daughter"],
      "communication_goals": ["daily needs", "family chat"],
      "aphasia_severity": "severe",
      "system_familiarity": 0.42,
      "fatigue_sensitivity": 1.12
    }
  ]
}
```

## Notes

- The click and fatigue models are simulation-based, not direct measurements from real interface logs.
- OpenAI is required for end-to-end multi-turn session generation.
- Many helper functions can still be imported and tested without making API calls.

## Development

Run the smoke test:

```bash
pytest -q
```
