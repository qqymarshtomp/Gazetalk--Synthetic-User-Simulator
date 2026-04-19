from __future__ import annotations

import argparse
from pathlib import Path

from .core import default_personas, generate_project_export, load_personas_from_json, save_project_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazetalk-sim",
        description="Generate persona-conditioned synthetic patient data for a GazeTalk-style simulator.",
    )
    parser.add_argument(
        "--personas-file",
        type=Path,
        help="Path to a JSON file containing persona definitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/persona_conditioned_dashboard_data.json"),
        help="Where to save the generated JSON output.",
    )
    parser.add_argument("--seed-start", type=int, default=100, help="Base seed used to generate per-persona seeds.")
    parser.add_argument("--visible-letter-slots", type=int, default=9, help="Number of visible letter slots in the simulated interface.")
    parser.add_argument("--word-prediction-slots", type=int, default=4, help="Number of word prediction slots.")
    parser.add_argument("--fallback-letter-clicks", type=int, default=2, help="Cost of selecting a hidden letter path.")
    parser.add_argument("--project-name", default="GazeTalk persona-conditioned synthetic user simulator", help="Project name written into the export payload.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    personas = load_personas_from_json(args.personas_file) if args.personas_file else default_personas()
    payload = generate_project_export(
        personas=personas,
        seed_start=args.seed_start,
        visible_letter_slots=args.visible_letter_slots,
        word_prediction_slots=args.word_prediction_slots,
        fallback_letter_clicks=args.fallback_letter_clicks,
        project_name=args.project_name,
    )
    out = save_project_export(payload, args.output)
    print(f"Saved output to {out}")


if __name__ == "__main__":
    main()
