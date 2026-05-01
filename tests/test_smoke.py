from gazetalk_simulator.core import (
    SuggestionProvider,
    GazetalkClickSimulator,
    build_patient_persona,
    build_synthetic_user_profile,
    personalize_user_profile,
    generate_daily_history,
)


def test_non_api_smoke():
    persona = build_patient_persona(
        patient_id="TEST_001",
        display_name="Test Persona",
        country_or_culture="China",
        age_group="50+",
        gender="female",
        interests=["cooking", "family"],
        daily_routines=["go market"],
        communication_goals=["daily needs"],
        aphasia_severity="severe",
    )
    history = generate_daily_history(persona, days=2, seed=1)
    base_profile = build_synthetic_user_profile("TEST_001", severity="severe", seed=1)
    profile = personalize_user_profile(base_profile, persona)

    sim = GazetalkClickSimulator(base_corpus=history["all_utterances"], seed=1)
    sim.preload_history(history["all_utterances"])
    report = sim.clicks_for_sentence("need water", profile=profile, weeks_progress=2, update_history=False)

    assert report["total_clicks"] >= 1
    assert report["total_time_seconds"] > 0
    assert "word_reports" in report


class FakeSuggestionProvider(SuggestionProvider):
    def get_word_suggestions(self, prompt: str, locale: str = "en_US", limit: int = 4):
        return ["water", "want", "work"][:limit]

    def get_letter_suggestions(self, prompt: str, locale: str = "en_US", limit: int = 9):
        return ["w", "a", "t", "e", "r"][:limit]


def test_external_suggestion_provider_smoke():
    persona = build_patient_persona(
        patient_id="TEST_002",
        display_name="Test Persona 2",
        country_or_culture="Generic",
        age_group="35-55",
        gender="female",
        interests=["health"],
        daily_routines=["home"],
        communication_goals=["daily needs"],
        aphasia_severity="moderate",
    )
    profile = personalize_user_profile(
        build_synthetic_user_profile("TEST_002", severity="moderate", seed=2),
        persona,
    )
    sim = GazetalkClickSimulator(
        base_corpus=["need water"],
        seed=2,
        suggestion_provider=FakeSuggestionProvider(),
    )

    assert sim.recommend_words([], profile, "")[0] == "water"
    assert "w" in sim.visible_letters([], profile, "")
    report = sim.clicks_for_sentence("water", profile=profile, weeks_progress=1, update_history=False)
    assert report["total_clicks"] >= 1
