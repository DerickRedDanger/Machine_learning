def feature_effect_interpretation(feature_effect):
    if not feature_effect:
        return []

    effect = feature_effect[0]

    if effect["recommended_for_all"]:
        recommendation = [
            "- Recommended for all models",
            f"- Mean delta: {effect['mean_delta']}",
        ]

    elif effect["discard"]:
        recommendation = [
            "- Discard",
            f"- Max delta: {effect['max_delta']}",
        ]

    else:
        rec_models = [
            f"  - {name}: {value}"
            for name, value in effect["recommended_model_deltas"].items()
        ]

        recommendation = [
            "- Recommended for specific models:",
            *rec_models,
        ]

    return [
        "",
        "#### Interpretation",
        "",
        f"- Verdict: {effect['verdict']}",
        *recommendation,
        "",
    ]