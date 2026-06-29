def feature_effect_interpretation(feature_effect):
    if not feature_effect:
        return []

    effect = feature_effect[0]

    section = [
        "",
        "#### Interpretation",
        "",
        f"- Verdict: {effect['verdict']}",
    ]

    if effect["recommended_for_all"]:
        section.append(
            f"- Recommended for all models: {effect['recommended_for_all']}"
        )
    elif effect["discard"]:
        section.append(f"- Discard: {effect['discard']}")
    else:
        section.append(
            f"- Recommended for the models: {effect['recommended_models']}"
        )

    section.append(f"- Max delta: {effect['max_delta']}")

    return section