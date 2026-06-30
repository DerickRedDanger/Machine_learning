def feature_effect_interpretation(feature_effect):
    if not feature_effect:
        return []

    effect = feature_effect[0]
    recommendation =[]

    if effect["recommended_for_all"]:
        recommendation=[
                f"- Recommended for all models",
              f"- Mean delta: {effect['mean_delta']}"
            ]
    elif effect["discard"]:
        recommendation=[
            f"- Discard",
            f"- Max delta: {effect['max_delta']}"
            ]
    else:
        rec_models = []
        print(effect['recommended_models'])
        for name, value in effect['recommended_models'].items():
            rec_models.append(f"- {name}: {value}")

        recommendation=[
                f"- Recommended for specific models:",
                *rec_models
            ]
        
    section = [
        "",
        "#### Interpretation",
        "",
        f"- Verdict: {effect['verdict']}",
        *recommendation,
        "",
    ]

    return section