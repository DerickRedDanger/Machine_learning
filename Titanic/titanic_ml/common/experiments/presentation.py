
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
        section.extend([
            "- Recommended for all models",
            f"- Mean delta: {effect['mean_delta']}",
        ])

    elif effect["discard"]:
        section.extend([
            "- Discard",
            f"- Max delta: {effect['max_delta']}",
        ])

    else:
        section.append("- Recommended for specific models:")

        for model_name, tradeoff in effect["recommended_model_tradeoffs"].items():
            section.append(f"  - {model_name}: {tradeoff['primary_metric']}: {tradeoff['primary_delta']}")

            if tradeoff["secondary_pros"]:
                section.append("    - Secondary gains:")
                for metric, delta in tradeoff["secondary_pros"].items():
                    section.append(f"      - {metric}: {delta}")

            if tradeoff["secondary_cons"]:
                section.append("    - Secondary losses:")
                for metric, delta in tradeoff["secondary_cons"].items():
                    section.append(f"      - {metric}: {delta}")

        if effect["notable_secondary_improvements"]:
            section.append("- Notable secondary improvements in non-recommended models:")

            for model_name, metrics in effect["notable_secondary_improvements"].items():
                metric_text = ", ".join(
                    f"{metric}: {delta}"
                    for metric, delta in metrics.items()
                )
                section.append(f"  - {model_name}: {metric_text}")

    section.append("")
    return section