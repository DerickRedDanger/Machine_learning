import copy


def create_config(
    base_config,
    patches,
    raw_features,
    stage,
    feature_group,
    domain=None,
    notes=None,
):
    if not isinstance(base_config, dict):
        raise TypeError(
            "base_config must be a single experiment configuration dictionary."
        )

    if not stage:
        raise ValueError("Experiment needs a stage.")

    if not feature_group:
        raise ValueError("Experiment needs a feature_group.")

    if not patches:
        patches = []

    if isinstance(patches, dict):
        raise TypeError(
            "patches must be a list of patch dictionaries, "
            "even when only one patch is used."
        )

    config = copy.deepcopy(base_config)

    model_name = config.get("model_name")

    if not model_name:
        raise ValueError(
            "Base configuration must contain 'model_name'."
        )

    # ---------------------------------------------------------
    # State used while resolving transformations
    # ---------------------------------------------------------

    available_features = set(raw_features)

    used_transform_ids = set()

    owned_features = {}

    feature_pipeline = []

    # ---------------------------------------------------------
    # Apply patches in the order provided
    # ---------------------------------------------------------

    for patch in patches:

        if not isinstance(patch, dict):
            raise TypeError(
                "Each patch must be a dictionary."
            )

        # -----------------------------------------------------
        # Transformations
        # -----------------------------------------------------

        for transformation in patch.get(
            "transformations",
            [],
        ):
            transform_id = transformation["id"]

            # Exact same transformation requested twice:
            # use it only once.
            if transform_id in used_transform_ids:
                continue

            # -------------------------------------------------
            # Check required features
            # -------------------------------------------------

            required = set(
                transformation.get(
                    "requires",
                    [],
                )
            )

            missing_required = (
                required - available_features
            )

            if missing_required:
                raise ValueError(
                    f"Transformation '{transform_id}' "
                    "requires unavailable features: "
                    f"{sorted(missing_required)}"
                )

            # -------------------------------------------------
            # Check ownership conflicts
            # -------------------------------------------------

            for feature in transformation.get(
                "owns",
                [],
            ):
                if feature in owned_features:
                    previous_owner = (
                        owned_features[feature]
                    )

                    raise ValueError(
                        f"Transformation '{transform_id}' "
                        f"attempts to own '{feature}', "
                        f"but it is already owned by "
                        f"'{previous_owner}'."
                    )

            # -------------------------------------------------
            # Instantiate transformer
            # -------------------------------------------------

            transformer_factory = (
                transformation["transformer"]
            )

            transformer = (
                transformer_factory()
            )

            feature_pipeline.append(
                (
                    transform_id,
                    transformer,
                )
            )

            used_transform_ids.add(
                transform_id
            )

            # -------------------------------------------------
            # Update ownership and available features
            # -------------------------------------------------

            for feature in transformation.get(
                "owns",
                [],
            ):
                owned_features[feature] = (
                    transform_id
                )

            available_features.update(
                transformation.get(
                    "produces",
                    [],
                )
            )

        # -----------------------------------------------------
        # Apply normal configuration changes
        # -----------------------------------------------------

        _apply_add(
            config,
            patch.get("add", {}),
        )

        _apply_remove(
            config,
            patch.get("remove", {}),
        )

        _apply_update(
            config,
            patch.get("update", {}),
        )

    # ---------------------------------------------------------
    # Attach resolved feature pipeline
    # ---------------------------------------------------------

    config["feature_pipeline"] = (
        feature_pipeline
    )

    # Legacy FE must no longer survive into the new config.
    config.pop(
        "feature_engineering",
        None,
    )

    # ---------------------------------------------------------
    # Validate model input availability
    # ---------------------------------------------------------

    model_features = (
        features_from_preprocessing(
            config["preprocessing"]
        )
    )

    unavailable_model_features = (
        set(model_features)
        - available_features
    )

    if unavailable_model_features:
        raise ValueError(
            "Preprocessing requests unavailable "
            "features: "
            f"{sorted(unavailable_model_features)}"
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    config["stage"] = stage
    config["feature_group"] = feature_group
    config["group"] = (
        f"{stage}__{feature_group}"
    )

    config["name"] = (
        f"{stage}__{feature_group}__"
        f"{model_name}"
    )

    config["domain"] = (
        domain
        if domain is not None
        else feature_group
    )

    if notes is not None:
        config["notes"] = notes

    # Human-readable derived metadata.
    config["features"] = model_features

    return config

def create_config_group(
    base_configs,
    patches,
    raw_features,
    stage,
    feature_group,
    domain=None,
    notes=None,
):
    if not isinstance(base_configs, dict):
        raise TypeError(
            "base_configs must be a dictionary "
            "of experiment configurations."
        )

    if not base_configs:
        raise ValueError(
            "No base configurations provided."
        )

    configs = {}

    for base_key, base_config in (
        base_configs.items()
    ):
        config = create_config(
            base_config=base_config,
            patches=patches,
            raw_features=raw_features,
            stage=stage,
            feature_group=feature_group,
            domain=domain,
            notes=notes,
        )

        model_name = config["model_name"]

        config_key = (
            f"{model_name}__{feature_group}"
        )

        if config_key in configs:
            raise ValueError(
                f"Duplicate config key generated: "
                f"'{config_key}'."
            )

        configs[config_key] = config

    validate_config_group(configs)

    return configs

# Append Helper
def _append_unique(target_list, values):
    for value in values:
        if value not in target_list:
            target_list.append(value)

# Add Helper
def _apply_add(config, additions):
    for section, changes in additions.items():

        if section not in config:
            config[section] = {}

        for key, value in changes.items():

            if isinstance(value, list):
                current = config[section].setdefault(
                    key,
                    []
                )

                _append_unique(
                    current,
                    value,
                )

            else:
                if key not in config[section]:
                    config[section][key] = value
# Remove Helper
def _apply_remove(config, removals):
    for section, changes in removals.items():

        if section not in config:
            continue

        for key, values in changes.items():

            if key not in config[section]:
                continue

            if not isinstance(
                config[section][key],
                list,
            ):
                raise ValueError(
                    f"Cannot remove list values "
                    f"from non-list config field "
                    f"'{section}.{key}'."
                )

            config[section][key] = [
                item
                for item
                in config[section][key]
                if item not in values
            ]

# Update Helper
def _apply_update(config, updates):
    for section, changes in updates.items():

        if section not in config:
            config[section] = {}

        for key, value in changes.items():
            config[section][key] = value

def features_from_preprocessing(preprocessing):
    feature_groups = [
        "numeric_features",
        "onehot_features",
        "ordinal_features",
    ]

    features = []
    seen = set()

    for group in feature_groups:
        for feature in preprocessing.get(group, []):
            if feature in seen:
                raise ValueError(
                    f"Feature '{feature}' appears in more than one "
                    "preprocessing group."
                )

            seen.add(feature)
            features.append(feature)

    return features

def validate_config_group(configs):
    for key, config in configs.items():

        model_name = config.get(
            "model_name"
        )

        if not model_name:
            raise ValueError(
                f"Config '{key}' has no "
                "'model_name'."
            )

        if (
            model_name.lower()
            not in key.lower()
        ):
            raise ValueError(
                f"Config key '{key}' must "
                f"contain model_name "
                f"'{model_name}'."
            )

    return True