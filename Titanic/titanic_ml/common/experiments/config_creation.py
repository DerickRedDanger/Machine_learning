import copy
import warnings


VALID_PATCH_KEYS = {
    "transformations",
    "add",
    "remove",
    "update",
}

VALID_PREPROCESSING_KEYS = {
    "numeric_features",
    "onehot_features",
    "ordinal_features",
    "numeric_imputer",
    "categorical_imputer",
    "scaler",
}

PREPROCESSING_FEATURE_KEYS = {
    "numeric_features",
    "onehot_features",
    "ordinal_features",
}

VALID_PRE_CV_SCOPE_KEYS = {
    "full_prediction_context",
}

def _validate_patch_structure(patch):
    unknown_patch_keys = (
        set(patch)
        - VALID_PATCH_KEYS
    )

    if unknown_patch_keys:
        raise ValueError(
            "Unknown patch keys: "
            f"{sorted(unknown_patch_keys)}"
        )

    for operation_name in [
        "add",
        "remove",
        "update",
    ]:
        operation = patch.get(
            operation_name,
            {},
        )

        if not isinstance(operation, dict):
            raise TypeError(
                f"Patch '{operation_name}' "
                "must be a dictionary."
            )

        preprocessing_changes = (
            operation.get("preprocessing")
        )

        if preprocessing_changes is None:
            continue

        if not isinstance(
            preprocessing_changes,
            dict,
        ):
            raise TypeError(
                f"Patch '{operation_name}.preprocessing' "
                "must be a dictionary."
            )

        unknown_keys = (
            set(preprocessing_changes)
            - VALID_PREPROCESSING_KEYS
        )

        if unknown_keys:
            raise ValueError(
                "Unknown preprocessing keys in "
                f"'{operation_name}': "
                f"{sorted(unknown_keys)}"
            )

        pre_cv_scope = patch.get("pre_cv_scope", None)
        if pre_cv_scope is not None and pre_cv_scope not in VALID_PRE_CV_SCOPE_KEYS:
            raise ValueError(
                f"Invalid pre_cv_scope '{pre_cv_scope}' in patch. "
                f"Valid values are: {VALID_PRE_CV_SCOPE_KEYS}"
            )


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

    produced_features = set()

    required_features = set()

    # ---------------------------------------------------------
    # Apply patches in the order provided
    # ---------------------------------------------------------

    for patch in patches:

        if not isinstance(patch, dict):
            raise TypeError(
                "Each patch must be a dictionary."
            )

        # -----------------------------------------------------
        # Validation check
        # -----------------------------------------------------
        _validate_patch_structure(patch)


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

            required_features.update(required)

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

            produced = set(
                transformation.get(
                    "produces",
                    [],
                )
            )

            produced_features.update(produced)
            available_features.update(produced)

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

    unused_produced_features = (
        produced_features
        - required_features
        - set(model_features)
    )

    if unused_produced_features:
        warnings.warn(
            "The following engineered features are produced "
            "but are neither consumed by another transformation "
            "nor included in preprocessing: "
            f"{sorted(unused_produced_features)}",
            UserWarning,
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
def _append_unique(
    target_list,
    values,
    field_name=None,
):
    for value in values:
        if value in target_list:
            warnings.warn(
                f"'{value}' is already present"
                + (
                    f" in '{field_name}'."
                    if field_name
                    else "."
                ),
                UserWarning,
            )
            continue

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
                    field_name=f"{section}.{key}",
                )

            else:
                if key not in config[section]:
                    config[section][key] = value
# Remove Helper
def _apply_remove(config, removals):
    for section, changes in removals.items():

        if section not in config:
            raise ValueError(
                f"Cannot remove from unknown config "
                f"section '{section}'."
            )

        for key, values in changes.items():

            if key not in config[section]:
                raise ValueError(
                    f"Cannot remove from unknown config field "
                    f"'{section}.{key}'."
                )

            current = config[section][key]

            if not isinstance(current, list):
                raise ValueError(
                    f"Cannot remove list values "
                    f"from non-list config field "
                    f"'{section}.{key}'."
                )

            missing_values = [
                value
                for value in values
                if value not in current
            ]

            if missing_values:
                raise ValueError(
                    f"Cannot remove {missing_values} from "
                    f"'{section}.{key}' because they are "
                    "not present there."
                )

            config[section][key] = [
                item
                for item in current
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