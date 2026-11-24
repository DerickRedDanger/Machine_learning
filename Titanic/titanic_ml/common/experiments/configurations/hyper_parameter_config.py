import copy
from titanic_ml.common.experiments.config_creation import config_to_group, update_experiment_identity
from titanic_ml.common.experiments.configurations.feature_selection_config import ALL_FS_CONFIGS
import numpy as np


ALL_HPO_CONFIGS = {}
ALL_THRESHOLD_CONFIGS = {}


# Log Reg hyperparameter optimization configurations

best_logreg_fs = ALL_FS_CONFIGS["pr03__removing_fare__logreg"]

logreg_selected_features = copy.deepcopy(
    next(iter(best_logreg_fs.values()))
    )


hpo01_logreg = logreg_selected_features

hpo01_logreg = update_experiment_identity(
    config=hpo01_logreg,
    stage="hpo01",
    feature_group="regularization"
)

hpo01_logreg.update({
    "hyperparameter_search": {
        "method": "grid",
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        "refit": "accuracy",
    },
})

hpo01_configs__logreg = config_to_group(hpo01_logreg)

ALL_HPO_CONFIGS['hpo01_configs__logreg'] = hpo01_configs__logreg

# ------------------------------------------

hpo02_logreg = hpo01_logreg

hpo02_logreg = update_experiment_identity(
    config=hpo02_logreg,
    stage="hpo02",
    feature_group="refine_c"
)

hpo02_logreg["hyperparameter_search"]["param_grid"] = {
    "model__C": [
        0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
        1.5, 2.0, 3.0, 5.0,
    ],
}

hpo02_configs__logreg = config_to_group(hpo02_logreg)
ALL_HPO_CONFIGS['hpo02_configs__logreg'] = hpo02_configs__logreg

#--------------------------------------------

hpo03_logreg = update_experiment_identity(
    config=hpo02_logreg,
    stage="hpo03",
    feature_group="solver_penalty",
)

c_values = [0.4, 0.6, 0.8, 1.0, 1.5]

hpo03_logreg["hyperparameter_search"]["param_grid"] = [
    {
        "model__solver": [
            "lbfgs",
            "newton-cg",
            "sag",
            "saga",
            "liblinear",
        ],
        "model__penalty": ["l2"],
        "model__C": c_values,
    },
    {
        "model__solver": [
            "liblinear",
            "saga",
        ],
        "model__penalty": ["l1"],
        "model__C": c_values,
    },
    {
        "model__solver": ["saga"],
        "model__penalty": ["elasticnet"],
        "model__l1_ratio": [0.25, 0.5, 0.75],
        "model__C": c_values,
    },
]

ALL_HPO_CONFIGS["hpo03_configs__logreg"] = config_to_group(
    hpo03_logreg
)

# #--------------------------------------------

# hpo03_logreg = hpo02_logreg

# hpo03_logreg = update_experiment_identity(
#     config=hpo03_logreg,
#     stage="hpo03",
#     feature_group="regularization_type"
# )

# hpo03_logreg["hyperparameter_search"]["param_grid"] = [
#     {
#         "model__solver": ["saga"],
#         "model__penalty": ["l1", "l2"],
#         "model__C": [0.4, 0.6, 0.8, 1.0, 1.5],
#     },
#     {
#         "model__solver": ["saga"],
#         "model__penalty": ["elasticnet"],
#         "model__l1_ratio": [0.25, 0.5, 0.75],
#         "model__C": [0.4, 0.6, 0.8, 1.0, 1.5],
#     },
# ]

# hpo03_configs__logreg = config_to_group(hpo03_logreg)
# ALL_HPO_CONFIGS['hpo03_configs__logreg'] = hpo03_configs__logreg

# #--------------------------------------------

# hpo04_logreg = hpo03_logreg

# hpo04_logreg = update_experiment_identity(
#     config=hpo04_logreg,
#     stage="hpo04",
#     feature_group="solver_comparison"
# )

# hpo04_logreg["hyperparameter_search"]["param_grid"] = {
#     "model__solver": [
#         "lbfgs",
#         "liblinear",
#         "newton-cg",
#         "sag",
#         "saga",
#     ],
#     "model__C": [0.6, 0.8, 1.0],
#     "model__penalty": ["l2"],
# }

# hpo04_configs__logreg = config_to_group(hpo04_logreg)

# ALL_HPO_CONFIGS['hpo04_configs__logreg'] = hpo04_configs__logreg

# #--------------------------------------------

hpo04_logreg = hpo03_logreg

hpo04_logreg = update_experiment_identity(
    config=hpo04_logreg,
    stage="hpo05",
    feature_group="class_weight"
)

hpo04_logreg["hyperparameter_search"]["param_grid"] = {
    "model__solver": ["lbfgs"],
    "model__penalty": ["l2"],
    "model__C": [0.6, 0.8, 1.0],
    "model__class_weight": [
        {0: 2.0, 1: 1.0},
        {0: 1.5, 1: 1.0},
        {0: 1.25, 1: 1.0},
        None,
        "balanced",
        {0: 1, 1: 1.25},
        {0: 1, 1: 1.5},
        {0: 1, 1: 2.0},
    ],
}

hpo04_configs__logreg = config_to_group(hpo04_logreg)

ALL_HPO_CONFIGS['hpo04_configs__logreg'] = hpo04_configs__logreg

#--------------------------------------------

hpo_final_logreg = hpo04_logreg

hpo_final_logreg = update_experiment_identity(
    config=hpo_final_logreg,
    stage="hpo_final",
    feature_group="selected"
)

hpo_final_logreg["model_params"].update({
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,
})

hpo_final_logreg.pop("hyperparameter_search", None)

hpo_final_logreg["threshold_experiment"] = "th01_logreg"

hpo_final_configs__logreg = config_to_group(hpo_final_logreg)
ALL_HPO_CONFIGS['hpo_final_configs__logreg'] = hpo_final_configs__logreg

th01_logreg = {
        "name": "th01_logreg",
        "source_experiment": hpo_final_logreg['name'],
        "thresholds":np.round(
    np.arange(0.30, 0.701, 0.025),
    3,),
        "score_method": "predict_proba",
        "default_threshold": 0.5,
        "outer_cv": 5,
        "inner_cv": 5,
    }

ALL_THRESHOLD_CONFIGS["th01_logreg"] = th01_logreg

# ------------------------------------------------

class_weights = hpo04_logreg[
    "hyperparameter_search"
]["param_grid"]["model__class_weight"]

th02_model_configs = {}
th02_threshold_configs = {}

for i, class_weight in enumerate(class_weights, start=1):

    threshold_name = f"th02_{i:02d}_logreg"

    model_config = update_experiment_identity(
        config=hpo_final_logreg,
        stage="th02",
        feature_group=f"class_weight{i:02d}",
    )

    model_config["model_params"]["class_weight"] = (
        copy.deepcopy(class_weight)
    )

    model_config["threshold_experiment"] = threshold_name

    threshold_config = copy.deepcopy(th01_logreg)

    threshold_config.update({
        "name": threshold_name,
        "source_experiment": model_config["name"],
        "thresholds": np.round(
            np.arange(0.20, 0.801, 0.025), 3
        ).tolist(),
    })

    th02_model_configs[model_config["name"]] = model_config
    th02_threshold_configs[threshold_name] = threshold_config


ALL_HPO_CONFIGS["th02_configs__logreg"] = th02_model_configs

ALL_THRESHOLD_CONFIGS.update(th02_threshold_configs)

# SVC hyperparameter optimization configurations


best_svc_fs = ALL_FS_CONFIGS['fs04__fare_fe10_full_context__svc']

best_svc_fs = copy.deepcopy(
    next(iter(best_svc_fs.values()))
)

hpo01_scv = best_svc_fs

hpo01_scv = update_experiment_identity(
    config=hpo01_scv,
    stage="hpo01",
    feature_group="regularization"
)

hpo01_scv["hyperparameter_search"] = {
    "method": "grid",
    "param_grid": {
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    "refit": "accuracy",
}

hpo01_scv_configs = config_to_group(hpo01_scv)

ALL_HPO_CONFIGS["hpo01_configs__svc"] = hpo01_scv_configs

#--------------------------------------------

hpo02_scv = hpo01_scv

hpo02_scv = update_experiment_identity(
    config=hpo02_scv,
    stage="hpo02",
    feature_group="refined_c"
)

hpo02_scv["hyperparameter_search"]["param_grid"] = {
    "model__C": [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    ],
}

hpo02_scv_configs = config_to_group(hpo02_scv)

ALL_HPO_CONFIGS["hpo02_configs__svc"] = hpo02_scv_configs

#--------------------------------------------

hpo03_scv = hpo02_scv

hpo03_scv = update_experiment_identity(
    config=hpo03_scv,
    stage="hpo03",
    feature_group="c_gamma"
)

hpo03_scv["hyperparameter_search"]["param_grid"] = {
    "model__C": [0.6, 1.0, 1.5, 2.0],
    "model__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
}

hpo03_scv_configs = config_to_group(hpo03_scv)

ALL_HPO_CONFIGS["hpo03_configs__svc"] = hpo03_scv_configs

#--------------------------------------------

hpo04_scv = hpo03_scv

hpo04_scv = update_experiment_identity(
    config=hpo04_scv,
    stage="hpo04",
    feature_group="c_gamma"
)

hpo04_scv["hyperparameter_search"]["param_grid"] = {
    "model__C": [0.6, 1.0, 1.5, 2.0],
    "model__gamma": [
        "scale",
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
    ],
}

hpo04_scv_configs = config_to_group(hpo04_scv)

ALL_HPO_CONFIGS["hpo04_configs__svc"] = hpo04_scv_configs

#--------------------------------------------
hpo05_scv = hpo04_scv

hpo05_scv = update_experiment_identity(
    config=hpo05_scv,
    stage="hpo05",
    feature_group="kernel_comparison"
)

hpo05_scv["hyperparameter_search"]["param_grid"] = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    {
        "model__kernel": ["rbf"],
        "model__C": [0.6, 1, 1.5, 2],
        "model__gamma": ["scale", 0.05, 0.1],
    },
    {
        "model__kernel": ["poly"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale"],
        "model__degree": [2, 3, 4],
        "model__coef0": [0, 1, 2],
    },
    {
        "model__kernel": ["sigmoid"],
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale"],
        "model__coef0": [0, 1, 2],
    },
]

hpo05_scv_configs = config_to_group(hpo05_scv)

ALL_HPO_CONFIGS["hpo05_configs__svc"] = hpo05_scv_configs


hpo06_scv = hpo05_scv

hpo06_scv = update_experiment_identity(
    config=hpo06_scv,
    stage="hpo06",
    feature_group="class_weight"
)

hpo06_scv["hyperparameter_search"]["param_grid"] = {
    "model__kernel": ["rbf"],
    "model__gamma": ["scale"],
    "model__C": [0.6, 1.0, 1.5],
    "model__class_weight": [
        None,
        "balanced",
        {0: 1, 1: 1.25},
        {0: 1, 1: 1.5},
        {0: 1, 1: 2},
        {0: 1.25, 1: 1},
        {0: 1.5, 1: 1},
        {0: 2, 1: 1},
    ],
}

hpo06_scv_configs = config_to_group(hpo06_scv)

ALL_HPO_CONFIGS["hpo06_configs__svc"] = hpo06_scv_configs

#--------------------------------------------

hpo_final_svc = update_experiment_identity(
    best_svc_fs,
    stage="hpo_final",
    feature_group="selected",
)

hpo_final_svc.pop("hyperparameter_search", None)

hpo_final_svc["threshold_experiment"] = "th01_scv"

hpo_final_svc_configs= config_to_group(hpo_final_svc)

ALL_HPO_CONFIGS["hpo_final_configs__svc"] = hpo_final_svc_configs


th01_scv = {
        "name": "th01_scv",
        "source_experiment": hpo_final_svc['name'],
        "thresholds":np.round(
    np.arange(-1.0, 1.01, 0.1),
    2,),
        "score_method": "decision_function",
        "default_threshold": 0.0,
        "outer_cv": 5,
        "inner_cv": 5,
    }

ALL_THRESHOLD_CONFIGS["th01_scv"] = th01_scv

