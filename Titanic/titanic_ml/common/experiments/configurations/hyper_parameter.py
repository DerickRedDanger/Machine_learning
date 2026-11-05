import copy
from titanic_ml.common.experiments.config_creation import config_to_group
from titanic_ml.common.experiments.configurations.feature_selection_config import ALL_FS_CONFIGS

ALL_HPO_CONFIGS = {}

# Log Reg hyperparameter optimization configurations

best_logreg_fs = ALL_FS_CONFIGS["pr03__removing_fare__logreg"]

logreg_selected_features = copy.deepcopy(
    next(iter(best_logreg_fs.values()))
    )



logreg_hpo01 = copy.deepcopy(logreg_selected_features)

logreg_hpo01.update({
    "name": "hpo01__regularization__logreg",
    "stage": "hpo01",
    "feature_group": "regularization",
    "group": "hpo01__regularization",
    "source_experiment": logreg_selected_features["name"],
    "hyperparameter_search": {
        "method": "grid",
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        "refit": "accuracy",
    },
})

hpo01_configs = config_to_group(logreg_hpo01)

ALL_HPO_CONFIGS['hpo01_configs__logreg'] = hpo01_configs

# ------------------------------------------

logreg_hpo02 = copy.deepcopy(logreg_hpo01)

logreg_hpo02.update({
    "name": "hpo02__refine_c__logreg",
    "stage": "hpo02",
    "feature_group": "refine_c",
    "group": "hpo02__refine_c",
})

logreg_hpo02["hyperparameter_search"]["param_grid"] = {
    "model__C": [
        0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
        1.5, 2.0, 3.0, 5.0,
    ],
}

hpo02_configs = config_to_group(logreg_hpo02)
ALL_HPO_CONFIGS['hpo02_configs__logreg'] = hpo02_configs

#--------------------------------------------

logreg_hpo03 = copy.deepcopy(logreg_hpo02)

logreg_hpo03.update({
    "name": "hpo03__regularization_type__logreg",
    "stage": "hpo03",
    "feature_group": "regularization_type",
    "group": "hpo03__regularization_type",
})

logreg_hpo03["hyperparameter_search"]["param_grid"] = [
    {
        "model__solver": ["saga"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.4, 0.6, 0.8, 1.0, 1.5],
    },
    {
        "model__solver": ["saga"],
        "model__penalty": ["elasticnet"],
        "model__l1_ratio": [0.25, 0.5, 0.75],
        "model__C": [0.4, 0.6, 0.8, 1.0, 1.5],
    },
]

hpo03_configs = config_to_group(logreg_hpo03)
ALL_HPO_CONFIGS['hpo03_configs__logreg'] = hpo03_configs

#--------------------------------------------

logreg_hpo04 = copy.deepcopy(logreg_hpo03)

logreg_hpo04.update({
    "name": "hpo04__solver_comparison__logreg",
    "stage": "hpo04",
    "feature_group": "solver_comparison",
    "group": "hpo04__solver_comparison",
})

logreg_hpo04["hyperparameter_search"]["param_grid"] = {
    "model__solver": [
        "lbfgs",
        "liblinear",
        "newton-cg",
        "sag",
        "saga",
    ],
    "model__C": [0.6, 0.8, 1.0],
    "model__penalty": ["l2"],
}

hpo04_configs = config_to_group(logreg_hpo04)

ALL_HPO_CONFIGS['hpo04_configs__logreg'] = hpo04_configs

#--------------------------------------------

logreg_hpo05 = copy.deepcopy(logreg_hpo04)

logreg_hpo05.update({
    "name": "hpo05__class_weight__logreg",
    "stage": "hpo05",
    "feature_group": "class_weight",
    "group": "hpo05__class_weight",
})

logreg_hpo05["hyperparameter_search"]["param_grid"] = {
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

hpo05_configs = config_to_group(logreg_hpo05)

ALL_HPO_CONFIGS['hpo05_configs__logreg'] = hpo05_configs

#--------------------------------------------

logreg_hpo06 = copy.deepcopy(logreg_hpo05)

logreg_hpo06["model_params"].update({
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,
})

hpo06_configs = config_to_group(logreg_hpo06)
ALL_HPO_CONFIGS['hpo06_configs__logreg'] = hpo06_configs

# SVC hyperparameter optimization configurations

best_svc_fs = ALL_FS_CONFIGS['fs04__fare_fe10_full_context__svc']

svc_base = copy.deepcopy(
    next(iter(best_svc_fs.values()))
)

svc_hpo01 = copy.deepcopy(svc_base)

svc_hpo01.update({
    "name": "svc_hpo01__broad_c",
    "stage": "hyperparameter_optimization",
    "group": "svc_hpo01__broad_c",
})

svc_hpo01["hyperparameter_search"] = {
    "method": "grid",
    "param_grid": {
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    "refit": "accuracy",
}

svc_hpo01_configs = config_to_group(svc_hpo01)

ALL_HPO_CONFIGS["hpo01_configs__svc"] = svc_hpo01_configs

#--------------------------------------------

svc_hpo02 = copy.deepcopy(svc_hpo01)

svc_hpo02.update({
    "name": "svc_hpo02__refined_c",
    "group": "svc_hpo02__refined_c",
})

svc_hpo02["hyperparameter_search"]["param_grid"] = {
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

svc_hpo02_configs = config_to_group(svc_hpo02)

ALL_HPO_CONFIGS["hpo02_configs__svc"] = svc_hpo02_configs

#--------------------------------------------

svc_hpo03 = copy.deepcopy(svc_hpo02)

svc_hpo03.update({
    "name": "svc_hpo03__c_gamma",
    "group": "svc_hpo03__c_gamma",
})

svc_hpo03["hyperparameter_search"]["param_grid"] = {
    "model__C": [0.6, 1.0, 1.5, 2.0],
    "model__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
}

svc_hpo03_configs = config_to_group(svc_hpo03)

ALL_HPO_CONFIGS["hpo03_configs__svc"] = svc_hpo03_configs

#--------------------------------------------

svc_hpo04 = copy.deepcopy(svc_hpo02)

svc_hpo04.update({
    "name": "svc_hpo04__c_gamma",
    "group": "svc_hpo04__c_gamma",
})

svc_hpo04["hyperparameter_search"]["param_grid"] = {
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

svc_hpo04_configs = config_to_group(svc_hpo04)

ALL_HPO_CONFIGS["hpo04_configs__svc"] = svc_hpo04_configs