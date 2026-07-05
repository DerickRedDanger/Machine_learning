# This file defines the configuration for machine learning experiments on the Titanic dataset.
# Each experiment is represented as a dictionary with specific fields that describe the model, features, preprocessing steps, evaluation method, and other relevant information.

from math import exp

from titanic_ml.feature_engineering import add_family_features, add_has_cabin, add_title, add_full_title_feature, age_imputed_by_title, age_imputed_by_title_pclass, add_fare_per_familysize, add_ticket_group_size, add_fare_per_ticket_member, add_deck, add_full_cabin_features
from titanic_ml.common.experiments.config_creation import create_experiment_group
import copy

from titanic_ml.feature_engineering.age_bin import add_age_bin
from titanic_ml.feature_engineering.sex_pclass import add_sex_pclass


# List to hold all experiment configurations
ALL_EXPERIMENTS = {}

# Baseline configuration for experiment config
baseline__config_model = {
        "name": "",
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],

        "feature_engineering": [],

        "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },

        "model_name": "",
        "model_params": {
        },

        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "return_train_score": True,
            "n_jobs": -1,
        },
        "notes": "",
        "stage": "baseline",
        "feature_group": "raw",
        "group": "baseline__raw",
    }

# Baseline configuration for models
baseline__models =[
    {   
    "name": "baseline__raw__logreg",
    "model_name": "logreg",
    "model_params": {
        "max_iter": 1000,
        "random_state": 42,
        },
        "notes": "Base logistic regression, using raw configuration. Baseline for comparison."
        
    },

    {  
    "name": "baseline__raw__knn",
    "model_name": 'knn',
    "model_params":{
        "n_neighbors": 5,
        },
        "notes": "Base kneighbors classifier, using raw configuration. Baseline for comparison."
        
    },

    {   
    "name": "baseline__raw__svc",
    "model_name": 'svc',
    "model_params":{
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": False,
        "random_state": 42,
        },
        "notes": "Base SVC, using raw configuration. Baseline for comparison.",
    },

    {   
    "name": "baseline__raw__decision_tree",
    "model_name": 'decision_tree',
    "model_params":{
        "max_depth": 4,
        "min_samples_leaf": 5,
        "random_state": 42,
        },
        "notes": "Base decision tree, using raw configuration. Baseline for comparison."
    },

    {
    "name": "baseline__raw__random_forest",
    "model_name": 'random_forest',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base random forest, using raw configuration. Baseline for comparison."
    },

    {
    "name": "baseline__raw__extra_trees",
    "model_name": 'extra_trees',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
        },
        "notes": "Base extra trees, using raw configuration. Baseline for comparison."
    },

    {   
    "name": "baseline__raw__xgb",
    "model_name": 'xgb',
    "model_params":{
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        },
        "notes": "Base xgb, using raw configuration. Baseline for comparison."
    },
]

# Baseline experiment configuration
baseline__raw = []

for model in baseline__models:
    config = copy.deepcopy(baseline__config_model)
    config.update(model)
    baseline__raw.append(config)

ALL_EXPERIMENTS['baseline__raw'] = baseline__raw
# New Naming convention:
# <stages>__<feature_group>__<model>



# Feature engineering group 01: Family features - replaces SibSp/Parch with FamilySize and IsAlone.
fe01__family = {}

fe01__family_override = {
    "feature_engineering": [add_family_features],

    "features": [
        "Pclass", "Sex", "Age", "Fare", "Embarked",
        "FamilySize", "IsAlone",
    ],

    "preprocessing": {
        "numeric_features": ["Age", "Fare", "FamilySize", "IsAlone"],
        "onehot_features": ["Sex", "Embarked"],
        "ordinal_features": ["Pclass"],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
    },

    "notes": "Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone.",
}

fe01__family = create_experiment_group(
    stage="fe01",
    feature_group="family",
    base_configs=baseline__raw,
    group_override=fe01__family_override,
)
ALL_EXPERIMENTS['fe01__family'] = fe01__family

# Feature engineering group 02: Has_Cabin - adds a binary feature indicating whether the passenger had a known cabin or not, which can be a signal in itself.
fe02__has_cabin = {}

fe02__has_cabin_override = {
    "feature_engineering": [add_has_cabin],
    "features": ["Pclass", "Sex", "Age", "SibSp", 
                 "Parch", "Fare", "Embarked", "Has_Cabin"],
    "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare",],
            "onehot_features": ["Sex", "Embarked"],
            "ordinal_features": ["Pclass", "Has_Cabin"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
    "notes": "Feature engineering 02: adds Has_Cabin feature, which indicates whether the passenger had a known cabin or not. This is a simple binary feature that attempts to check if the missingess is a signal in itself."
}

fe02__has_cabin = create_experiment_group(
    stage="fe02",
    feature_group="has_cabin",
    base_configs=baseline__raw,
    group_override=fe02__has_cabin_override,
)
ALL_EXPERIMENTS['fe02__has_cabin'] = fe02__has_cabin

# Feature engineering group 03: Deck - adds a feature indicating the deck level of the passenger's cabin, which can be a signal in itself.
fe03__deck = {}

fe03__deck_override = {
    "feature_engineering": [add_deck],
    "features": ["Pclass", "Sex", "Age", "SibSp", 
                 "Parch", "Fare", "Embarked", "Deck"],
    "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked", "Deck"],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
    "notes": "Feature engineering 03: adds Deck feature, which indicates the deck level of the passenger's cabin. This is a categorical feature that can be a signal for survival."
}

fe03__deck = create_experiment_group(
    stage="fe03",
    feature_group="deck",
    base_configs=baseline__raw,
    group_override=fe03__deck_override,
)
ALL_EXPERIMENTS['fe03__deck'] = fe03__deck

# Feature engineering group 04: Cabin features - adds both the Has_Cabin and Deck features together to see if they have a stronger signal when combined.
fe04__cabin_features = {}

fe04__cabin_features_override = {
    "feature_engineering": [add_full_cabin_features],
    "features": ["Pclass", "Sex", "Age", "SibSp", 
                 "Parch", "Fare", "Embarked", "Has_Cabin", "Deck"],
    "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
            "onehot_features": ["Sex", "Embarked", "Deck"],
            "ordinal_features": ["Pclass","Has_Cabin"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
    "notes": "Feature engineering 04: adds both Has_Cabin and Deck features, which indicate the presence of a cabin and its level, respectively."
}

fe04__cabin_features = create_experiment_group(
    stage="fe04",
    feature_group="cabin_features",
    base_configs=baseline__raw,
    group_override=fe04__cabin_features_override,
)
ALL_EXPERIMENTS['fe04__cabin_features'] = fe04__cabin_features

# Feature engineering group 05: Title - adds a feature that extracts the title from the passenger's name, which can give more information about their social status and thus survival chances.
fe05__title = {}

fe05__title_override = {
    "feature_engineering": [add_full_title_feature],
    "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"],
    "preprocessing": {
        "numeric_features": ["Age", "SibSp", "Parch", "Fare"],
        "onehot_features": ["Sex", "Embarked", "Title"],
        "ordinal_features": ["Pclass"],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
        },
    "notes": "Feature engineering 05: adds Title feature, which extracts the title from the passenger's name. This is expected to give more information about the passenger's social status, which can be a strong signal for survival. The title is extracted and then grouped into common titles and a 'Rare' category for less common titles."
}

fe05__title = create_experiment_group(
    stage="fe05",
    feature_group="title",
    base_configs=baseline__raw,
    group_override=fe05__title_override,
)
ALL_EXPERIMENTS['fe05__title'] = fe05__title

# Feature engineering group 06: Age imputation with Title: Imputes age by group imputation utilizing Title.
# Title gives us information about both a passenger age and socio-economical status, which should help us get more precise ages.
fe06__age_imputation_title = {}

fe06__age_imputation_title_override = {
    "feature_engineering": [age_imputed_by_title],
    "notes": "Feature engineering 06: imputing age by grouped impute using title, expected to give better results then mean imputation"
}

fe06__age_imputation_title = create_experiment_group(
    stage="fe06",
    feature_group="age_imputation_title",
    base_configs=baseline__raw,
    group_override=fe06__age_imputation_title_override,
)
ALL_EXPERIMENTS['fe06__age_imputation_title'] = fe06__age_imputation_title

# Feature engineering group 07: Age imputation with Title and Pclass: Imputes age by group imputation utilizing Title and Pclass.
# Titles like Mr are broad, which might lead to a bad imputation on such cases. Pclass might be a good option to narrow down their age.
fe07__age_imputation_title_pclass = {}

fe07__age_imputation_title_pclass_override = {
    "feature_engineering": [age_imputed_by_title_pclass],
    "notes": "Feature engineering 07: imputing age by grouped impute using title and pclass, expected to give better results then imputing with only title"
}

fe07__age_imputation_title_pclass = create_experiment_group(
    stage="fe07",
    feature_group="age_imputation_title_pclass",
    base_configs=baseline__raw,
    group_override=fe07__age_imputation_title_pclass_override,
)
ALL_EXPERIMENTS['fe07__age_imputation_title_pclass'] = fe07__age_imputation_title_pclass


# Feature engineering group 08: Fare divided by family size - adds a feature meant to extract the economical cost of the fare per family member, rather then total.
# This should give the df a better approach of each passengers actual individual fare
fe08__fare_per_family_member = {}

fe08__fare_per_family_member_override = {
    "feature_engineering": [add_fare_per_familysize],
        "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare/FamilySize"],
            "preprocessing": {
                "numeric_features": ["Age", "SibSp", "Parch", "Fare/FamilySize"],
                "onehot_features": ["Sex", "Embarked",],
                "ordinal_features": ["Pclass"],
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "scaler": "standard",
        },
    "notes": "Feature engineering 08: Dividing fare by family size, expected to give a small increase in accuracy."
}

fe08__fare_per_family_member = create_experiment_group(
    stage="fe08",
    feature_group="fare_per_family_member",
    base_configs=baseline__raw,
    group_override=fe08__fare_per_family_member_override,
)
ALL_EXPERIMENTS['fe08__fare_per_family_member'] = fe08__fare_per_family_member


# Feature engineering group 09: Ticket group size - adds a feature meant to show how many passengers share the same ticket.
# Expected to have a influence akin to family size, it not slightly better, as passengers sharing the same ticket might form more consistent groups then family size. 
# As not all family members might be together, sharing the same ticket
fe09__ticket_group_size = {}

fe09__ticket_group_size_override = {
    "feature_engineering": [add_ticket_group_size],
    "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "TicketGroupSize"],
    "preprocessing": {
        "numeric_features": ["Age", "SibSp", "Parch", "Fare", "TicketGroupSize"],
        "onehot_features": ["Sex", "Embarked",],
        "ordinal_features": ["Pclass"],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
        },
    "notes": "Feature engineering 09: imputing age by grouped impute using title and pclass, expected to give better results then imputing with only title"
}

fe09__ticket_group_size = create_experiment_group(
    stage="fe09",
    feature_group="ticket_group_size",
    base_configs=baseline__raw,
    group_override=fe09__ticket_group_size_override,
)
ALL_EXPERIMENTS['fe09__ticket_group_size'] = fe09__ticket_group_size

# Feature engineering group 10: Fare divided by ticket ground size - adds a feature meant to extract the economical cost of the fare per ticket member, rather then total.
# This is expected to give a slightly better approach then family members, as passengers sharing the same ticket are expected to form more consitent groups.
fe10__fare_per_ticket_member = {}

fe10__fare_per_ticket_member_override = {
    "feature_engineering": [add_fare_per_ticket_member],
    "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare/TicketMember"],
    "preprocessing": {
            "numeric_features": ["Age", "SibSp", "Parch", "Fare/TicketMember"],
            "onehot_features": ["Sex", "Embarked",],
            "ordinal_features": ["Pclass"],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
    "notes": "Feature engineering 10: Dividing fare by Ticket member, expected to give a small increase in accuracy, better then fare/Family size."
}

fe10__fare_per_ticket_member = create_experiment_group(
    stage="fe10",
    feature_group="fare_per_ticket_member",
    base_configs=baseline__raw,
    group_override=fe10__fare_per_ticket_member_override,
)
ALL_EXPERIMENTS['fe10__fare_per_ticket_member'] = fe10__fare_per_ticket_member


# Feature engineering group 11: Age bins - adds a feature meant to categorize passengers by age groups.
# Expected to give better results then using raw age values, by better evaluating the survival chances based on the passenger's age group.
# While taking advantage of the labels to turn it into a ordinal feature, which is expected to give better results then onehot encoding the age groups.
fe11__age_bin = {}

fe11__age_bin_override = {
    "feature_engineering": [add_age_bin],
    "features": ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Age_bin"],
    "preprocessing": {
            "numeric_features": ["SibSp", "Parch",],
            "onehot_features": ["Sex", "Embarked",],
            "ordinal_features": ["Pclass", "Age_bin",],
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
    "notes": "Feature engineering 11: Creating age bins, expected to give better results then using raw age values, by better evaluating the survival chances based on the passenger's age group."
}

fe11__age_bin = create_experiment_group(
    stage="fe11",
    feature_group="age_bin",
    base_configs=baseline__raw,
    group_override=fe11__age_bin_override,
)

ALL_EXPERIMENTS['fe11__age_bin'] = fe11__age_bin


# Feature engineering group 12: Sex Pclass - adds a feature meant to combine the passenger's sex and pclass into a single feature.
# This is a proof of concept to see if combining two features into a single feature can give better results then using them separately.

fe12__sex_pclass={}

fe12__sex_pclass = {
    "feature_engineering": [add_sex_pclass],

    "features": [
        "Age", "Fare", "Embarked", "Sex_Pclass",
    ],

    "preprocessing": {
        "numeric_features": ["Age", "Fare",],
        "onehot_features": ["Sex_Pclass", "Embarked"],
        "ordinal_features": [],
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "standard",
    },

    "notes": "Feature engineering 12: Combines Sex and Pclass into a single feature.",
}

fe12__sex_pclass = create_experiment_group(
    stage="fe12",
    feature_group="sex_pclass",
    base_configs=baseline__raw,
    group_override=fe12__sex_pclass,
)
ALL_EXPERIMENTS['fe12__sex_pclass'] = fe12__sex_pclass

EXPERIMENTS_GUIDE = [
    {
        "name": "baseline_raw_logreg",
        "model_name": "logreg",
        "model_params": {
            "max_iter": 1000
        },
        "features": [
            "Pclass", "Sex", "Age", "Fare", "Embarked"
        ],
        "preprocessing": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
            "encoder": "onehot",
            "handle_outliers": False
        },
        "evaluation": {
            "method": "cross_validation",
            "cv": 5,
            "scoring": ["accuracy", "precision", "recall", "f1"]
        },
        "use_tuning": False,
        "param_grid": None,
        "notes": "Base logistic regression, using raw configuration. Baseline for comparission"
    },
]

'''
| Feature family | Experiments      |
| -------------- | ---------------- |
| Family         | fe01, fe09       |
| Cabin          | fe02, fe03, fe04 |
| Title          | fe05             |
| Age            | fe06, fe07, fe11 |
| Fare/Ticket    | fe08, fe10       |
| Sex + Pclass   | fe12             |
'''

'''
# Combo group 1 - redundancy

# cb01
# Age + Age_bin

# cb02
# Fare + Fare/Family

# cb03
# Fare + Fare/Ticket

# cb04
# Fare + Fare/Family + Fare/Ticket

# cb05
# SibSp/Parch + FamilySize

# cb06
# Sex + Pclass + Sex_Pclass
'''

'''
# Combo group 2 - Same domain

# cb07
# Title + Age Imputation

# cb08
# Title + Age Bin

# cb09
# Age Imputation + Age Bin

# cb10
# Title + Age Imputation + Age Bin
'''

"""
Field by field meaning:

- name: A unique identifier for the experiment.
Readable, short, and descriptive.
example: "Baseline Logistic Regression", "Random Forest with Tuning", "XGBoost with Feature Engineering"

- model_name: The key corresponding to the model in the MODEL_REGISTRY.
Must match one of the keys defined in the MODEL_REGISTRY dictionary.
example: "logreg", "rf", "xgb"

- model_params: A dictionary of fixed parameters to initialize the model with.
example: {"n_estimators": 200, "max_depth": 5}

- features: A list of feature names to be used in the experiment.
example: ["Pclass", "Sex", "Age", "Fare", "Embarked"],

- preprocessing: A dictionary defining the preprocessing steps to be applied to the experiment before training the model.
example: {"numeric_imputer": "median", "categorical_imputer": "most_frequent", "scaler": "standard", "encoder": "onehot", "handle_outliers": False}

- evaluation: A dictionary defining the evaluation method and metrics to be used for assessing the model's performance.
example: {"method": "cross_validation", "cv": 5, "scoring": ["accuracy", "precision", "recall", "f1"]}

- use_tuning: A boolean indicating whether to perform hyperparameter tuning for this experiment.
example: True, False

- param_grid: A dictionary defining the hyperparameters and their respective values to be used for tuning.
Only relevant if use_tuning is True.
example: {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]}

- notes: A brief description of the experiment and the reasoning behind it, or any additional information about the experiment.
example: "This experiment tests the performance of a Random Forest model with default parameters. Meant to be used as baseline for comparison with tuned models."
"""