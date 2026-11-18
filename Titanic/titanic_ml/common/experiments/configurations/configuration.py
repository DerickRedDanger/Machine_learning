from titanic_ml.common.experiments.configurations.exp_config import ALL_EXPERIMENTS
from titanic_ml.common.experiments.configurations.feature_selection_config import ALL_FS_CONFIGS, LOGREG_FS_CONFIGS
from titanic_ml.common.experiments.configurations.hyper_parameter_config import ALL_HPO_CONFIGS, ALL_THRESHOLD_CONFIGS
from titanic_ml.common.experiments.utils import index_experiments_by_name, validate_threshold_config_references


ALL_FE_CONFIG = ALL_EXPERIMENTS
ALL_FS_CONFIGS = ALL_FS_CONFIGS
ALL_HPO_CONFIGS = ALL_HPO_CONFIGS

FE_CONFIGS_BY_NAME = index_experiments_by_name(
    ALL_FE_CONFIG
)
FS_CONFIGS_BY_NAME = index_experiments_by_name(
    ALL_FS_CONFIGS
)
HPO_CONFIGS_BY_NAME = index_experiments_by_name(
    ALL_HPO_CONFIGS
)

LOGREG_FS_CONFIGS = LOGREG_FS_CONFIGS


validate_threshold_config_references(
    ALL_THRESHOLD_CONFIGS,
    HPO_CONFIGS_BY_NAME,
    debug=False
)

ALL_THRESHOLD_CONFIGS = ALL_THRESHOLD_CONFIGS