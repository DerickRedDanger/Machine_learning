from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group,config_to_group
import copy
from titanic_ml.feature_engineering.updated import fe as FE
from titanic_ml.pre_cv_feature_engineering import pre_cv_fe as PRE_CV_FE
from titanic_ml.common.experiments.configurations.exp_config import RAW_FEATURES, baseline_config, ALL_PATCHES

ALL_FS_CONFIGS = {}
LOGREG_FS_CONFIGS = {}

# LogReg feature selection

fs__baseline_logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='logreg',
    notes=('Baseline configuration for LogReg feature selection.'
           'Including only title features.')
)

ALL_FS_CONFIGS['fs__baseline_logreg'] = config_to_group(fs__baseline_logreg)
LOGREG_FS_CONFIGS['fs__baseline_logreg'] = config_to_group(fs__baseline_logreg)

# print(baseline_config)