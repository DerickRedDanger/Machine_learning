from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group
import copy
from titanic_ml.feature_engineering.updated import fe as FE
from titanic_ml.pre_cv_feature_engineering import pre_cv_fe as PRE_CV_FE
from titanic_ml.common.experiments.configurations.exp_config import RAW_FEATURES, baseline_config, ALL_PATCHES

ALL_FS_CONFIGS = {}
LOGREG_FS_CONFIGS = {}

# LogReg feature selection

fs_baseline__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs_baseline',
    feature_group='logreg',
    domain='logreg',
    notes=('Baseline configuration for LogReg feature selection.'
           'Including only title features.')
)

ALL_FS_CONFIGS['logreg__baseline'] = fs_baseline__logreg
LOGREG_FS_CONFIGS['logreg__baseline'] = fs_baseline__logreg

# print(baseline_config)