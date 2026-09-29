from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group,config_to_group
import copy
from titanic_ml.feature_engineering.updated import fe as FE
from titanic_ml.pre_cv_feature_engineering import pre_cv_fe as PRE_CV_FE
from titanic_ml.common.experiments.configurations.exp_config import RAW_FEATURES, baseline_config, ALL_PATCHES

ALL_FS_CONFIGS = {}
LOGREG_FS_CONFIGS = {}

# LogReg feature selection

fs__baseline__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='logreg',
    notes=('Baseline configuration for LogReg feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__logreg'] = config_to_group(fs__baseline__logreg)
LOGREG_FS_CONFIGS['fs__baseline__logreg'] = config_to_group(fs__baseline__logreg)

# fs01 - age logreg

fs01__age_cb03__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb03',
    domain='logreg',
    notes=('experiment 01 for LogReg feature selection.'
           'Title + Cb03 (age imputed by title and pclass + age bins).')
)

ALL_FS_CONFIGS['fs01__age_cb03__logreg'] = config_to_group(fs01__age_cb03__logreg)
LOGREG_FS_CONFIGS['fs01__age_cb03__logreg'] = config_to_group(fs01__age_cb03__logreg)

# # Quick reminder of all Patches names
# for key in ALL_PATCHES.keys():
#     print(key)



fs01__age_cb02__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb02',
    domain='logreg',
    notes=('experiment 01 for LogReg feature selection, second attempt.'
           'Title + Cb02 (age imputed by title + age bins).')
)

ALL_FS_CONFIGS['fs01__age_cb02__logreg'] = config_to_group(fs01__age_cb02__logreg)
LOGREG_FS_CONFIGS['fs01__age_cb02__logreg'] = config_to_group(fs01__age_cb02__logreg)

# Both Fs01 returned small and near similar results. 
# Cb03 was choosen as main option due to slightly better results for accuracy.
# while cb02 will remain as reserve alternative


# fs02 - family - logreg

fs02__family_cb07__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['cb07__family_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='family_cb07',
    domain='logreg',
    notes=('experiment 02 for LogReg feature selection.'
           'addition of cb07 all family features.')
)

ALL_FS_CONFIGS['fs02__family_cb07__logreg'] = config_to_group(fs02__family_cb07__logreg)
LOGREG_FS_CONFIGS['fs02__family_cb07__logreg'] = config_to_group(fs02__family_cb07__logreg)

fs02__family_fe01__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='family_fe01',
    domain='logreg',
    notes=('experiment 02 for LogReg feature selection.'
           'addition of fe01 - Family in place of SibSp and Parch.')
)

ALL_FS_CONFIGS['fs02__family_fe01__logreg'] = config_to_group(fs02__family_fe01__logreg)
LOGREG_FS_CONFIGS['fs02__family_fe01__logreg'] = config_to_group(fs02__family_fe01__logreg)

# Fe01 was choosen as provisional option, it's non negative, but it led to minimal gains

# Current approach:
# Forward assembly: Carry on strong gains and tolerate small non negative gains
# Backward pruning: Once final features are set, slowly remove initially weak features, until only the meaningful ones remain


# fs03 - Cabin - LogReg

fs03__cabin_fe04__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='cabin_fe04',
    domain='logreg',
    notes=('experiment 03 for LogReg feature selection.'
           'addition of fe04 - cabin features.')
)

ALL_FS_CONFIGS['fs03__cabin_fe04__logreg'] = config_to_group(fs03__cabin_fe04__logreg)
LOGREG_FS_CONFIGS['fs03__cabin_fe04__logreg'] = config_to_group(fs03__cabin_fe04__logreg)

# Moderate gains, sufficient to carry forward without worry

# fs04 - Ticket - LogReg

fs04__ticket_fe09_batch__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_batch_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='ticket_fe09_batch',
    domain='logreg',
    notes=('experiment 04 for LogReg feature selection.'
           'addition of fe09 - ticket group size, batch approach.')
)

ALL_FS_CONFIGS['fs04__ticket_fe09_batch__logreg'] = config_to_group(fs04__ticket_fe09_batch__logreg)
LOGREG_FS_CONFIGS['fs04__ticket_fe09_batch__logreg'] = config_to_group(fs04__ticket_fe09_batch__logreg)

fs04__ticket_fe09_fitted__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='ticket_fe09_fitted',
    domain='logreg',
    notes=('experiment 04 for LogReg feature selection.'
           'addition of fe09 - ticket group size, fitted approach.')
)

ALL_FS_CONFIGS['fs04__ticket_fe09_fitted__logreg'] = config_to_group(fs04__ticket_fe09_fitted__logreg)
LOGREG_FS_CONFIGS['fs04__ticket_fe09_fitted__logreg'] = config_to_group(fs04__ticket_fe09_fitted__logreg)

fs04__ticket_fe09_full_context__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='ticket_fe09_full_context',
    domain='logreg',
    notes=('experiment 04 for LogReg feature selection.'
           'addition of fe09 - ticket group size, full_context approach.'),
    pre_cv_scope="full_prediction_context"

)

ALL_FS_CONFIGS['fs04__ticket_fe09_full_context__logreg'] = config_to_group(fs04__ticket_fe09_full_context__logreg)
LOGREG_FS_CONFIGS['fs04__ticket_fe09_full_context__logreg'] = config_to_group(fs04__ticket_fe09_full_context__logreg)

# Minimal gains, one of the prunning candidates for the backward prunning