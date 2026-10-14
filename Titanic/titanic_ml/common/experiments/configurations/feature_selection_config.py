from titanic_ml.common.experiments.config_creation import create_config, create_config_group, features_from_preprocessing, validate_config_group,config_to_group
import copy
from titanic_ml.feature_engineering.updated import fe as FE
from titanic_ml.pre_cv_feature_engineering import pre_cv_fe as PRE_CV_FE
from titanic_ml.common.experiments.configurations.exp_config import RAW_FEATURES, baseline_config, ALL_PATCHES

ALL_FS_CONFIGS = {}
LOGREG_FS_CONFIGS = {}

removing_fare_patch={
        "remove": {
                "preprocessing": {
                    "numeric_features": [
                        "Fare",
                    ],
                },
            },
}

removing_sibSp_parch={
        "remove": {
                "preprocessing": {
                    "numeric_features": [
                        "SibSp",
                        "Parch"
                    ],
                },
            },
}

removing_raw_age_patch={
        "remove": {
                "preprocessing": {
                    "numeric_features": [
                        "Age",
                    ],
                },
            },
}

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

# Fitted and batched achieved best results, choosing fitted as main option, due to being easier to reason about and implement
# but still achieved minimal gains, one of the prunning candidates for the backward prunning

# fs05 - fare - LogReg
fs05__fare_fe08__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['fe08__fare_per_family_member_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_fe08',
    domain='logreg',
    notes=('experiment 05 for LogReg feature selection.'
           'addition of fe08 - fare per family member.')
)

ALL_FS_CONFIGS['fs05__fare_fe08__logreg'] = config_to_group(fs05__fare_fe08__logreg)
LOGREG_FS_CONFIGS['fs05__fare_fe08__logreg'] = config_to_group(fs05__fare_fe08__logreg)

fs05__fare_cb05_fitted__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_cb05_fitted',
    domain='logreg',
    notes=('experiment 05 for LogReg feature selection.'
           'addition of cb05 - fare and fare per ticket, fitted approach.')
)

ALL_FS_CONFIGS['fs05__fare_cb05_fitted__logreg'] = config_to_group(fs05__fare_cb05_fitted__logreg)
LOGREG_FS_CONFIGS['fs05__fare_cb05_fitted__logreg'] = config_to_group(fs05__fare_cb05_fitted__logreg)

fs05__fare_cb05_full_context__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_full_context_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_cb05_full_context',
    domain='logreg',
    notes=('experiment 05 for LogReg feature selection.'
           'addition of cb05 - fare and fare per ticket, full_context approach.'),
    pre_cv_scope="full_prediction_context"

)

ALL_FS_CONFIGS['fs05__fare_cb05_full_context__logreg'] = config_to_group(fs05__fare_cb05_full_context__logreg)
LOGREG_FS_CONFIGS['fs05__fare_cb05_full_context__logreg'] = config_to_group(fs05__fare_cb05_full_context__logreg)

# carrying cb05 fitted, it's results were completely neutral, meaning it didn't add anything to our model
# But it still passes the original idea of carrying the best non negative foward and prune them at the end. and this is now the main prunning option.

# fs06 - Sex/Pclass combination - logReg

fs06__sex_pclass_fe12__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'],
        ALL_PATCHES['fe12__sex_pclass_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='sex_pclass_fe12',
    domain='logreg',
    notes=('experiment 06 for LogReg feature selection.'
           'addition of fe12 - sex_pclass, replacing sex and pclass.')
)

ALL_FS_CONFIGS['fs06__sex_pclass_fe12__logreg'] = config_to_group(fs06__sex_pclass_fe12__logreg)
LOGREG_FS_CONFIGS['fs06__sex_pclass_fe12__logreg'] = config_to_group(fs06__sex_pclass_fe12__logreg)



fs06__sex_pclass_cb08__logreg = create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe01__family_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'],
        ALL_PATCHES['cb08__sex_pclass_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='sex_pclass_cb08',
    domain='logreg',
    notes=('experiment 06 for LogReg feature selection.'
           'addition of cb08 - sex_pclass features, adding sex_pclass to raw features.')
)

ALL_FS_CONFIGS['fs06__sex_pclass_cb08__logreg'] = config_to_group(fs06__sex_pclass_cb08__logreg)
LOGREG_FS_CONFIGS['fs06__sex_pclass_cb08__logreg'] = config_to_group(fs06__sex_pclass_cb08__logreg)

# Both options led negative results, so no sex_pclass feature was carried over
# final fs configuration: fs05__fare_cb05_fitted__logreg
# final patch configuration after fs:
# [
#         ALL_PATCHES['fe05__title_patch'],
#         ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
#         ALL_PATCHES['fe01__family_patch'],
#         ALL_PATCHES['fe04__cabin_features_patch'],
#         ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
#         ALL_PATCHES['cb05__fare_and_fare_per_ticket_full_context_patch'],
#     ],

# Prunning 01 - fare - LogReg

pr01__removing_family__logreg=create_config(
    base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr01',
    feature_group='removing_family',
    domain='logreg',
    notes=('pruning 01 for LogReg feature selection.'
            'removal of family feature.')
    )

ALL_FS_CONFIGS['pr01__removing_family__logreg'] = config_to_group(pr01__removing_family__logreg)
LOGREG_FS_CONFIGS['pr01__removing_family__logreg'] = config_to_group(pr01__removing_family__logreg)

# 0.000/+0.001, minimal gains

pr02__removing_ticket__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='removing_ticket',
    domain='logreg',
    notes=('pruning 02 for LogReg feature selection.'
            'removal of ticket feature.')
    )

ALL_FS_CONFIGS['pr02__removing_ticket__logreg'] = config_to_group(pr02__removing_ticket__logreg)
LOGREG_FS_CONFIGS['pr02__removing_ticket__logreg'] = config_to_group(pr02__removing_ticket__logreg)

# +0.002/+0.003

pr03__removing_non_raw_fare__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='removing_non_raw_fare',
    domain='logreg',
    notes=('pruning 03 for LogReg feature selection.'
            'removal of non-raw fare feature.')
    )

ALL_FS_CONFIGS['pr03__removing_non_raw_fare__logreg'] = config_to_group(pr03__removing_non_raw_fare__logreg)
LOGREG_FS_CONFIGS['pr03__removing_non_raw_fare__logreg'] = config_to_group(pr03__removing_non_raw_fare__logreg)

# -0.005/-0.007

pr03__removing_fare__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='removing_fare',
    domain='logreg',
    notes=('pruning 03 for LogReg feature selection.'
            'removal of fare feature.')
    )

ALL_FS_CONFIGS['pr03__removing_fare__logreg'] = config_to_group(pr03__removing_fare__logreg)
LOGREG_FS_CONFIGS['pr03__removing_fare__logreg'] = config_to_group(pr03__removing_fare__logreg)
# +0.004/+0.005

pr03__removing_raw_fare__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='removing_raw_fare',
    domain='logreg',
    notes=('pruning 03 for LogReg feature selection.'
            'removal of raw fare feature.')
    )

ALL_FS_CONFIGS['pr03__removing_raw_fare__logreg'] = config_to_group(pr03__removing_raw_fare__logreg)
LOGREG_FS_CONFIGS['pr03__removing_raw_fare__logreg'] = config_to_group(pr03__removing_raw_fare__logreg)
# -0.001/+0.001
# carried on with pr03__removing_fare__logreg


pr04__removing_age__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr04',
    feature_group='removing_age',
    domain='logreg',
    notes=('pruning 04 for LogReg feature selection.'
            'removal of age feature.')
    )

ALL_FS_CONFIGS['pr04__removing_age__logreg'] = config_to_group(pr04__removing_age__logreg)
LOGREG_FS_CONFIGS['pr04__removing_age__logreg'] = config_to_group(pr04__removing_age__logreg)
# - 0.017/ -0.019
# Not removing age

pr05__removing_cabin__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr05',
    feature_group='removing_cabin',
    domain='logreg',
    notes=('pruning 05 for LogReg feature selection.'
            'removal of cabin feature.')
    )

ALL_FS_CONFIGS['pr05__removing_cabin__logreg'] = config_to_group(pr05__removing_cabin__logreg)
LOGREG_FS_CONFIGS['pr05__removing_cabin__logreg'] = config_to_group(pr05__removing_cabin__logreg)
# -0.009/-0.015

pr06__removing_title__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr06',
    feature_group='removing_title',
    domain='logreg',
    notes=('pruning 06 for LogReg feature selection.'
            'removal of title feature.')
    )

ALL_FS_CONFIGS['pr06__removing_title__logreg'] = config_to_group(pr06__removing_title__logreg)
LOGREG_FS_CONFIGS['pr06__removing_title__logreg'] = config_to_group(pr06__removing_title__logreg)
#-0.037/-0.053


prsc__age_cb02__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='prsc',
    feature_group='age_cb02',
    domain='logreg',
    notes=('pruning sanity check for LogReg feature selection.'
            'Cb02 and Cb03 had nearly identical performance.'
            'This experiment is meant to see if they result remains the same.')
    )

ALL_FS_CONFIGS['prsc__age_cb02__logreg'] = config_to_group(prsc__age_cb02__logreg)
LOGREG_FS_CONFIGS['prsc__age_cb02__logreg'] = config_to_group(prsc__age_cb02__logreg)
#-0.009/-0.01, cb03 is going to remain as the main option.


pr07__removing_sibSp_parch__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
        removing_sibSp_parch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr07',
    feature_group='removing_sibSp_parch',
    domain='logreg',
    notes=('pruning 07 for LogReg feature selection.'
            'removal of sibSp and parch features.')
    )

ALL_FS_CONFIGS['pr07__removing_sibSp_parch__logreg'] = config_to_group(pr07__removing_sibSp_parch__logreg)
LOGREG_FS_CONFIGS['pr07__removing_sibSp_parch__logreg'] = config_to_group(pr07__removing_sibSp_parch__logreg)
# -0.041/-0.05



pr08__removing_all_age_related__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
        removing_raw_age_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr08',
    feature_group='removing_all_age_related',
    domain='logreg',
    notes=('pruning 08 for LogReg feature selection.'
            'removal of all age-related features.')
    )

ALL_FS_CONFIGS['pr08__removing_all_age_related__logreg'] = config_to_group(pr08__removing_all_age_related__logreg)
LOGREG_FS_CONFIGS['pr08__removing_all_age_related__logreg'] = config_to_group(pr08__removing_all_age_related__logreg)
#-0.021/ -0.025

pr08__removing_raw_age__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
        removing_raw_age_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr08',
    feature_group='removing_raw_age',
    domain='logreg',
    notes=('pruning 08 for LogReg feature selection.'
            'removal of raw age feature.')
    )

ALL_FS_CONFIGS['pr08__removing_raw_age__logreg'] = config_to_group(pr08__removing_raw_age__logreg)
LOGREG_FS_CONFIGS['pr08__removing_raw_age__logreg'] = config_to_group(pr08__removing_raw_age__logreg)
# -0.023/ -0.032

pr08__removing_raw_age__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
        removing_raw_age_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr08',
    feature_group='removing_raw_age',
    domain='logreg',
    notes=('pruning 08 for LogReg feature selection.'
            'removal of raw age feature.')
    )

ALL_FS_CONFIGS['pr08__removing_raw_age__logreg'] = config_to_group(pr08__removing_raw_age__logreg)
LOGREG_FS_CONFIGS['pr08__removing_raw_age__logreg'] = config_to_group(pr08__removing_raw_age__logreg)

# SVC feature selection

SVC_FS_CONFIGS = {}

# SVC Baseline

fs__baseline__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='svc',
    notes=('Baseline configuration for SVC feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__svc'] = config_to_group(fs__baseline__svc)
SVC_FS_CONFIGS['fs__baseline__svc'] = config_to_group(fs__baseline__svc)

# fs 01 - age - SVC

fs01__age_cb02__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb02',
    domain='svc',
    notes=('Experiment 01 for SVC feature selection.'
           'Addition of cb02, Age feature.')
)

ALL_FS_CONFIGS['fs01__age_cb02__svc'] = config_to_group(fs01__age_cb02__svc)
SVC_FS_CONFIGS['fs01__age_cb02__svc'] = config_to_group(fs01__age_cb02__svc)

# +0.001/+0.002

fs01__age_cb03__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb03',
    domain='svc',
    notes=('Experiment 01 for SVC feature selection.'
           'Addition of cb03, Age feature.')
)

ALL_FS_CONFIGS['fs01__age_cb03__svc'] = config_to_group(fs01__age_cb03__svc)
SVC_FS_CONFIGS['fs01__age_cb03__svc'] = config_to_group(fs01__age_cb03__svc)
# +0.001/+0.002

# Subce cb02 uses a simpler approach, it will be the one carried forward, while cb03 will remain as a reserve option.

# fs 02 - ticket - SVC

fs02__ticket_fe09__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='ticket_fe09_full_context',
    domain='svc',
    notes=('Experiment 02 for SVC feature selection.'
           'Addition of fe09 - ticket full context, Ticket feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs02__ticket_fe09_full_context__svc'] = config_to_group(fs02__ticket_fe09__svc)
SVC_FS_CONFIGS['fs02__ticket_fe09_full_context__svc'] = config_to_group(fs02__ticket_fe09__svc)
# -0.001/0


fs02__ticket_fe09_fitted__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='ticket_fe09_fitted',
    domain='svc',
    notes=('Experiment 02 for SVC feature selection.'
           'Addition of fe09 - ticket fitted, Ticket feature.'),
)

ALL_FS_CONFIGS['fs02__ticket_fe09_fitted__svc'] = config_to_group(fs02__ticket_fe09_fitted__svc)
SVC_FS_CONFIGS['fs02__ticket_fe09_fitted__svc'] = config_to_group(fs02__ticket_fe09_fitted__svc)
# -0.006/-0.005

# Both results were negative, none of the options will be carried over.

# This was the last of the recommended feature engineering, but due to it's still low results
# I decided to check other features that still lead to positive, even if minimal results to see how it
# would affect the model.

# fs 03 - family - SVC

fs03__family_cb07__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb07__family_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='family_cb07',
    domain='svc',
    notes=('Experiment 03 for SVC feature selection.'
           'Addition of cb07 - family features.'),
)

ALL_FS_CONFIGS['fs03__family_cb07__svc'] = config_to_group(fs03__family_cb07__svc)
SVC_FS_CONFIGS['fs03__family_cb07__svc'] = config_to_group(fs03__family_cb07__svc)
# -0.002/-0.003
# Negative results, not carried over

# fs04 - fare - SVC

fs04__fare_fe10_full_context__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='fare_fe10_full_context',
    domain='svc',
    notes=('Experiment 04 for SVC feature selection.'
           'Addition of fe10, Fare/ticket feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs04__fare_fe10_full_context__svc'] = config_to_group(fs04__fare_fe10_full_context__svc)
SVC_FS_CONFIGS['fs04__fare_fe10_full_context__svc'] = config_to_group(fs04__fare_fe10_full_context__svc)
# +0.001/+0.001

fs04__fare_fe10_fitted__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_fitted_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='fare_fe10_fitted',
    domain='svc',
    notes=('Experiment 04 for SVC feature selection.'
           'Addition of fe10, Fare/ticket feature.')
)

ALL_FS_CONFIGS['fs04__fare_fe10_fitted__svc'] = config_to_group(fs04__fare_fe10_fitted__svc)
SVC_FS_CONFIGS['fs04__fare_fe10_fitted__svc'] = config_to_group(fs04__fare_fe10_fitted__svc)
#-0.003/-0.003

fs04__fare_fe08__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe08__fare_per_family_member_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='fare_fe08',
    domain='svc',
    notes=('Experiment 04 for SVC feature selection.'
           'Addition of fe08, Fare per family member feature.')
)

ALL_FS_CONFIGS['fs04__fare_fe08__svc'] = config_to_group(fs04__fare_fe08__svc)
SVC_FS_CONFIGS['fs04__fare_fe08__svc'] = config_to_group(fs04__fare_fe08__svc)
# -0.001/-0.002

# fe10 full context was chosen as main option, as it was the only positive, if minimal, gain.
# fe10 fitted and fe08 were negative, so they will not be carried over.

# fs 05 - cabin - SVC
fs05__cabin_fe04__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe04__cabin_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='cabin_fe04',
    domain='svc',
    notes=('Experiment 05 for SVC feature selection.'
           'Addition of fe04, Cabin features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs05__cabin_fe04__svc'] = config_to_group(fs05__cabin_fe04__svc)
SVC_FS_CONFIGS['fs05__cabin_fe04__svc'] = config_to_group(fs05__cabin_fe04__svc)
# -0.002/-0.002
# Cabin features were negative, so they will not be carried over.
# Since cabin features were pretty similar and achieved similar results in fe, I decided not to test the others.

# fs 06 - sex_pclass - SVC

fs06__sex_pclass_cb08__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['cb08__sex_pclass_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='sex_pclass_cb08',
    domain='svc',
    notes=('Experiment 06 for SVC feature selection.'
           'Addition of cb08, Sex/Pclass features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs06__sex_pclass_cb08__svc'] = config_to_group(fs06__sex_pclass_cb08__svc)
SVC_FS_CONFIGS['fs06__sex_pclass_cb08__svc'] = config_to_group(fs06__sex_pclass_cb08__svc)
# -0.004/-0.009

fs06__sex_pclass_fe12__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe12__sex_pclass_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='sex_pclass_fe12',
    domain='svc',
    notes=('Experiment 06 for SVC feature selection.'
           'Addition of fe12, Sex/Pclass features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs06__sex_pclass_fe12__svc'] = config_to_group(fs06__sex_pclass_fe12__svc)
SVC_FS_CONFIGS['fs06__sex_pclass_fe12__svc'] = config_to_group(fs06__sex_pclass_fe12__svc)
# -0.005/-0.014

# Both returned negative results, so none were carried over

# Prunning phase for SVC

# pr01 - removing fare - SVC

pr01__removing_all_fare__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr01',
    feature_group='removing_all_fare',
    domain='svc',
    notes=('Pruning Experiment 01 for SVC feature selection.'
           'Removal of all fare features.'),
)

ALL_FS_CONFIGS['pr01__removing_all_fare__svc'] = config_to_group(pr01__removing_all_fare__svc)
SVC_FS_CONFIGS['pr01__removing_all_fare__svc'] = config_to_group(pr01__removing_all_fare__svc)

# -0.002/-0.004
# Due to the negative results, fare was not removed.

# Pr02 - removing age - SVC
pr02__removing_age_imputed__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='removing_age_imputed',
    domain='svc',
    notes=('Pruning Experiment 02 for SVC feature selection.'
           'Removal of age imputed features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr02__removing_age_imputed__svc'] = config_to_group(pr02__removing_age_imputed__svc)
SVC_FS_CONFIGS['pr02__removing_age_imputed__svc'] = config_to_group(pr02__removing_age_imputed__svc)
#-0.003/-0.005

pr02__removing_all_age__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        removing_raw_age_patch
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='removing_all_age',
    domain='svc',
    notes=('Pruning Experiment 02 for SVC feature selection.'
           'Removal of all age features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr02__removing_all_age__svc'] = config_to_group(pr02__removing_all_age__svc)
SVC_FS_CONFIGS['pr02__removing_all_age__svc'] = config_to_group(pr02__removing_all_age__svc)
#-0.002/-0.003

# pr03 - removing title - SVC

pr03__removing_title__svc = create_config(
    base_config=baseline_config['svc__raw'],
    patches = [
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='removing_title',
    domain='svc',
    notes=('Pruning Experiment 03 for SVC feature selection.'
           'Removal of title features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr03__removing_title__svc'] = config_to_group(pr03__removing_title__svc)
SVC_FS_CONFIGS['pr03__removing_title__svc'] = config_to_group(pr03__removing_title__svc)

 # Random forest Feature selection
RANDOMFOREST_FS_CONFIGS={}
 # Baseline configuration for Random Forest feature selection

fs__baseline__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='randomforest',
    notes=('Baseline configuration for Random Forest feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__random_forest'] = config_to_group(fs__baseline__random_forest)
RANDOMFOREST_FS_CONFIGS['fs__baseline__random_forest'] = config_to_group(fs__baseline__random_forest)

# Age - cb03 > cb02

# family - raw

# cabin - raw

# ticket - raw

# fare - cb06, fe10 / fare ablation

# sex/pclass - raw


# fs 01 - age - Random Forest

fs01__age_cb03__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb03',
    domain='randomforest',
    notes=('Experiment 01 for Random Forest feature selection.'
           'addition of cb03, age imputed by title/pclass and bins.')
)

ALL_FS_CONFIGS['fs01__age_cb03__random_forest'] = config_to_group(fs01__age_cb03__random_forest)
RANDOMFOREST_FS_CONFIGS['fs01__age_cb03__random_forest'] = config_to_group(fs01__age_cb03__random_forest)
#-0.001/-0.004

fs01__age_cb02__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb02',
    domain='randomforest',
    notes=('Experiment 01 for Random Forest feature selection.'
           'addition of cb02, age imputed by title and bins.')
)

ALL_FS_CONFIGS['fs01__age_cb02__random_forest'] = config_to_group(fs01__age_cb02__random_forest)
RANDOMFOREST_FS_CONFIGS['fs01__age_cb02__random_forest'] = config_to_group(fs01__age_cb02__random_forest)
# -0.004/-0.008

# Both were negative, so no age was carried over

# fs02 - fare - Random Forest

fs02__fare_fe10_full_context__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='fare_fe10_full_context',
    domain='randomforest',
    notes=('Experiment 02 for Random Forest feature selection.'
           'addition of fe10, fare per ticket member and full context.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs02__fare_fe10_full_context__random_forest'] = config_to_group(fs02__fare_fe10_full_context__random_forest)
RANDOMFOREST_FS_CONFIGS['fs02__fare_fe10_full_context__random_forest'] = config_to_group(fs02__fare_fe10_full_context__random_forest)
#-0.004/-0.009

fs02__fare_cb06_fitted__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb06__all_fare_features_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='fare_cb06_fitted',
    domain='randomforest',
    notes=('Experiment 02 for Random Forest feature selection.'
           'addition of cb06, all fare features.')
)

ALL_FS_CONFIGS['fs02__fare_cb06_fitted__random_forest'] = config_to_group(fs02__fare_cb06_fitted__random_forest)
RANDOMFOREST_FS_CONFIGS['fs02__fare_cb06_fitted__random_forest'] = config_to_group(fs02__fare_cb06_fitted__random_forest)
# 0/-0.002

# none of the results were positive, so no fare features were carried over.

# Due to the absence of other recommended features, as to the other feature engineering not returning positive results during fe
# I decided to stop the feature selection here and start feature pruning

# pruning phase for Random Forest

pr01__removing_raw_fare__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr01',
    feature_group='removing_raw_fare',
    domain='randomforest',
    notes=('pruning experiment 01 for Random Forest.'
           'removing raw fare.')
)

ALL_FS_CONFIGS['pr01__removing_raw_fare__random_forest'] = config_to_group(pr01__removing_raw_fare__random_forest)
RANDOMFOREST_FS_CONFIGS['pr01__removing_raw_fare__random_forest'] = config_to_group(pr01__removing_raw_fare__random_forest)
# 0/-0.001
# the result wasn't positive, so this isn't a configuration to carry over.
# with that said, results from Fe with ablation showed that removing fare while using cb03 resulting in positive gains.
# will make one last experiment to see if that result is still true while having title

pr02__age_without_fare_cb03__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='age_without_fare_cb03',
    domain='randomforest',
    notes=('pruning experiment 02 for Random Forest.'
           'removing fare while adding cb03, age imputed by title/pclass and bins.')
)

ALL_FS_CONFIGS['pr02__age_without_fare_cb03__random_forest'] = config_to_group(pr02__age_without_fare_cb03__random_forest)
RANDOMFOREST_FS_CONFIGS['pr02__age_without_fare_cb03__random_forest'] = config_to_group(pr02__age_without_fare_cb03__random_forest)
# -0.003/-0.002

pr03__age_without_fare_no_title_cb03__random_forest = create_config(
    base_config=baseline_config['random_forest__raw'],
    patches = [
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
        removing_fare_patch,
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='age_without_fare_no_title_cb03',
    domain='randomforest',
    notes=('pruning experiment 03 for Random Forest.'
           'removing fare while adding cb03, age imputed by title/pclass and bins.')
)

ALL_FS_CONFIGS['pr03__age_without_fare_no_title_cb03__random_forest'] = config_to_group(pr03__age_without_fare_no_title_cb03__random_forest)
RANDOMFOREST_FS_CONFIGS['pr03__age_without_fare_no_title_cb03__random_forest'] = config_to_group(pr03__age_without_fare_no_title_cb03__random_forest)
# +0.002/-0.006


# K - Nearest Neighbor Feature selection

KNN_FS_CONFIGS={}
# Baseline configuration for KNN feature selection

fs__baseline__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='knn',
    notes=('Baseline configuration for KNN feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__knn'] = config_to_group(fs__baseline__knn)
KNN_FS_CONFIGS['fs__baseline__knn'] = config_to_group(fs__baseline__knn)

# fs01 - fare - KNN

fs01__fare_fe10_full_context__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='fare_fe10_full_context',
    domain='knn',
    notes=('experiment 01 for KNN feature selection.'
           'adding fare per ticket member full context feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs01__fare_fe10_full_context__knn'] = config_to_group(fs01__fare_fe10_full_context__knn)
KNN_FS_CONFIGS['fs01__fare_fe10_full_context__knn'] = config_to_group(fs01__fare_fe10_full_context__knn)
# + 0.009/0.011

# fs02 - cabin - KNN

fs02__cabin_fe03__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe03__deck_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='cabin_fe03',
    domain='knn',
    notes=('experiment 02 for KNN feature selection.'
           'adding deck feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs02__cabin_fe03__knn'] = config_to_group(fs02__cabin_fe03__knn)
KNN_FS_CONFIGS['fs02__cabin_fe03__knn'] = config_to_group(fs02__cabin_fe03__knn)
# +0.001/-0.003

fs02__cabin_fe04__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='cabin_fe04',
    domain='knn',
    notes=('experiment 02 for KNN feature selection.'
           'adding all cabin features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs02__cabin_fe04__knn'] = config_to_group(fs02__cabin_fe04__knn)
KNN_FS_CONFIGS['fs02__cabin_fe04__knn'] = config_to_group(fs02__cabin_fe04__knn)
# -0.004/-0.007

# fe03 achieve slightly positive result, despite lower f1, so it was carried over as a main pruning target

# fs03 - Ticket - KNN

fs03__ticket_fe09_full_context__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe03__deck_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='ticket_fe09_full_context',
    domain='knn',
    notes=('experiment 03 for KNN feature selection.'
           'adding ticket group size feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs03__ticket_fe09_full_context__knn'] = config_to_group(fs03__ticket_fe09_full_context__knn)
KNN_FS_CONFIGS['fs03__ticket_fe09_full_context__knn'] = config_to_group(fs03__ticket_fe09_full_context__knn)
# +0.002/+0.008

# considerably positive result, so fe09 was carried over.

# fs04 - family - KNN

fs04__family_fe01__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe03__deck_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['fe01__family_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='family_fe01',
    domain='knn',
    notes=('experiment 04 for KNN feature selection.'
           'adding family features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs04__family_fe01__knn'] = config_to_group(fs04__family_fe01__knn)
KNN_FS_CONFIGS['fs04__family_fe01__knn'] = config_to_group(fs04__family_fe01__knn)
#-0.005/-0.005

fs04__family_cb07__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe03__deck_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['cb07__family_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='family_cb07',
    domain='knn',
    notes=('experiment 04 for KNN feature selection.'
           'adding family features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs04__family_cb07__knn'] = config_to_group(fs04__family_cb07__knn)
KNN_FS_CONFIGS['fs04__family_cb07__knn'] = config_to_group(fs04__family_cb07__knn)
#-0.006/-0.006

# Due to their negative results, family features were not carried over.
# With that said, during one of the test while updating the feature engineering,
# I noticed that family together with tickets had positive result.
# It might be worth double checking this during the pruning phase

# pruning phase for KNN

pr01__removing_deck__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='pr01',
    feature_group='removing_deck',
    domain='knn',
    notes=('experiment 01 for KNN backward pruning.'
           'removing deck.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr01__removing_deck__knn'] = config_to_group(pr01__removing_deck__knn)
KNN_FS_CONFIGS['pr01__removing_deck__knn'] = config_to_group(pr01__removing_deck__knn)
# -0.007/-0.005

prsc__reading_family__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['fe01__family_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='prsc',
    feature_group='reading_family',
    domain='knn',
    notes=('sanity check experiment for KNN backward pruning.'
           'removing deck while readding family features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['prsc__reading_family__knn'] = config_to_group(prsc__reading_family__knn)
KNN_FS_CONFIGS['prsc__reading_family__knn'] = config_to_group(prsc__reading_family__knn)
#-0.005/-0.002

pr02__removing_fare_fe10__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe03__deck_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='removing_fare_fe10',
    domain='knn',
    notes=('experiment 02 for KNN backward pruning.'
           'removing fe10.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr02__removing_fare_fe10__knn'] = config_to_group(pr02__removing_fare_fe10__knn)
KNN_FS_CONFIGS['pr02__removing_fare_fe10__knn'] = config_to_group(pr02__removing_fare_fe10__knn)
#-0.006/-0.01



pr03__removing_title__knn = create_config(
    base_config=baseline_config['knn__raw'],
    patches = [
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch'],
        ALL_PATCHES['fe03__deck_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='pr03',
    feature_group='removing_title',
    domain='knn',
    notes=('experiment 03 for KNN backward pruning.'
           'removing title.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['pr03__removing_title__knn'] = config_to_group(pr03__removing_title__knn)
KNN_FS_CONFIGS['pr03__removing_title__knn'] = config_to_group(pr03__removing_title__knn)
#-0.009/-0.01


XGB_FS_CONFIGS = {}

# XGB Baseline

fs__baseline__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='xgb',
    notes=('Baseline configuration for XGB feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__xgb'] = config_to_group(fs__baseline__xgb)
XGB_FS_CONFIGS['fs__baseline__xgb'] = config_to_group(fs__baseline__xgb)

# fs01 - cabin - XGB

fs01__cabin_fe04__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe04__cabin_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='cabin_fe04',
    domain='xgb',
    notes=('Experiment 01 for XGB feature selection.'
           'adding fe04, cabin features.')
)

ALL_FS_CONFIGS['fs01__cabin_fe04__xgb'] = config_to_group(fs01__cabin_fe04__xgb)
XGB_FS_CONFIGS['fs01__cabin_fe04__xgb'] = config_to_group(fs01__cabin_fe04__xgb)
# -0.005/-0.006

# Negative result, will not carry over fe04
# fe04 was the only feature selection for cabin with positive results for XGB
# So no attempt to use another cabin feature was made

# fs02 - family - XGB

fs02__family_cb07__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb07__family_features_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='family_cb07',
    domain='xgb',
    notes=('Experiment 02 for XGB feature selection.'
           'adding cb07, family features.')
)

ALL_FS_CONFIGS['fs02__family_cb07__xgb'] = config_to_group(fs02__family_cb07__xgb)
XGB_FS_CONFIGS['fs02__family_cb07__xgb'] = config_to_group(fs02__family_cb07__xgb)
# -0.005/ -0.005
# Negative, not carried over
# Solo experiment for the same reason as cabin

fs03__ticket_fe09_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='ticket_fe09_fitted',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'adding fe09, ticket fitted features.')
)

ALL_FS_CONFIGS['fs03__ticket_fe09_fitted__xgb'] = config_to_group(fs03__ticket_fe09_fitted__xgb)
XGB_FS_CONFIGS['fs03__ticket_fe09_fitted__xgb'] = config_to_group(fs03__ticket_fe09_fitted__xgb)
# +0.001/+0.002

fs03__ticket_fe09_full_context__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='ticket_fe09_full_context',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'adding fe09, ticket full context features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs03__ticket_fe09_full_context__xgb'] = config_to_group(fs03__ticket_fe09_full_context__xgb)
XGB_FS_CONFIGS['fs03__ticket_fe09_full_context__xgb'] = config_to_group(fs03__ticket_fe09_full_context__xgb)
# 0/+0.005

# Both had positive results.
# Our priority's Accuracy, so fitted should be the best pick.
# but the rise in f1 from full context is more interesting, leading to a stalemate.

# will use ticket + family as tie breaker for next base.

# fs04 - ticket and family features - XGB

fs04__ticket_family_features_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb07__family_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='ticket_family_features_fitted',
    domain='xgb',
    notes=('Experiment 04 for XGB feature selection.'
           'Adding Ticket group size fitted and family features')
)


ALL_FS_CONFIGS['fs04__ticket_family_features_fitted__xgb'] = config_to_group(fs04__ticket_family_features_fitted__xgb)
XGB_FS_CONFIGS['fs04__ticket_family_features_fitted__xgb'] = config_to_group(fs04__ticket_family_features_fitted__xgb)
#0/+0.002

fs04__ticket_family_features_full_contest__xgb=create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['cb07__family_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='ticket_family_features_full_contest',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'Adding Ticket group size full context and family features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs04__ticket_family_features_full_contest__xgb'] = config_to_group(fs04__ticket_family_features_full_contest__xgb)
XGB_FS_CONFIGS['fs04__ticket_family_features_full_contest__xgb'] = config_to_group(fs04__ticket_family_features_full_contest__xgb)
# -0.007/-0.004

# Neither were positive, so they weren't carried over

#-------------------- old test and unused experiments -----------------

# fs04__ticket_family_fitted__xgb = create_config(
#     base_config=baseline_config['xgb__raw'],
#     patches = [
#         ALL_PATCHES['fe05__title_patch'],
#         ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
#         ALL_PATCHES['fe01__family_patch'],
#     ],
#     raw_features=RAW_FEATURES,
#     stage='fs04',
#     feature_group='ticket_family_features_fitted',
#     domain='xgb',
#     notes=('Experiment 04 for XGB feature selection.'
#            'Adding Ticket group size fitted and family ')
# )


# ALL_FS_CONFIGS['fs04__ticket_family_fitted__xgb'] = config_to_group(fs04__ticket_family_fitted__xgb)
# XGB_FS_CONFIGS['fs04__ticket_family_fitted__xgb'] = config_to_group(fs04__ticket_family_fitted__xgb)
# # -0.001/+0.001

# fs04__ticket_family_full_contest__xgb=create_config(
#     base_config=baseline_config['xgb__raw'],
#     patches = [
#         ALL_PATCHES['fe05__title_patch'],
#         ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
#         ALL_PATCHES['fe01__family_patch'],
#     ],
#     raw_features=RAW_FEATURES,
#     stage='fs04',
#     feature_group='ticket_family_features_full_contest',
#     domain='xgb',
#     notes=('Experiment 03 for XGB feature selection.'
#            'Adding Ticket group size full context and family.'),
#     pre_cv_scope="full_prediction_context"
# )

# ALL_FS_CONFIGS['fs04__ticket_family_full_contest__xgb'] = config_to_group(fs04__ticket_family_full_contest__xgb)
# XGB_FS_CONFIGS['fs04__ticket_family_full_contest__xgb'] = config_to_group(fs04__ticket_family_full_contest__xgb)
# # -0.003/-0.001



# fs04__ticket_family_fare_fitted__xgb = create_config(
#     base_config=baseline_config['xgb__raw'],
#     patches = [
#         ALL_PATCHES['fe05__title_patch'],
#         ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
#         ALL_PATCHES['fe01__family_patch'],
#         ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch']
#     ],
#     raw_features=RAW_FEATURES,
#     stage='fs04',
#     feature_group='ticket_family_fare_fitted',
#     domain='xgb',
#     notes=('Experiment 04 for XGB feature selection.'
#            'Adding Ticket group size fitted, family and fare ')
# )


# ALL_FS_CONFIGS['fs04__ticket_family_fare_fitted__xgb'] = config_to_group(fs04__ticket_family_fare_fitted__xgb)
# XGB_FS_CONFIGS['fs04__ticket_family_fare_fitted__xgb'] = config_to_group(fs04__ticket_family_fare_fitted__xgb)

# fs04__ticket_family_fare_full_contest__xgb=create_config(
#     base_config=baseline_config['xgb__raw'],
#     patches = [
#         ALL_PATCHES['fe05__title_patch'],
#         ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
#         ALL_PATCHES['fe01__family_patch'],
#         ALL_PATCHES['cb09__fare_per_family_and_ticket_full_context_patch']
#     ],
#     raw_features=RAW_FEATURES,
#     stage='fs04',
#     feature_group='ticket_family_fare_full',
#     domain='xgb',
#     notes=('Experiment 04 for XGB feature selection.'
#            'Adding Ticket group size full context, family and fare.'),
#     pre_cv_scope="full_prediction_context"
# )

# ALL_FS_CONFIGS['fs04__ticket_family_fare_full_contest__xgb'] = config_to_group(fs04__ticket_family_fare_full_contest__xgb)
# XGB_FS_CONFIGS['fs04__ticket_family_fare_full_contest__xgb'] = config_to_group(fs04__ticket_family_fare_full_contest__xgb)


# fs05 - fare - XGB

fs05__fare_cb05_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_fitted_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_cb05_fitted',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding cb05, fare and fare per ticket fitted.')
)

ALL_FS_CONFIGS['fs05__fare_cb05_fitted__xgb'] = config_to_group(fs05__fare_cb05_fitted__xgb)
XGB_FS_CONFIGS['fs05__fare_cb05_fitted__xgb'] = config_to_group(fs05__fare_cb05_fitted__xgb)
# -0.013/-0.013

fs05__fare_cb05_full_context__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['cb05__fare_and_fare_per_ticket_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_cb05_full_context',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding cb05, fare and fare per ticket full context.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs05__fare_cb05_full_context__xgb'] = config_to_group(fs05__fare_cb05_full_context__xgb)
XGB_FS_CONFIGS['fs05__fare_cb05_full_context__xgb'] = config_to_group(fs05__fare_cb05_full_context__xgb)
# -0.002/+0.003

fs05__fare_fe08_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['fe08__fare_per_family_member_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_fe08_fitted',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding fe08, fare per family, fitted.')
)


ALL_FS_CONFIGS['fs05__fare_fe08_fitted__xgb'] = config_to_group(fs05__fare_fe08_fitted__xgb)
XGB_FS_CONFIGS['fs05__fare_fe08_fitted__xgb'] = config_to_group(fs05__fare_fe08_fitted__xgb)
# +0.001/+0.003

fs05__fare_fe08_full_context__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['fe08__fare_per_family_member_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_fe08_full_context',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding fe08, fare per family, full context.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs05__fare_fe08_full_context__xgb'] = config_to_group(fs05__fare_fe08_full_context__xgb)
XGB_FS_CONFIGS['fs05__fare_fe08_full_context__xgb'] = config_to_group(fs05__fare_fe08_full_context__xgb)
# -0.002/+0.001

fs05__fare_fe10_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_fitted_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_fe10_fitted',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding fe10, fare per ticket, fitted.')
)


ALL_FS_CONFIGS['fs05__fare_fe10_fitted__xgb'] = config_to_group(fs05__fare_fe10_fitted__xgb)
XGB_FS_CONFIGS['fs05__fare_fe10_fitted__xgb'] = config_to_group(fs05__fare_fe10_fitted__xgb)
# 0/+0.002

fs05__fare_fe10_full_context__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['fe10__fare_per_ticket_member_full_context_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='fare_fe10_full_context',
    domain='xgb',
    notes=('Experiment 05 for XGB feature selection.'
           'adding fe10, fare per ticket, full context.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs05__fare_fe10_full_context__xgb'] = config_to_group(fs05__fare_fe10_full_context__xgb)
XGB_FS_CONFIGS['fs05__fare_fe10_full_context__xgb'] = config_to_group(fs05__fare_fe10_full_context__xgb)
# -0.005/ -0.003

# None of the results were compeling when compared to fe03 with it's +0.001/0.002 and 0/+0.005
# fare fe08 fitted did achieve a 0/+0.001 on top of fe03 fitted, but that gain didnt felt worth carring foward

# For now fs03 fitted will be carried over provisionally while full context is kept as alternative

# fs06 - age - XGB

fs06__age_fe06_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['fe06__age_imputation_title_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='age_fe06_fitted',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'adding fe06, age imputation features.')
)

ALL_FS_CONFIGS['fs06__age_fe06_fitted__xgb'] = config_to_group(fs06__age_fe06_fitted__xgb)
XGB_FS_CONFIGS['fs06__age_fe06_fitted__xgb'] = config_to_group(fs06__age_fe06_fitted__xgb)
# -0.005/-0.007

fs06__age_cb01_fitted__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb01__age_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='age_cb01_fitted',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'adding fe06, age imputation features.')
)

ALL_FS_CONFIGS['fs06__age_cb01_fitted__xgb'] = config_to_group(fs06__age_cb01_fitted__xgb)
XGB_FS_CONFIGS['fs06__age_cb01_fitted__xgb'] = config_to_group(fs06__age_cb01_fitted__xgb)
# +0.001/+0.002

# cb01 was the one chosen to be carried over

# Pruning phase for XGB

pr01__removing_ticket__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb01__age_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr01',
    feature_group='removing_ticket',
    domain='xgb',
    notes=('Experiment 01 for XGB backward pruning.'
           'removing ticket.')
)

ALL_FS_CONFIGS['pr01__removing_ticket__xgb'] = config_to_group(pr01__removing_ticket__xgb)
XGB_FS_CONFIGS['pr01__removing_ticket__xgb'] = config_to_group(pr01__removing_ticket__xgb)
#-0.005/-0.008

pr02__removing_title__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
        ALL_PATCHES['cb01__age_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='pr02',
    feature_group='removing_title',
    domain='xgb',
    notes=('Experiment 02 for XGB backward pruning.'
           'removing title.')
)

ALL_FS_CONFIGS['pr02__removing_title__xgb'] = config_to_group(pr02__removing_title__xgb)
XGB_FS_CONFIGS['pr02__removing_title__xgb'] = config_to_group(pr02__removing_title__xgb)
# -0.012/-0.018

prsc__age_cb01_full_context__xgb = create_config(
    base_config=baseline_config['xgb__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
        ALL_PATCHES['cb01__age_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs06',
    feature_group='age_cb01_full_context',
    domain='xgb',
    notes=('Experiment 03 for XGB feature selection.'
           'adding fe06, age imputation features.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['prsc__age_cb01_full_context__xgb'] = config_to_group(prsc__age_cb01_full_context__xgb)
XGB_FS_CONFIGS['prsc__age_cb01_full_context__xgb'] = config_to_group(prsc__age_cb01_full_context__xgb)
# -0.004/-0.002


# Feature selection - Extra Tree

EXTRA_TREES_FS_CONFIGS = {}

# Extra trees Baseline

fs__baseline__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch']
    ],
    raw_features=RAW_FEATURES,
    stage='fs',
    feature_group='baseline',
    domain='extra_trees',
    notes=('Baseline configuration for Extra Trees feature selection.'
           'Starts from the raw model configuration with Title added.')
)

ALL_FS_CONFIGS['fs__baseline__extra_trees'] = config_to_group(fs__baseline__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs__baseline__extra_trees'] = config_to_group(fs__baseline__extra_trees)

# Fs01 - age - Extra trees

fs01__age_cb02__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb02',
    domain='extra_trees',
    notes=('Experiment 01 for Extra Trees feature selection.'
           'adding cb02, age imputed by title and bins.')
)

ALL_FS_CONFIGS['fs01__age_cb02__extra_trees'] = config_to_group(fs01__age_cb02__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs01__age_cb02__extra_trees'] = config_to_group(fs01__age_cb02__extra_trees)
# +0.002/+0.002



fs01__age_cb03__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb03__age_imputed_title_pclass_and_bins_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs01',
    feature_group='age_cb03',
    domain='extra_trees',
    notes=('Experiment 01 for Extra Trees feature selection.'
           'adding cb03, age imputed by title,pclass and bins.')
)

ALL_FS_CONFIGS['fs01__age_cb03__extra_trees'] = config_to_group(fs01__age_cb03__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs01__age_cb03__extra_trees'] = config_to_group(fs01__age_cb03__extra_trees)
# +0.002/+0.002

# Since cb02 was originally the best result, it will be the one carried foward, while cb02 will remain as alternative

# fs02 - fare - Extra Trees

fs02__fare_cb09_full_context__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_full_context_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='fare_cb09_full_context',
    domain='extra_trees',
    notes=('Experiment 02 for Extra Trees feature selection.'
           'adding cb09, Fare per family and ticket, full context.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs02__fare_cb09_full_context__extra_trees'] = config_to_group(fs02__fare_cb09_full_context__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs02__fare_cb09_full_context__extra_trees'] = config_to_group(fs02__fare_cb09_full_context__extra_trees)
#+0.001/+0.002

fs02__fare_cb09_fitted__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs02',
    feature_group='fare_cb09_fitted',
    domain='extra_trees',
    notes=('Experiment 02 for Extra Trees feature selection.'
           'adding cb09, Fare per family and ticket, fitted.'),
)

ALL_FS_CONFIGS['fs02__fare_cb09_fitted__extra_trees'] = config_to_group(fs02__fare_cb09_fitted__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs02__fare_cb09_fitted__extra_trees'] = config_to_group(fs02__fare_cb09_fitted__extra_trees)
#+0.004/+0.007

# Due to it's clear positive results, cb09 was the one carried over

# Fs03 - cabin - Extra Tree

fs03__cabin_fe02__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
        ALL_PATCHES['fe02__has_cabin_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='cabin_fe02',
    domain='extra_trees',
    notes=('Experiment 03 for Extra Trees feature selection.'
           'adding fe02, has cabin feature.'),
)

ALL_FS_CONFIGS['fs03__cabin_fe02__extra_trees'] = config_to_group(fs03__cabin_fe02__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs03__cabin_fe02__extra_trees'] = config_to_group(fs03__cabin_fe02__extra_trees)
# -0.007/-0.011

fs03__cabin_fe04__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs03',
    feature_group='cabin_fe04',
    domain='extra_trees',
    notes=('Experiment 03 for Extra Trees feature selection.'
           'adding fe04, all cabin feature.'),
)

ALL_FS_CONFIGS['fs03__cabin_fe04__extra_trees'] = config_to_group(fs03__cabin_fe04__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs03__cabin_fe04__extra_trees'] = config_to_group(fs03__cabin_fe04__extra_trees)
#-0.009/-0.013

# fe04 - family - Extra Trees

fs04__family_fe01__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
        ALL_PATCHES['fe01__family_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='family_fe01',
    domain='extra_trees',
    notes=('Experiment 04 for Extra Trees feature selection.'
           'adding fe01, family feature.'),
)

ALL_FS_CONFIGS['fs04__family_fe01__extra_trees'] = config_to_group(fs04__family_fe01__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs04__family_fe01__extra_trees'] = config_to_group(fs04__family_fe01__extra_trees)
#0/-0.002

fs04__family_cb07__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
        ALL_PATCHES['cb07__family_features_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs04',
    feature_group='family_cb07',
    domain='extra_trees',
    notes=('Experiment 04 for Extra Trees feature selection.'
           'adding cb07, all family feature.'),
)

ALL_FS_CONFIGS['fs04__family_cb07__extra_trees'] = config_to_group(fs04__family_cb07__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs04__family_cb07__extra_trees'] = config_to_group(fs04__family_cb07__extra_trees)
#0/0

# Due to fs04 being the only negative, even if neutral, it was the one carried forward

# Fe05 - ticket - Extra trees

fs05__family_fe09_fitted__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_fitted_patch'],
        ALL_PATCHES['cb07__family_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_fitted_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='family_fe09_fitted',
    domain='extra_trees',
    notes=('Experiment 04 for Extra Trees feature selection.'
           'adding fe09, ticket group size fitted feature.'),
)

ALL_FS_CONFIGS['fs05__family_fe09_fitted__extra_trees'] = config_to_group(fs05__family_fe09_fitted__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs05__family_fe09_fitted__extra_trees'] = config_to_group(fs05__family_fe09_fitted__extra_trees)
#-0.002/-0.003

fs05__family_fe09_full_context__extra_trees = create_config(
    base_config=baseline_config['extra_trees__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['cb02__age_imputed_title_and_bins_patch'],
        ALL_PATCHES['cb09__fare_per_family_and_ticket_full_context_patch'],
        ALL_PATCHES['cb07__family_features_patch'],
        ALL_PATCHES['fe09__ticket_group_size_full_context_patch'],
    ],
    raw_features=RAW_FEATURES,
    stage='fs05',
    feature_group='family_fe09_full_context',
    domain='extra_trees',
    notes=('Experiment 04 for Extra Trees feature selection.'
           'adding fe09, ticket group size full context feature.'),
    pre_cv_scope="full_prediction_context"
)

ALL_FS_CONFIGS['fs05__family_fe09_full_context__extra_trees'] = config_to_group(fs05__family_fe09_full_context__extra_trees)
EXTRA_TREES_FS_CONFIGS['fs05__family_fe09_full_context__extra_trees'] = config_to_group(fs05__family_fe09_full_context__extra_trees)
# +0.003/+0.003