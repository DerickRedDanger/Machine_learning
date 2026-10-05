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

removing_fare_patch={
        "remove": {
                "preprocessing": {
                    "numeric_features": [
                        "Fare",
                    ],
                },
            },
}

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

removing_raw_age={
        "remove": {
                "preprocessing": {
                    "numeric_features": [
                        "Age",
                    ],
                },
            },
}

pr08__removing_all_age_related__logreg=create_config(
base_config=baseline_config['logreg__raw'],
    patches = [
        ALL_PATCHES['fe05__title_patch'],
        ALL_PATCHES['fe04__cabin_features_patch'],
        removing_fare_patch,
        removing_raw_age,
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
        removing_raw_age,
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
        removing_raw_age,
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