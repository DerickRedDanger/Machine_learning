from titanic_ml.common.utils.sklearn_compatible.Juxtaposition import JuxtapositionTransformer

def sex_pclass_transform():
    return JuxtapositionTransformer(
        source_cols=['Sex', 'Pclass'],
        output_col='Sex_Pclass',)

ADD_SEX_PCLASS = {
    "id": "sex_pclass",
    "transformer": sex_pclass_transform,
    "requires": ["Sex", "Pclass"],
    "produces": ["Sex_Pclass"],
}