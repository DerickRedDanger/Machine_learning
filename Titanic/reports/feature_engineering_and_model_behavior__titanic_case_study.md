# Feature Engineering and Model Behavior: A Titanic Case Study

This project serves both as a machine-learning study and as a demonstration of
a reproducible experimentation workflow.

Using the Titanic dataset as a controlled environment, the project investigates
how different feature-engineering strategies affect several classical machine-
learning models. Rather than focusing only on achieving the highest possible
score, the study aims to understand why particular feature representations help
some models, harm others, or become redundant when combined.

The process covers dataset exploration, feature engineering, controlled
experiments, model-specific feature selection, and finally the construction and
tuning of the strongest resulting models.

Experiment definitions are maintained in
`titanic_ml/common/experiments/config.py`, while experiment results and the
configurations used to produce them are stored under `titanic_ml/results/`.

## Current leaderboard

Leaderboard
| experiment                                   | model_name    |   test_accuracy_mean |   test_f1_mean |
|:---------------------------------------------|:--------------|---------------------:|---------------:|
| fe05__title__xgb                             | xgb           |                0.836 |          0.772 |
| fe05__title__svc                             | svc           |                0.834 |          0.771 |
| fe11__age_bin__random_forest                 | random_forest |                0.833 |          0.759 |
| fe09__ticket_group_size__svc                 | svc           |                0.832 |          0.77  |
| fe05__title__random_forest                   | random_forest |                0.832 |          0.768 |
| fe04__cabin_features__xgb                    | xgb           |                0.832 |          0.767 |
| cb02__age_imputed_title_and_bins__svc        | svc           |                0.831 |          0.767 |
| cb03__age_imputed_title_Pclass_and_bins__svc | svc           |                0.831 |          0.767 |
| fe06__age_imputation_title__svc              | svc           |                0.829 |          0.764 |
| fe07__age_imputation_title_pclass__svc       | svc           |                0.829 |          0.764 |

## Dataset exploration (EDA)

### Dataset overview

- 891 passengers
- 12 features (including target)
- Missing values concentrated in Age and Cabin
- No duplicate rows
- Binary Target (61.6% dead, 38.4% survived)


<details>
<summary>Reference:</summary>

- **Dataframe Health**:

|                      |                                                                                                                      |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------|
| rows                 | 891                                                                                                                  |
| columns              | 12                                                                                                                   |
| duplicate_rows       | 0                                                                                                                    |
| duplicate_%          | 0.0                                                                                                                  |
| rows_with_missing    | 708                                                                                                                  |
| rows_with_missing_%  | 79.46                                                                                                                |
| total_missing_values | 866                                                                                                                  |
| dataFrame_columns    | ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] |
| memory_usage         | Total memory usage: 315.03 KB                                                        
|                      |                                                                                                                      |

- **Dataframe summary**:

|             | dtype   |   non_null_count |   missing_count |   missing_% |   unique |   cardinality_% | cardinality_label   | top_value           |   dominance_% | dominance_label   | bottom_value             |
|:------------|:--------|-----------------:|----------------:|------------:|---------:|----------------:|:--------------------|:--------------------|--------------:|:------------------|:-------------------------|
| PassengerId | int64   |              891 |               0 |        0    |      891 |          100    | potential_id        | 891                 |      0.112233 | balanced          | 12                       |
| Survived    | int64   |              891 |               0 |        0    |        2 |            0.22 | low_cardinality     | 0                   |     61.6162   | some_dominance    | 1                        |
| Pclass      | int64   |              891 |               0 |        0    |        3 |            0.34 | low_cardinality     | 3                   |     55.1066   | some_dominance    | 2                        |
| Name        | object  |              891 |               0 |        0    |      891 |          100    | potential_id        | Dooley, Mr. Patrick |      0.112233 | balanced          | Bonnell, Miss. Elizabeth |
| Sex         | object  |              891 |               0 |        0    |        2 |            0.22 | low_cardinality     | male                |     64.7587   | some_dominance    | female                   |
| Age         | float64 |              714 |             177 |       19.87 |       89 |            9.99 | high_cardinality    | MISSING             |     19.8653   | balanced          | 74.0                     |
| SibSp       | int64   |              891 |               0 |        0    |        7 |            0.79 | low_cardinality     | 0                   |     68.2379   | some_dominance    | 5                        |
| Parch       | int64   |              891 |               0 |        0    |        7 |            0.79 | low_cardinality     | 0                   |     76.0943   | some_dominance    | 6                        |
| Ticket      | object  |              891 |               0 |        0    |      681 |           76.43 | high_cardinality    | 347082              |      0.785634 | balanced          | STON/O2. 3101282         |
| Fare        | float64 |              891 |               0 |        0    |      248 |           27.83 | high_cardinality    | 8.05                |      4.82604  | balanced          | 10.5167                  |
| Cabin       | object  |              204 |             687 |       77.1  |      148 |           16.61 | high_cardinality    | MISSING             |     77.1044   | some_dominance    | C148                     |
| Embarked    | object  |              889 |               2 |        0.22 |        4 |            0.45 | low_cardinality     | S                   |     72.2783   | some_dominance    | MISSING                  |

- **Sample from the dataframe**:

|   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
|             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
|             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
|             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
|            48 |          1 |        3 | O'Driscoll, Miss. Bridget                           | female |   nan |       0 |       0 | 14311            |  7.75   | nan     | Q          |
|           392 |          1 |        3 | Jansson, Mr. Carl Olof                              | male   |    21 |       0 |       0 | 350034           |  7.7958 | nan     | S          |
|           647 |          0 |        3 | Cor, Mr. Liudevit                                   | male   |    19 |       0 |       0 | 349231           |  7.8958 | nan     | S          |
|           707 |          1 |        2 | Kelly, Mrs. Florence "Fannie"                       | female |    45 |       0 |       0 | 223596           | 13.5    | nan     | S          |
|           889 |          0 |        3 | Johnston, Miss. Catherine Helen "Carrie"            | female |   nan |       1 |       2 | W./C. 6607       | 23.45   | nan     | S          |
|           890 |          1 |        1 | Behr, Mr. Karl Howell                               | male   |    26 |       0 |       0 | 111369           | 30      | C148    | C          |
|           891 |          0 |        3 | Dooley, Mr. Patrick                                 | male   |    32 |       0 |       0 | 370376           |  7.75   | nan     | Q          |

</details>

---

### Numerical features



**General observations.**

- **Outliers:**
    - Age: 11 (1.54%)
    - SibSp: 46 (5.16%)
    - Parch: 213 (23.91%)
    - Fare: 116 (13.2%)

- **Distributions:**
    - Pclass: Moderate skew and normal Tail
    - Age: Low skew and normal tail
    - SibSp: High skew and heavy tail
    - Parch: High skew and heavy tail
    - Fare: High skew and heavy tail

- **Correlation with the target variable:**

    |             |   Survived |
    |:------------|-----------:|
    | Pclass      |     -0.338 |
    | Fare        |      0.257 |
    | Parch       |      0.082 |
    | Age         |     -0.077 |
    | SibSp       |     -0.035 |
    | PassengerId |     -0.005 |

    - Pclass has the strongest linear relationship with survival. Were passengers in higher classes better positioned to reach the lifeboats, or did they receive preferential treatment?

    - Fare shows the second strongest linear relationship with survival after Pclass.

    - Age has a surprisingly weak linear correlation. Would a nonlinear transformation capture its information better? Or perhaps grouping?

- **Preprocessing considerations**:

    - Fare contains many extreme values, likely to benefit from scaling or transformation.
    - Age has 20% missing values, too many for a median imputation to suffice. will likely require a more informative imputation to make the most of it.
    - PassengerId is purely an identifier, thus unlikely to contribute to predictions.

- **Feature engineering ideas**:


    - Some third-class passengers paid more than first-class passengers. This suggests Fare may represent the total price paid by a travelling group rather than an individual passenger.



<details>
<summary>Reference:</summary>

---

**Numerical summary:**

 - **PassengerId:**

|                      |                                                                                                                      |
|:------------------------|:--------------------|
| count                   | 891.0               |
| mean                    | 446.0               |
| std                     | 257.3538420152301   |
| min                     | 1.0                 |
| 25%                     | 223.5               |
| 50%                     | 446.0               |
| 75%                     | 668.5               |
| max                     | 891.0               |
| missing_count           | 0                   |
| missing_%               | 0.0                 |
| outlier_count           | 0                   |
| outlier_%               | 0.0                 |
| skew                    | 0.0                 |
| skew_classification     | low skew            |
| kurtosis                | -1.1999999999999997 |
| kurtosis_classification | normal_tails        |

- **Pclass:**

|                      |                                                                                                                      |
|:------------------------|:--------------------|
| count                   | 891.0               |
| mean                    | 2.308641975308642   |
| std                     | 0.836071240977049   |
| min                     | 1.0                 |
| 25%                     | 2.0                 |
| 50%                     | 3.0                 |
| 75%                     | 3.0                 |
| max                     | 3.0                 |
| missing_count           | 0                   |
| missing_%               | 0.0                 |
| outlier_count           | 0                   |
| outlier_%               | 0.0                 |
| skew                    | -0.6305479068752845 |
| skew_classification     | moderate skew       |
| kurtosis                | -1.2800149715782825 |
| kurtosis_classification | normal_tails        |

- **Age:**

|                      |                                                                                                                      |
|:------------------------|:--------------------|
| count                   | 714.0               |
| mean                    | 29.69911764705882   |
| std                     | 14.526497332334042  |
| min                     | 0.42                |
| 25%                     | 20.125              |
| 50%                     | 28.0                |
| 75%                     | 38.0                |
| max                     | 80.0                |
| missing_count           | 177                 |
| missing_%               | 19.87               |
| outlier_count           | 11                  |
| outlier_%               | 1.54                |
| skew                    | 0.38910778230082693 |
| skew_classification     | low skew            |
| kurtosis                | 0.1782741536421022  |
| kurtosis_classification | normal_tails        |

- **SibSp:**

|                      |                                                                                                                      |
|:------------------------|:-------------------|
| count                   | 891.0              |
| mean                    | 0.5230078563411896 |
| std                     | 1.1027434322934317 |
| min                     | 0.0                |
| 25%                     | 0.0                |
| 50%                     | 0.0                |
| 75%                     | 1.0                |
| max                     | 8.0                |
| missing_count           | 0                  |
| missing_%               | 0.0                |
| outlier_count           | 46                 |
| outlier_%               | 5.16               |
| skew                    | 3.6953517271630565 |
| skew_classification     | high skew          |
| kurtosis                | 17.880419726645968 |
| kurtosis_classification | heavy_tails        |

- **Parch:**

|                      |                                                                                                                      |
|:------------------------|:--------------------|
| count                   | 891.0               |
| mean                    | 0.38159371492704824 |
| std                     | 0.8060572211299483  |
| min                     | 0.0                 |
| 25%                     | 0.0                 |
| 50%                     | 0.0                 |
| 75%                     | 0.0                 |
| max                     | 6.0                 |
| missing_count           | 0                   |
| missing_%               | 0.0                 |
| outlier_count           | 213                 |
| outlier_%               | 23.91               |
| skew                    | 2.7491170471010933  |
| skew_classification     | high skew           |
| kurtosis                | 9.778125179021648   |
| kurtosis_classification | heavy_tails         |

- **Fare:**

|                      |                                                                                                                      |
|:------------------------|:-------------------|
| count                   | 891.0              |
| mean                    | 32.204207968574636 |
| std                     | 49.6934285971809   |
| min                     | 0.0                |
| 25%                     | 7.9104             |
| 50%                     | 14.4542            |
| 75%                     | 31.0               |
| max                     | 512.3292           |
| missing_count           | 0                  |
| missing_%               | 0.0                |
| outlier_count           | 116                |
| outlier_%               | 13.02              |
| skew                    | 4.787316519674893  |
| skew_classification     | high skew          |
| kurtosis                | 33.39814088089868  |
| kurtosis_classification | heavy_tails        |


- **Correlation matrix:**

|             |   PassengerId |   Survived |   Pclass |    Age |   SibSp |   Parch |   Fare |
|:------------|--------------:|-----------:|---------:|-------:|--------:|--------:|-------:|
| PassengerId |         1     |     -0.005 |   -0.035 |  0.037 |  -0.058 |  -0.002 |  0.013 |
| Survived    |        -0.005 |      1     |   -0.338 | -0.077 |  -0.035 |   0.082 |  0.257 |
| Pclass      |        -0.035 |     -0.338 |    1     | -0.369 |   0.083 |   0.018 | -0.549 |
| Age         |         0.037 |     -0.077 |   -0.369 |  1     |  -0.308 |  -0.189 |  0.096 |
| SibSp       |        -0.058 |     -0.035 |    0.083 | -0.308 |   1     |   0.415 |  0.16  |
| Parch       |        -0.002 |      0.082 |    0.018 | -0.189 |   0.415 |   1     |  0.216 |
| Fare        |         0.013 |      0.257 |   -0.549 |  0.096 |   0.16  |   0.216 |  1     |



</details>

---

### Categorical features

**General observations.**

- **Cardinality**
    - Name: 100% - Potential Id
    - Sex: 0.22% - Low cardinality
    - Ticket: 76.43% - Higt cardinality
    - Cabin: 16.61% - High cardinality
    - Embarked: 0.45% - Low cardinality

- **Rare categories**
    - Name: No rare categories - too many unique values
    - Sex: No rare categories
    - Ticket: No rare categories - too many unique values
    - Cabin: No rare categories - too many unique values
    - Embarked: MISSING - 2 (0.22%)

- **Missing values**
    - Name: 0
    - Sex: 0
    - Ticket: 0
    - Cabin: 687 (77.1%)
    - Embarked: 2 (0.22%)

- **Feature engineering ideas**:
    - Name contains titles and family names.
    - Cabin is missing most of its values, could this be a signal in itself?
    - All cabins starts with a letter, position on the ship? deck?
    - Many passengers have the same ticket. Shared tickets? are they traveling in groups?


<details>
<summary>Reference:</summary>

**Categorical summary**:

- **Sex:**

| Sex    |   count |   percent |   Survived_0_% |   Survived_1_% |
|:-------|--------:|----------:|---------------:|---------------:|
| male   |     577 |     64.76 |          81.11 |          18.89 |
| female |     314 |     35.24 |          25.8  |          74.2  |

- **Embarked:**

| Embarked   |   count |   percent |   Survived_0_% |   Survived_1_% |
|:-----------|--------:|----------:|---------------:|---------------:|
| S          |     644 |     72.28 |          66.3  |          33.7  |
| C          |     168 |     18.86 |          44.64 |          55.36 |
| Q          |      77 |      8.64 |          61.04 |          38.96 |
| MISSING    |       2 |      0.22 |           0    |         100    |


**Categorical samples**:

- **Name:**

|                          |   count |   percent |   Survived_0_% |   Survived_1_% |
|:-------------------------|--------:|----------:|---------------:|---------------:|
| Dooley, Mr. Patrick      |       1 |      0.11 |            100 |              0 |
| Braund, Mr. Owen Harris  |       1 |      0.11 |            100 |              0 |
| Masselmani, Mrs. Fatima  |       1 |      0.11 |              0 |            100 |
| Moran, Mr. James         |       1 |      0.11 |            100 |              0 |
| Bonnell, Miss. Elizabeth |       1 |      0.11 |              0 |            100 |

- **Ticket:**

|                  |   count |   percent |   Survived_0_% |   Survived_1_% |
|:-----------------|--------:|----------:|---------------:|---------------:|
| 347082           |       7 |      0.79 |         100    |           0    |
| 1601             |       7 |      0.79 |          28.57 |          71.43 |
| 345765           |       1 |      0.11 |         100    |           0    |
| 382652           |       5 |      0.56 |         100    |           0    |
| STON/O2. 3101282 |       1 |      0.11 |           0    |         100    |

- **Cabin:**

|         |   count |   percent |   Survived_0_% |   Survived_1_% |
|:--------|--------:|----------:|---------------:|---------------:|
| MISSING |     687 |     77.1  |          70.01 |          29.99 |
| B69     |       1 |      0.11 |           0    |         100    |
| E101    |       3 |      0.34 |           0    |         100    |
| G6      |       4 |      0.45 |          50    |          50    |
| C148    |       1 |      0.11 |           0    |         100    |

</details>

---

### Initial preprocessing plan

- **drop:**
    - PassengerId
- **Impute:**
    - Age
    - Embarked
- **Engineer:**
    - Title
    - Deck
    - Family
    - Ticket_group
    - Age_bin
- **Investigate:**
    - Name 

---

### Initial hypotheses

#### Parch and SibSp
- Combine SibSp and Parch into family

#### Cabin
- Investigate if missingness is a signal
- Extract Deck

#### Name
- Investigate title
- ~~Investigate family name~~ - Discarded. SibSp and Parch already provide sufficient information

#### Age
- Investigate better imputation

#### Ticket
- Investigate ticket groups

#### Fare
- investigate effects of scaling and transformations
- Fare seems to be the price paid per ticket. Divide by family and ticket groups to find individual fare

## Baseline experiment

Before investigating feature engineering, a common baseline was established for
all evaluated models. This provides a fixed reference against which later
experiments can be compared.

### Baseline features

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

`PassengerId`, `Name`, `Ticket`, and `Cabin` were not used directly in the
baseline. Although several of them later provided useful engineered features,
their raw representations were either identifier-like, highly sparse, or
high-cardinality.

### Models evaluated

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree
- Random Forest
- Extra Trees
- XGBoost

The baseline model configurations are defined in
`titanic_ml/common/experiments/config.py`. The exact configurations used for
executed experiments are also preserved in
`titanic_ml/results/experiments_used_config.json`.

Except for the final model-selection and tuning stages, all experiments use the
same baseline model parameters. Experimental changes are restricted to feature
engineering and preprocessing so that differences in performance can be
attributed as directly as possible to changes in the data representation.

### Result

| model_name    | accuracy      | f1            |
|:--------------|:--------------|:--------------|
| logreg        | 0.786 ± 0.018 | 0.713 ± 0.026 |
| knn           | 0.809 ± 0.021 | 0.742 ± 0.026 |
| svc           | 0.827 ± 0.015 | 0.76 ± 0.026  |
| decision_tree | 0.803 ± 0.023 | 0.702 ± 0.055 |
| random_forest | 0.822 ± 0.02  | 0.744 ± 0.041 |
| extra_trees   | 0.804 ± 0.012 | 0.721 ± 0.025 |
| xgb           | 0.826 ± 0.025 | 0.758 ± 0.041 |

### Observations

SVC and XGBoost produced the strongest initial baselines, followed closely by Random Forest.

Logistic Regression began with the weakest accuracy among the evaluated models, but its simplicity makes it a useful reference model. Its later response to explicit feature engineering also makes it particularly informative when studying how feature representation affects simpler linear models.
---

## Feature Investigation

Following the initial hypotheses, each original feature was investigated individually. Each new feature created from there were grouped inside their progenitor feature, together with each of their experiments.

### Parch and SibSp

#### Hypothesis

SibSp and Parch describe complementary aspects of family composition.

Combining them into FamilySize may strengthen the information available to the model.

A binary IsAlone feature may further highlight passengers travelling alone.


#### Experiments performed:

#### Fe01__family

Experiment testing the effects of the creation of the features Familysize and IsAlone, and whether they effectively replace SibSp and Parch.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.009
  - decision_tree: test_accuracy_mean: 0.003
    - Secondary gains:
      - test_f1_mean: 0.01

##### Conclusion

Small impact overall. LogReg improved slightly.
Apparently, most models already manage to extract the information from SibSp and Parch, making this feature engineering mostly unescessary.

One exception being LogReg, which had a small gain in accuracy. Was it unable to make the most of these features on its own? or it just gained from receiving processed information?

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe01__family    | logreg        |                          0.786 |                        0.795 |                      0.009 |                    0.713 |                  0.721 |                0.008 |
| baseline__raw     | fe01__family    | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.737 |               -0.005 |
| baseline__raw     | fe01__family    | svc           |                          0.827 |                        0.826 |                     -0.001 |                    0.76  |                  0.756 |               -0.004 |
| baseline__raw     | fe01__family    | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.712 |                0.01  |
| baseline__raw     | fe01__family    | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.735 |               -0.009 |
| baseline__raw     | fe01__family    | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.726 |                0.005 |
| baseline__raw     | fe01__family    | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.756 |               -0.002 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe01__family    |                     0.000428571 |                         -0.006 |                          0.009 |               0.000428571 |                   -0.009 |                     0.01 |

</details>

#### cb07__family_features

This experiment keeps both the engineered and original family to see if they complimet each other.

<details>
<summary>Conclusion</summary>


#### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.004
  - decision_tree: test_accuracy_mean: 0.004
    - Secondary gains:
      - test_f1_mean: 0.012
  - extra_trees: test_accuracy_mean: 0.003
  - xgb: test_accuracy_mean: 0.003


#### Conclusion

The results suggest that FamilySize and IsAlone contain useful information, but
not enough to replace SibSp and Parch. Instead, the engineered features appear
to complement the original representation for some models.

Logistic Regression remained one of the main beneficiaries, while Decision Tree
and XGBoost also showed small improvements. Random Forest recovered the losses
observed in fe01, suggesting that replacing the original features discarded
information, whereas keeping both representations preserved it.

Overall, the combination produced small but broadly positive results, indicating
that FamilySize and IsAlone are better viewed as complementary features than as
replacements for SibSp and Parch.

</details>

<details>
<summary>Experiment details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group         | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb07__family_features | logreg        |                          0.786 |                        0.79  |                      0.004 |                    0.713 |                  0.719 |                0.006 |
| baseline__raw     | cb07__family_features | knn           |                          0.809 |                        0.801 |                     -0.008 |                    0.742 |                  0.733 |               -0.009 |
| baseline__raw     | cb07__family_features | svc           |                          0.827 |                        0.828 |                      0.001 |                    0.76  |                  0.763 |                0.003 |
| baseline__raw     | cb07__family_features | decision_tree |                          0.803 |                        0.807 |                      0.004 |                    0.702 |                  0.714 |                0.012 |
| baseline__raw     | cb07__family_features | random_forest |                          0.822 |                        0.824 |                      0.002 |                    0.744 |                  0.752 |                0.008 |
| baseline__raw     | cb07__family_features | extra_trees   |                          0.804 |                        0.807 |                      0.003 |                    0.721 |                  0.727 |                0.006 |
| baseline__raw     | cb07__family_features | xgb           |                          0.826 |                        0.829 |                      0.003 |                    0.758 |                  0.763 |                0.005 |

#### Summary

| compare_group         |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb07__family_features |                      0.00128571 |                         -0.008 |                          0.004 |                0.00442857 |                   -0.009 |                    0.012 |

</details>

#### Overall conclusion

Overall, the experiments indicate that the relationship between SibSp and Parchis already well exploited by most models.

Constructing FamilySize and IsAlone alone rarely improves performance enough to justify replacing the original variables. However, keeping both the engineered and original representations often produces small positive gains, suggesting that the engineered features provide complementary information rather than strictly redundant information.

The strongest and most consistent improvements were observed for Logistic
Regression, indicating that simpler models benefit more from receiving explicit family-related representations than tree-based models.

#### Findings

- FamilySize and IsAlone do not consistently outperform the original SibSp and Parch features when used as replacements.

- Keeping both the engineered and original family features generally performs better than replacing the originals.

- Logistic Regression consistently benefits from engineered family features.

- Most tree-based models gain only modest improvements, suggesting they already extract much of the relationship between SibSp and Parch directly.

- FamilySize and IsAlone appear to provide complementary rather than redundant information.

#### Current recommendation

- Logistic Regression
    - SibSp + Parch + FamilySize + IsAlone.
    - The engineered features consistently improve performance, and the combination performs better than replacing the original variables.

- Decision Tree
    - SibSp + Parch + FamilySize + IsAlone may be worthwhile.
    - The gains are modest but consistently positive.

- Random Forest
    - Prefer the combined representation.

- Extra Trees
    - The combined representation provides small improvements but is not essential.

- XGBoost
    - Small improvement from the combined representation.
    - Either approach is acceptable.

- KNN
    - Prefer the original SibSp and Parch features.

- SVC
    - Prefer the original SibSp and Parch features.

---

### Cabin

#### Hypothesis

Cabin contains a very large proportion of missing values, while most other features are relatively complete. This suggests that the missingness may not be entirely random, but instead related to some characteristic of the passengers or the data collection process.

One possibility is that passengers whose cabin is unknown are systematically different from those whose cabin is known. For example, because around 70% of passengers without cabin information did not survive, the absence of cabin information may itself contain predictive information.

This hypothesis is tested through the Has_Cabin feature.

Cabin's values always starts with a letter, which likely points to a location within the ship, like its deck. Perhaps knowing the passenges' cabin position on the ship makes it easier to predict their survival.

This hypothesis is tested through the Deck feature.

#### Experiments performed:

#### fe02__has_cabin

Experiment meant to tests if the missingness of the cabins is a signal in itself.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.005
    - Secondary gains:
      - test_f1_mean: 0.01
  - extra_trees: test_accuracy_mean: 0.003

##### Conclusion

Negligible changes on its own, likely on the level of noise. Surprisingly, LogReg had a small improvement. Going to explore the impact of Deck and Deck + has_cabin to find out whether they complement each other or if Deck's enough on its own.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe02__has_cabin | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.723 |                0.01  |
| baseline__raw     | fe02__has_cabin | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.743 |                0.001 |
| baseline__raw     | fe02__has_cabin | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.756 |               -0.004 |
| baseline__raw     | fe02__has_cabin | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe02__has_cabin | random_forest |                          0.822 |                        0.818 |                     -0.004 |                    0.744 |                  0.739 |               -0.005 |
| baseline__raw     | fe02__has_cabin | extra_trees   |                          0.804 |                        0.807 |                      0.003 |                    0.721 |                  0.729 |                0.008 |
| baseline__raw     | fe02__has_cabin | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.752 |               -0.006 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe02__has_cabin |                    -0.000714286 |                         -0.004 |                          0.005 |               0.000428571 |                   -0.006 |                     0.01 |

</details>

#### fe03__deck

Testing the impact of the feature deck in the models. Expecting a higher impact than has_cabin, but not by a large margen, given that only 23% of the decks are know.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.005
  - knn: test_accuracy_mean: 0.009
    - Secondary gains:
      - test_f1_mean: 0.01

##### Conclusion

Cabin-derived features showed only minor impact. Interestingly, Logistic Regression gained exactly the same accuracy improvement (+0.005) as with Has_Cabin. At this point it is unclear whether this reflects a genuine pattern or simply coincidence, so later experiments will help determine which explanation is more likely.

However, approximately 77% of cabin values are missing, leaving usable cabin information for only about 23% of passengers. This severely limits the feature's potential contribution. The weak results may therefore reflect limited coverage rather than lack of predictive signal. Cabin-based features remain an interesting indicator, but their usefulness is constrained by the large amount of missing data.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe03__deck      | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.722 |                0.009 |
| baseline__raw     | fe03__deck      | knn           |                          0.809 |                        0.818 |                      0.009 |                    0.742 |                  0.752 |                0.01  |
| baseline__raw     | fe03__deck      | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.757 |               -0.003 |
| baseline__raw     | fe03__deck      | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe03__deck      | random_forest |                          0.822 |                        0.813 |                     -0.009 |                    0.744 |                  0.734 |               -0.01  |
| baseline__raw     | fe03__deck      | extra_trees   |                          0.804 |                        0.799 |                     -0.005 |                    0.721 |                  0.719 |               -0.002 |
| baseline__raw     | fe03__deck      | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.751 |               -0.007 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe03__deck      |                     -0.00128571 |                         -0.009 |                          0.009 |              -0.000571429 |                    -0.01 |                     0.01 |

</details>

#### fe04__cabin_features

Testing the impact of using both has_cabin and deck together, to see if this union generates better information or if they are redundant.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.005
    - Secondary gains:
      - test_f1_mean: 0.011
  - knn: test_accuracy_mean: 0.008
    - Secondary gains:
      - test_f1_mean: 0.01
  - xgb: test_accuracy_mean: 0.006

##### Conclusion

Results were surprising. Meanwhile the mean delta is essentially 0, this combination had greater influence on the models than deck or has_cabin. This suggests that the two features capture different aspects of the underlying information rather than simply encoding the same signal. 

Random forest was the model that suffered the most from it, One possible explanation is that Random Forest already extracts most of the available information from the existing variables, making the additional Cabin-derived features partially redundant.

Logreg continued to gain exactly 0.005 on all three attempts, meaning that its likely getting the same information from all three approaches, initially telling me that just using one of them would do. But Fe04 also increased its f1 by 0.011, meaning its actually generalizing better when using both features together

Knn gains were about the same as using Deck features, accuracy's slightly lower (-0.001), but the difference's small enough to be confused with noise or coincidence.

Xgb benefited from explicitly separating cabin presence (Has_Cabin) from cabin location (Deck), rather than having to infer both from a single feature. Enough to raise it to the top of the current leaderboard, with a mean accuracy of 0.832 (+0.006 compared to raw), ahead of Raw SVC by 0.005.

These results strengthen the hypothesis that Cabin contains useful information, although its practical value is heavily limited by the large amount of missing data.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group        | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe04__cabin_features | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.724 |                0.011 |
| baseline__raw     | fe04__cabin_features | knn           |                          0.809 |                        0.817 |                      0.008 |                    0.742 |                  0.752 |                0.01  |
| baseline__raw     | fe04__cabin_features | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.757 |               -0.003 |
| baseline__raw     | fe04__cabin_features | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe04__cabin_features | random_forest |                          0.822 |                        0.808 |                     -0.014 |                    0.744 |                  0.726 |               -0.018 |
| baseline__raw     | fe04__cabin_features | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.728 |                0.007 |
| baseline__raw     | fe04__cabin_features | xgb           |                          0.826 |                        0.832 |                      0.006 |                    0.758 |                  0.767 |                0.009 |

##### Summary

| compare_group        |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe04__cabin_features |                     0.000142857 |                         -0.014 |                          0.008 |                0.00214286 |                   -0.018 |                    0.011 |

</details>

#### Overall conclusion

Although none of the Cabin-derived features produced large improvements in isolation, combining Has_Cabin and Deck consistently outperformed either feature individually for several models.

This suggests that Cabin contains meaningful information, but its predictive power is fundamentally limited by the fact that approximately 77% of cabin values are missing.

#### Findings

- Cabin contains useful predictive information despite its extremely high missing rate.

- Separating Cabin into Has_Cabin and Deck preserves more information than using either feature alone.

- The usefulness of Cabin-derived features appears to be model dependent.


#### Working hypotheses

- Logistic Regression appears to benefit from explicit feature refinement. If this pattern continues across unrelated feature engineering experiments, it may indicate that the model performs better when informative relationships are made explicit rather than left for the model to infer.

- KNN may benefit from explicit one-hot representations of categorical information. Future experiments using additional categorical features will help determine whether this is a consistent pattern or simply an artifact of the Cabin experiments.

- XGBoost showed its largest improvement when Has_Cabin and Deck were used together rather than separately. This may indicate that the model benefits from having multiple related features available instead of relying on a single representation. Future experiments combining and fusing features will help determine whether this is a broader characteristic of the model.

#### Current recommendation

- Logistic Regression
    - Cabin Features

- KNN
    - Deck
    - Cabin Features

- XGBoost
    - Cabin Features

---

### Name

#### Hypothesis

Every passenger's name contains a title (e.g., Mr., Mrs., Miss., Master). These titles may encode information about the passenger's gender, approximate age, and social status.

Since Sex, Pclass and Fare already capture part of this information, I expect Title to provide only a modest improvement. Nevertheless, because it combines multiple characteristics into a single feature, it was worth investigating. Additionally, Title may prove useful for imputing missing Age values.

#### Experiments performed:


#### fe05__title

Title is expected to encode information related to the passenger's gender, age and social status. This experiment evaluates how much additional predictive power this feature provides beyond the variables already present in the baseline model.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: strong_general
- Recommended for all models
- Mean delta: 0.016

##### Conclusion

Unlike the previous feature engineering experiments, Title produced consistent improvements across every tested model. The feature appears to summarize multiple passenger characteristics—including age, gender and social status—into a single, highly informative variable. Given the magnitude and consistency of the improvements, Title is a strong candidate for inclusion in future feature sets.

The addition of Title also reshaped the leaderboard. Title + XGBoost and Title + SVC moved into first and second place respectively, while Title + Random Forest climbed close behind Cabin Features + XGBoost.

One particularly interesting observation is that Logistic Regression improved by 0.039 accuracy, the largest gain obtained by any model from a single feature engineering experiment so far. Combined with its positive response to previous engineered features, this suggests that Logistic Regression benefits substantially from features that make informative relationships explicit, rather than requiring the model to infer them from the original variables.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe05__title     | logreg        |                          0.786 |                        0.825 |                      0.039 |                    0.713 |                  0.765 |                0.052 |
| baseline__raw     | fe05__title     | knn           |                          0.809 |                        0.822 |                      0.013 |                    0.742 |                  0.76  |                0.018 |
| baseline__raw     | fe05__title     | svc           |                          0.827 |                        0.834 |                      0.007 |                    0.76  |                  0.771 |                0.011 |
| baseline__raw     | fe05__title     | decision_tree |                          0.803 |                        0.823 |                      0.02  |                    0.702 |                  0.756 |                0.054 |
| baseline__raw     | fe05__title     | random_forest |                          0.822 |                        0.832 |                      0.01  |                    0.744 |                  0.768 |                0.024 |
| baseline__raw     | fe05__title     | extra_trees   |                          0.804 |                        0.82  |                      0.016 |                    0.721 |                  0.754 |                0.033 |
| baseline__raw     | fe05__title     | xgb           |                          0.826 |                        0.836 |                      0.01  |                    0.758 |                  0.772 |                0.014 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe05__title     |                       0.0164286 |                          0.007 |                          0.039 |                 0.0294286 |                    0.011 |                    0.054 |

</details>

#### Overall conclusion

Title proved to be the most informative engineered feature explored so far, improving every tested model. Unlike previous feature engineering attempts, its benefits were both consistent and substantial.

One possible explanation is that Title summarizes several relevant passenger characteristics—such as gender, approximate age and social status—into a single feature. Rather than introducing entirely new information, it appears to organize information already present in the dataset into a representation that models can exploit more effectively.

#### Findings

- Title proved considerably more predictive than initially expected.

- Title is the first engineered feature to consistently improve every tested model.

- Logistic Regression obtained the largest improvement observed so far (+0.039 accuracy), reinforcing the hypothesis that explicit feature engineering benefits linear models.

#### Working hypotheses

- Title may be effective because it combines multiple weak signals (age, gender and social status) into a single informative feature.

- Logistic Regression appears to benefit substantially from explicit feature refinement. Future feature engineering experiments will help determine whether this remains a consistent pattern.

#### Current recommendation

- All models
    - Title feature

---

### Age

#### Hypothesis

Age is expected to influence survival because it affects both physical capability and independence during an emergency. Children may depend on accompanying adults, while elderly passengers may have reduced mobility. Adults occupy an intermediate range where physical capability and experience may both contribute to survival.

This investigation explores three independent questions:

- Can missing Age values be estimated more accurately than simple median imputation?
- Is Age better represented as a continuous or ordinal variable?
- Does combining both representations provide complementary information?

#### Experiments performed:

#### Age Imputation:

#### fe06__age_imputation_title

Age contains a substantial proportion of missing values. While median imputation provides a simple baseline, passenger title may provide additional information about likely age. This experiment tests median Age imputation grouped by Title. Group statistics are learned independently within each cross-validation training fold and applied to its validation fold, ensuring that validation observations do not influence the imputation statistics.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.003
  - xgb: test_accuracy_mean: 0.003


##### Conclusion

Title-based Age imputation is a viable alternative to global median imputation, but the CV-safe results do not show a strong general advantage. Its effects are model-dependent and generally small, with KNN showing a meaningful deterioration.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group              | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe06__age_imputation_title | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.714 |                0.001 |
| baseline__raw     | fe06__age_imputation_title | knn           |                          0.809 |                        0.804 |                     -0.005 |                    0.742 |                  0.73  |               -0.012 |
| baseline__raw     | fe06__age_imputation_title | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe06__age_imputation_title | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.703 |                0.001 |
| baseline__raw     | fe06__age_imputation_title | random_forest |                          0.822 |                        0.823 |                      0.001 |                    0.744 |                  0.747 |                0.003 |
| baseline__raw     | fe06__age_imputation_title | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | fe06__age_imputation_title | xgb           |                          0.826 |                        0.829 |                      0.003 |                    0.758 |                  0.761 |                0.003 |

##### Summary

| compare_group              |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe06__age_imputation_title |                     0.000857143 |                         -0.005 |                          0.003 |                         0 |                   -0.012 |                    0.004 |

</details>

#### fe07__age_imputation_title_pclass

Age contains a substantial proportion of missing values. The baseline uses median imputation, while FE06 investigated estimating missing Age values from passenger Title.

This experiment tests whether adding Pclass as additional context improves the usefulness of the imputation. Missing Age values are therefore imputed using the median Age of passengers sharing both Title and Pclass.

The grouped statistics are learned independently within each cross-validation training fold and applied to the corresponding validation fold.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.012
    - Secondary gains:
      - test_f1_mean: 0.011


##### Conclusion

Title-and-Pclass-based Age imputation is not a generally superior replacement for median imputation, but it is a promising model-specific choice for Logistic Regression.

The experiment also demonstrates that increasing the contextual specificity of an imputation strategy can affect models very differently. More conditional imputation does not automatically produce a better representation: its value ultimately depends on how the resulting feature interacts with the model.

Logistic Regression should retain this configuration as a candidate for later model-specific feature selection.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                     | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe07__age_imputation_title_pclass | logreg        |                          0.786 |                        0.798 |                      0.012 |                    0.713 |                  0.724 |                0.011 |
| baseline__raw     | fe07__age_imputation_title_pclass | knn           |                          0.809 |                        0.799 |                     -0.01  |                    0.742 |                  0.733 |               -0.009 |
| baseline__raw     | fe07__age_imputation_title_pclass | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe07__age_imputation_title_pclass | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | fe07__age_imputation_title_pclass | random_forest |                          0.822 |                        0.824 |                      0.002 |                    0.744 |                  0.746 |                0.002 |
| baseline__raw     | fe07__age_imputation_title_pclass | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | fe07__age_imputation_title_pclass | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.757 |               -0.001 |

##### Summary

| compare_group                     |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe07__age_imputation_title_pclass |                     0.000857143 |                          -0.01 |                          0.012 |               0.000714286 |                   -0.009 |                    0.011 |

</details>

#### Age representation:

#### fe11__age_bin

This experiment tests whether replacing continuous Age with a discretized representation changes what the models learn.

Age is divided into bins derived from previous exploratory analysis. The bins are ordinally encoded according to their observed relationship with survival rather than chronological age, allowing the representation to emphasize survival-relevant age groups.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.01
    - Secondary gains:
      - test_f1_mean: 0.02
  - random_forest: test_accuracy_mean: 0.003


##### Conclusion

Age binning is strongly model-specific rather than generally beneficial.

The representation should be retained as a candidate for Decision Tree, where it produced meaningful improvements in both accuracy and F1. Random Forest showed a smaller benefit worth retaining for comparison, while there is little evidence to prefer the binned representation for the remaining models.

The results reinforce that feature engineering can be valuable when its representation aligns with a model's inductive structure, even when it does not introduce new information.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe11__age_bin   | logreg        |                          0.786 |                        0.785 |                     -0.001 |                    0.713 |                  0.713 |                0     |
| baseline__raw     | fe11__age_bin   | knn           |                          0.809 |                        0.801 |                     -0.008 |                    0.742 |                  0.74  |               -0.002 |
| baseline__raw     | fe11__age_bin   | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.759 |               -0.001 |
| baseline__raw     | fe11__age_bin   | decision_tree |                          0.803 |                        0.813 |                      0.01  |                    0.702 |                  0.722 |                0.02  |
| baseline__raw     | fe11__age_bin   | random_forest |                          0.822 |                        0.825 |                      0.003 |                    0.744 |                  0.748 |                0.004 |
| baseline__raw     | fe11__age_bin   | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.724 |                0.003 |
| baseline__raw     | fe11__age_bin   | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.756 |               -0.002 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe11__age_bin   |                     0.000428571 |                         -0.008 |                           0.01 |                0.00314286 |                   -0.002 |                     0.02 |

</details>

#### Age representation combination

#### cb01__age_and_bins

This experiment combines continuous Age with the ordinal Age_bin representation tested in FE11.

Rather than asking whether Age bins provide a better replacement for continuous Age, this experiment tests whether the two representations provide complementary information when available simultaneously.

It also investigates whether the effect of a feature representation can be inferred from its performance in isolation, or whether interactions between alternative representations need to be evaluated directly.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.008
  - random_forest: test_accuracy_mean: 0.003
  - extra_trees: test_accuracy_mean: 0.014
    - Secondary gains:
      - test_f1_mean: 0.022


##### Conclusion

Continuous Age and Age_bin should not be treated as interchangeable representations. Their usefulness depends both on the model and on whether they are used independently or together.

Extra Trees is the strongest example of complementarity: Age_bin alone provides little benefit, while combining it with continuous Age produces substantial improvements in both accuracy and F1. Logistic Regression shows a similar, though smaller, pattern.

Decision Tree shows the opposite behavior, benefiting from Age_bin as a replacement for continuous Age but not from their combination. This supports retaining Age_bin alone as the preferred Age representation for Decision Tree.

The experiment demonstrates that the value of engineered representations cannot always be predicted from their isolated performance. Alternative representations that appear redundant independently may become useful when combined, while representations that perform well alone may lose their advantage when the original feature is restored.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group      | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb01__age_and_bins | logreg        |                          0.786 |                        0.794 |                      0.008 |                    0.713 |                  0.722 |                0.009 |
| baseline__raw     | cb01__age_and_bins | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | cb01__age_and_bins | svc           |                          0.827 |                        0.828 |                      0.001 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | cb01__age_and_bins | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.702 |                0     |
| baseline__raw     | cb01__age_and_bins | random_forest |                          0.822 |                        0.825 |                      0.003 |                    0.744 |                  0.748 |                0.004 |
| baseline__raw     | cb01__age_and_bins | extra_trees   |                          0.804 |                        0.818 |                      0.014 |                    0.721 |                  0.743 |                0.022 |
| baseline__raw     | cb01__age_and_bins | xgb           |                          0.826 |                        0.827 |                      0.001 |                    0.758 |                  0.761 |                0.003 |

##### Summary

| compare_group      |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb01__age_and_bins |                      0.00285714 |                         -0.007 |                          0.014 |                0.00485714 |                   -0.008 |                    0.022 |

</details>


#### cb02__age_imputed_title_and_bins

This experiment combines title-based Age imputation from FE06 with the continuous-and-binned Age representation explored in CB01.

Missing Age values are first imputed using the median Age of passengers sharing the same Title. The resulting Age is then retained as a continuous feature while also being transformed into the ordinal Age_bin representation.

This tests whether a more contextual Age estimate changes the usefulness of combining continuous and discretized representations.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.005
  - svc: test_accuracy_mean: 0.004
  - decision_tree: test_accuracy_mean: 0.003
  - random_forest: test_accuracy_mean: 0.005
  - extra_trees: test_accuracy_mean: 0.018
    - Secondary gains:
      - test_f1_mean: 0.027


#### Conclusion

Title-imputed continuous Age combined with Age_bin is a strong model-specific representation, particularly for Extra Trees, where it produces the best Age-related improvement observed so far.

Random Forest and SVC also benefit, though more modestly. For KNN, the combination remains detrimental and should not be retained.

The experiment further demonstrates that feature transformations should not be evaluated only in isolation. Neither Age binning nor title-based imputation independently predicts the substantial Extra Trees improvement produced when the representations are combined.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                    | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb02__age_imputed_title_and_bins | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | cb02__age_imputed_title_and_bins | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.729 |               -0.013 |
| baseline__raw     | cb02__age_imputed_title_and_bins | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | cb02__age_imputed_title_and_bins | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.703 |                0.001 |
| baseline__raw     | cb02__age_imputed_title_and_bins | random_forest |                          0.822 |                        0.827 |                      0.005 |                    0.744 |                  0.753 |                0.009 |
| baseline__raw     | cb02__age_imputed_title_and_bins | extra_trees   |                          0.804 |                        0.822 |                      0.018 |                    0.721 |                  0.748 |                0.027 |
| baseline__raw     | cb02__age_imputed_title_and_bins | xgb           |                          0.826 |                        0.827 |                      0.001 |                    0.758 |                  0.759 |                0.001 |

##### Summary

| compare_group                    |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb02__age_imputed_title_and_bins |                      0.00414286 |                         -0.007 |                          0.018 |                     0.005 |                   -0.013 |                    0.027 |

</details>

### cb03__age_imputed_title_pclass_and_bins

This experiment combines Title-and-Pclass-based Age imputation from FE07 with the continuous-and-binned Age representation explored in the previous combination experiments.

Missing Age values are estimated from passengers sharing both Title and Pclass. The resulting Age is retained as a continuous feature while also being transformed into the ordinal Age_bin representation.

This tests whether combining a more context-specific Age estimate with multiple representations provides additional predictive value.

<details>
<summary>Conclusion</summary>


#### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.02
    - Secondary gains:
      - test_f1_mean: 0.022
  - svc: test_accuracy_mean: 0.004
  - random_forest: test_accuracy_mean: 0.006
    - Secondary gains:
      - test_f1_mean: 0.01
  - extra_trees: test_accuracy_mean: 0.016
    - Secondary gains:
      - test_f1_mean: 0.026


#### Conclusion

Title-and-Pclass-imputed Age combined with Age_bin is a strong model-specific Age representation, particularly for Logistic Regression, where it produces the best Age-related result observed so far.

Extra Trees also benefits substantially from the combined representation, reinforcing the evidence that continuous and ordinal Age provide complementary predictive structure for this model. Random Forest and SVC show smaller positive effects.

The experiment further demonstrates that increasing preprocessing complexity is not universally beneficial: the same representation that substantially helps Logistic Regression and Extra Trees is detrimental to KNN and provides no advantage to Decision Tree or XGBoost.

</details>

<details>
<summary>Experiment details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group                           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | logreg        |                          0.786 |                        0.806 |                      0.02  |                    0.713 |                  0.735 |                0.022 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | knn           |                          0.809 |                        0.798 |                     -0.011 |                    0.742 |                  0.73  |               -0.012 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | random_forest |                          0.822 |                        0.828 |                      0.006 |                    0.744 |                  0.754 |                0.01  |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | extra_trees   |                          0.804 |                        0.82  |                      0.016 |                    0.721 |                  0.747 |                0.026 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | xgb           |                          0.826 |                        0.824 |                     -0.002 |                    0.758 |                  0.756 |               -0.002 |

#### Summary

| compare_group                           |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb03__age_imputed_title_pclass_and_bins |                      0.00471429 |                         -0.011 |                           0.02 |                0.00685714 |                   -0.012 |                    0.026 |

</details>

#### Accidental Fare ablation review

<details>
<summary>Ablation details</summary>

A configuration review found that Fare had been unintentionally omitted from several historical Age experiments. The affected experiments were corrected, and the corrected configurations are used as the canonical Age experiments.

Because the original configurations differ from their corrected counterparts specifically through the absence of Fare, they were retained as explicit ablation experiments. This converts the accidental runs into controlled comparisons that can be used to examine Fare's marginal contribution under different Age representations.

Four Age configurations are represented:

- continuous Age combined with Age_bin;
- Title-imputed Age combined with Age_bin;
- Title-and-Pclass-imputed Age combined with Age_bin;
- Age_bin replacing continuous Age.

Together, these experiments test whether Fare's usefulness remains stable as the representation of Age changes.

**Accuracy effect of removing Fare**

| Model               |      AB01 |      AB02 |      AB03 |      AB04 | Pattern                                               |
| ------------------- | --------: | --------: | --------: | --------: | ----------------------------------------------------- |
| Logistic Regression |     +.001 |     +.001 |     -.004 |     -.002 | Small, context-dependent                              |
| KNN                 |     -.007 |     +.004 | **+.015** |     +.009 | Highly context-dependent; removal increasingly useful |
| SVC                 |      .000 |     +.001 |      .000 |     +.004 | Mostly insensitive                                    |
| Decision Tree       |     +.004 |      .000 |     +.002 |     -.004 | Small accuracy effects                                |
| Random Forest       | **+.007** |     +.002 |     +.006 | **+.008** | **Consistently benefits from removal**                |
| Extra Trees         |     -.001 | **-.005** |     -.001 | **+.014** | Strong representation interaction                     |
| XGBoost             | **-.007** | **+.008** |      .000 |     +.001 | Highly context-dependent                              |

**F1 effect of removing Fare**

| Model               |      AB01 |      AB02 |      AB03 |      AB04 | Pattern                                |
| ------------------- | --------: | --------: | --------: | --------: | -------------------------------------- |
| Logistic Regression |      .000 |     +.002 |     -.005 |     -.003 | Small/context-dependent                |
| KNN                 | **-.016** |     -.002 |     +.007 |     -.003 | Context-dependent                      |
| SVC                 |     -.002 |      .000 |     -.001 |     +.005 | Mostly insensitive                     |
| Decision Tree       | **+.017** | **+.011** | **+.015** |      .000 | Removal helps combinations             |
| Random Forest       | **+.012** |     +.001 | **+.008** | **+.011** | **Consistently benefits from removal** |
| Extra Trees         |      .000 |     -.006 |     -.001 | **+.013** | Strong representation interaction      |
| XGBoost             | **-.017** |     +.007 |     -.008 |     -.004 | Highly context-dependent               |


#### Fare ablation conclusion

Fare's contribution is strongly dependent on both the model and the surrounding Age representation. Removing Fare does not produce a universal improvement or deterioration; instead, several models show distinct and repeatable interactions.

Random Forest shows the clearest consistent pattern. Removing Fare improves accuracy across all four Age configurations and either improves or preserves F1, with several configurations producing meaningful gains in both metrics. This provides sufficient evidence to justify explicitly testing Fare removal from the eventual Random Forest final candidate.

Extra Trees exhibits a strong representation interaction. Fare is neutral or beneficial when continuous Age and Age_bin are available together, but becomes substantially detrimental when Age_bin replaces continuous Age. This suggests that Fare's marginal contribution depends on how other continuous and ordinal information is represented rather than being inherently useful or harmful to the model.

Decision Tree shows a different interaction. Removing Fare consistently improves F1 when continuous and binned Age are combined, but provides no benefit when Age_bin replaces continuous Age. Since the latter representation already performs substantially better for Decision Tree, Fare removal is not currently indicated for its preferred Age configuration.

Logistic Regression and SVC are comparatively insensitive to Fare removal across these experiments, while KNN and XGBoost show context-dependent changes that do not support a general inclusion or exclusion rule.

Overall, these ablations reinforce that the marginal value of a feature cannot be evaluated independently of the surrounding representation or model. Fare contains predictive information, but whether that information improves generalization depends on what alternative and overlapping structure is already available to the estimator.

</details>

#### Overall conclusion

Age proved to be one of the richest features investigated, involving two distinct feature-engineering questions: how missing Age values should be estimated and how Age should be represented to each model.

More contextual imputation did not provide a universal improvement. Title-based imputation was mostly neutral, while adding Pclass produced a substantial improvement for Logistic Regression but little or no additional benefit for several other models. This indicates that a more specific imputation strategy should not automatically be considered a better representation simply because it uses additional information.

The representation experiments revealed stronger model-specific differences. Replacing continuous Age with the ordinal Age_bin representation produced a substantial improvement for Decision Tree, but little benefit for most other models. Conversely, retaining continuous Age alongside Age_bin produced substantial gains for Extra Trees and Logistic Regression under suitable imputation strategies. Extra Trees was particularly notable: neither Age binning nor contextual imputation was especially useful independently, yet their combination with continuous Age produced some of the strongest Age-related improvements observed.

KNN showed the opposite pattern. Every investigated Age transformation reduced its performance relative to the baseline, suggesting that the engineered representations tested here are poorly suited to its distance-based decision process.

The Fare ablations further demonstrated that the usefulness of an Age representation cannot always be separated from the surrounding feature set. Random Forest consistently improved when Fare was removed from the tested Age configurations, while Extra Trees and Decision Tree showed interactions dependent on whether continuous and binned Age were provided together.

Overall, there is no universally optimal treatment of Age. Its usefulness depends not only on the information contained in the transformation, but on how that representation interacts with the inductive behavior of the model and with other available features.

#### Findings

- More contextual Age imputation is not universally better. Title-based imputation produced mostly small effects, while Title + Pclass imputation was particularly beneficial to Logistic Regression but neutral or detrimental to some other models.

- Age binning is strongly model-dependent rather than generally beneficial to tree models. Decision Tree clearly benefited from replacing continuous Age with Age_bin, while Random Forest and Extra Trees showed only small improvements from binning alone.

- Multiple representations can be complementary even when neither transformation is particularly useful independently. This was clearest for Extra Trees, where continuous Age combined with Age_bin produced substantial gains despite little benefit from Age binning or contextual imputation alone.

- Logistic Regression benefited from explicit Age structure. Its strongest Age result came from combining Title + Pclass imputation with continuous and ordinal Age representations, suggesting that explicitly engineered structure can expose relationships that the linear model cannot represent as easily from raw Age alone.

- Decision Tree favored simplification rather than representation multiplicity. Age_bin alone produced its strongest Age result, while restoring continuous Age alongside it removed that advantage.

- KNN consistently responded negatively to the investigated Age engineering. All reviewed Age transformations reduced performance, making raw Age the strongest current choice for this model.

- Feature effects depend on surrounding features as well as the model. Fare ablation showed repeatable model-specific interactions, particularly for Random Forest, Decision Tree, and Extra Trees.

#### Current recommendation

- Logistic Regression
  - Use Title + Pclass-imputed Age together with Age_bin (CB03).
  - This produced the strongest Age-related result for Logistic Regression: +0.020 accuracy and +0.022 F1.
  - Both the contextual imputation and alternative Age representation appear useful in combination.

- SVC
  - Use Title-imputed Age together with Age_bin (CB02), although the benefit is modest.
  - CB02 and CB03 both reached +0.004 accuracy and +0.007 F1, so the simpler Title-based variant is preferable unless later feature combinations change the result.
  - Age engineering appears useful but not particularly important to SVC.

- Decision Tree
  - Replace continuous Age with Age_bin (FE11).
  - This produced +0.010 accuracy and +0.020 F1, clearly outperforming the combined Age representations.
  - Decision Tree appears to benefit from the simplified ordinal representation rather than having continuous and binned Age simultaneously.

- Random Forest
  - Current Age candidate: Title + Pclass-imputed Age together with Age_bin (CB03).
  - CB03 produced +0.006 accuracy and +0.010 F1, stronger than corrected FE11 and the other canonical Age experiments.
  - However, Fare ablations consistently improved Random Forest, so the preferred Age representation should be reevaluated with and without Fare during final-model construction.

- Extra Trees
  - Use Title-imputed Age together with Age_bin (CB02).
  - CB02 produced the strongest canonical Age result at +0.018 accuracy and +0.027 F1.
  - CB03 was very close (+0.016/+0.026), but adding Pclass to the imputation provides no advantage, making CB02 the simpler choice.
  - The evidence strongly favors retaining both continuous and ordinal Age representations rather than either alone.

- KNN
  - Retain baseline Age processing.
  - Every investigated Age transformation reduced performance relative to baseline.
  - There is currently no evidence that additional Age engineering benefits KNN.

- XGBoost
  - Retain baseline Age processing for now.
  - None of the Age transformations produced a convincing improvement, and several were slightly detrimental.
  - Age engineering therefore has low priority for XGBoost unless interactions with later final-model features justify reconsideration.

#### Open questions

- Why does Logistic Regression benefit so strongly from Title + Pclass Age imputation when most other models do not?

  - Is the imputation exposing a relationship that is easier for the linear decision boundary to exploit?

- Why does Decision Tree strongly prefer Age_bin alone, while Extra Trees benefits from continuous and binned Age together?

  - Is the manually engineered threshold structure conserving the constrained Decision Tree's splitting capacity, while Extra Trees benefits from having multiple candidate representations available?

- Why does KNN consistently deteriorate under Age engineering?

  - Does adding correlated or discretized Age representations distort the distance space or effectively overweight Age-related information?

- Would learned Age bins outperform the manually designed survival-informed bins?

- Would other nonlinear representations—such as splines, quantile bins, or other continuous transformations—provide the benefits of Age binning without discarding within-bin information?

- Imputation-quality experiment: mask a subset of known Age values and directly compare median, Title, and Title + Pclass imputation error.

  - This would separate better Age estimation from better downstream survival prediction.

- Final-model Fare ablation for Random Forest: compare the final RF candidate with and without Fare. Removing Fare improved RF accuracy across all four Age ablations and frequently improved F1 as well.

- Conditional Fare ablation for Extra Trees: if the final Extra Trees model retains the continuous + binned Age representation, test Fare removal again rather than extrapolating from Age_bin alone. The current ablations show that Fare's effect reverses depending on the Age representation.

#### Practical takeaway

Age demonstrated that feature engineering is not simply a process of creating increasingly informative versions of a variable. The same underlying information can become more or less useful depending on how it is represented, which model receives it, and which other features are available.

More contextual imputation helped some models but not others; discretization strongly benefited Decision Tree but was largely ineffective by itself for Extra Trees; and combining continuous and ordinal representations produced substantial gains for Logistic Regression and Extra Trees while consistently hurting KNN.

Therefore, feature transformations should be treated as model-dependent representations rather than universal improvements to the data. Evaluate transformations individually, test potentially complementary representations together, and verify important interactions with the surrounding feature set before selecting the final configuration.

---

### Ticket

#### Hypothesis

Multiple passengers share the same Ticket identifier, suggesting that tickets may represent travel groups rather than individuals. If passengers sharing a ticket remained together during boarding or evacuation, the number of passengers associated with a ticket may contain information beyond the family relationships captured by SibSp and Parch.

Unlike FamilySize, TicketGroupSize depends entirely on the passengers present in the current dataset. If members of the same ticket group are absent from the dataset, the feature underestimates the true group size. This makes it inherently dataset-dependent, and its effect on unseen data is uncertain.

#### Experiments performed:

#### fe09__ticket_group_size

TicketGroupSize counts the number of passengers sharing the same ticket. The objective is to determine whether actual travel groups contain more predictive information than family relationships.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - svc: test_accuracy_mean: 0.005
    - Secondary gains:
      - test_f1_mean: 0.01


##### Conclusion

Ambiguous results, it helped some models, but hurt others. Considering how low its effects are, the feature either introduces noise or captures information that is already available through existing features, such as Parch and SibSp or Fare. Only Svc seemed to have gained something meaningful from it.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe09__ticket_group_size | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | fe09__ticket_group_size | knn           |                          0.809 |                        0.804 |                     -0.005 |                    0.742 |                  0.733 |               -0.009 |
| baseline__raw     | fe09__ticket_group_size | svc           |                          0.827 |                        0.832 |                      0.005 |                    0.76  |                  0.77  |                0.01  |
| baseline__raw     | fe09__ticket_group_size | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.702 |                0     |
| baseline__raw     | fe09__ticket_group_size | random_forest |                          0.822 |                        0.82  |                     -0.002 |                    0.744 |                  0.747 |                0.003 |
| baseline__raw     | fe09__ticket_group_size | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.724 |                0.003 |
| baseline__raw     | fe09__ticket_group_size | xgb           |                          0.826 |                        0.818 |                     -0.008 |                    0.758 |                  0.749 |               -0.009 |

##### Summary

| compare_group           |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe09__ticket_group_size |                     -0.00128571 |                         -0.008 |                          0.005 |                         0 |                   -0.009 |                     0.01 |

</details>

#### Overall conclusion

TicketGroupSize alone appears to provide only limited additional information. The modest improvements suggest that passengers traveling together do share some survival characteristics, but this information is either weak or already partially represented by existing features.

The remaining question is whether TicketGroupSize becomes more informative when combined with Ticket-derived features such as Fare per Ticket Member.

#### Findings

- TicketGroupSize contains some predictive information, but considerably less than initially expected.
- SVC was the only model to benefit consistently from the feature.
- The overlap between TicketGroupSize and existing family-related features appears larger than originally hypothesized.

#### hypotheses

- TicketGroupSize may become more useful when combined with other Ticket-derived features.
- The partial overlap between TicketGroupSize and FamilySize may explain the limited improvements observed.
- Dataset-specific ticket groups may limit generalization to unseen passengers.

#### Current recommendation

- SVC
  - Ticket_group_size

---

### Fare

#### Hypothesis

Fare does not map cleanly to Pclass, and multiple passengers sometimes share the same Ticket identifier. This suggests that Fare may not always represent the amount paid by one individual passenger. In some cases, it may instead represent the cost associated with a family or ticket group.

This investigation considers two possible ways to estimate the number of passengers covered by a fare:

- **FamilySize**, which uses SibSp and Parch to approximate the passenger's family group.
- **TicketGroupSize**, which counts the passengers sharing the same Ticket identifier within the available dataframe.

Neither approach is guaranteed to represent the true paying group. Family members may have travelled under different tickets, while passengers sharing a ticket may not all appear in the same dataframe. The objective is therefore not to recover an exact individual fare, but to test whether either approximation produces a more useful representation than raw Fare.

Another approach worth of investigation is whether using both raw fate and the engineered one could lead to better predictions, or if it's just redundancy.

#### Experiments performed:

#### Alternative Fare representations

#### fe08__fare_per_family_member

I expected that fare is not the amount paid by one passenger, but rather the whole family. By dividing fare by the family member, I expect to give the model a more precise feature to work with. FamilySize provides a dataset-independent approximation of the passenger’s immediate family group because it is calculated from SibSp and Parch rather than by counting matching rows.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003
  - decision_tree: test_accuracy_mean: 0.004

##### Conclusion

Fare_per_family appears to contain some useful information, but FamilySize is likely an imperfect approximation of the number of passengers covered by a fare. Because family size does not always correspond to the number of passengers sharing a fare, the feature introduces a considerable amount of noise. The experiments show inconsistent behavior across models: some (especially Decision Tree and, to a lesser extent, Logistic Regression) benefit slightly, while others lose performance. Overall, the feature does not consistently outperform the original Fare feature and is therefore not recommended as a general replacement.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-----------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe08__fare_per_family_member | logreg        |                          0.786 |                        0.789 |                      0.003 |                    0.713 |                  0.717 |                0.004 |
| baseline__raw     | fe08__fare_per_family_member | knn           |                          0.809 |                        0.804 |                     -0.005 |                    0.742 |                  0.735 |               -0.007 |
| baseline__raw     | fe08__fare_per_family_member | svc           |                          0.827 |                        0.826 |                     -0.001 |                    0.76  |                  0.759 |               -0.001 |
| baseline__raw     | fe08__fare_per_family_member | decision_tree |                          0.803 |                        0.807 |                      0.004 |                    0.702 |                  0.711 |                0.009 |
| baseline__raw     | fe08__fare_per_family_member | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.732 |               -0.012 |
| baseline__raw     | fe08__fare_per_family_member | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | fe08__fare_per_family_member | xgb           |                          0.826 |                        0.823 |                     -0.003 |                    0.758 |                  0.752 |               -0.006 |

#### Summary

| compare_group                |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe08__fare_per_family_member |                    -0.000857143 |                         -0.006 |                          0.004 |               -0.00171429 |                   -0.012 |                    0.009 |

</details>

#### fe10__fare_per_ticket_member
 
Feature akin to Fare/family size, but based on ticket member instead. Expected to give better results them family size, since ticket member better represents the situation inside the dataframe. But this result can vary between train/test, as it only counts the passenger inside that dataframe.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003
  - knn: test_accuracy_mean: 0.005
  - decision_tree: test_accuracy_mean: 0.007
    - Secondary gains:
      - test_f1_mean: 0.023


##### Conclusion

Fare_per_ticket_member performed somewhat better than Fare_per_family_member, particularly for KNN and Decision Tree. This suggests that passengers sharing a ticket may provide a more useful approximation of the fare group than family relationships alone.

However, the results remain strongly model-dependent. Neither derived feature consistently outperformed the baseline across all models, so neither should replace raw Fare by default. At this stage, raw Fare remains the more dependable general representation, while Fare_per_ticket_member may be useful for selected models.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-----------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe10__fare_per_ticket_member | logreg        |                          0.786 |                        0.789 |                      0.003 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | fe10__fare_per_ticket_member | knn           |                          0.809 |                        0.814 |                      0.005 |                    0.742 |                  0.747 |                0.005 |
| baseline__raw     | fe10__fare_per_ticket_member | svc           |                          0.827 |                        0.823 |                     -0.004 |                    0.76  |                  0.755 |               -0.005 |
| baseline__raw     | fe10__fare_per_ticket_member | decision_tree |                          0.803 |                        0.81  |                      0.007 |                    0.702 |                  0.725 |                0.023 |
| baseline__raw     | fe10__fare_per_ticket_member | random_forest |                          0.822 |                        0.815 |                     -0.007 |                    0.744 |                  0.73  |               -0.014 |
| baseline__raw     | fe10__fare_per_ticket_member | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | fe10__fare_per_ticket_member | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.748 |               -0.01  |

##### Summary

| compare_group                |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe10__fare_per_ticket_member |                     0.000142857 |                         -0.007 |                          0.007 |               0.000428571 |                   -0.014 |                    0.023 |

</details>

#### Fare representation combinations

#### cb04__fare_and_fare_per_family

Combination exploring the effects of using both Fare and Fare/family. To find whether they bring additional information together, a new representation, or are redundant.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.008
    - Secondary gains:
      - test_f1_mean: 0.028


##### Conclusion

Keeping both Fare and Fare_per_family did not improve performance for most models. Since both features describe closely related information, the additional representation appears to introduce more redundancy than useful information for most learning algorithms.

Decision Tree was a notable exception, achieving the largest improvement among all models (+0.008 accuracy and +0.028 F1). This suggests that Decision Tree can exploit the additional split opportunities provided by the two Fare representations, even though other tree-based models did not obtain the same benefit.

Overall, Fare_per_family does not appear to be broadly useful when added alongside Fare, but it may still provide meaningful complementary information for specific tree-based models.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                  | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb04__fare_and_fare_per_family | logreg        |                          0.786 |                        0.785 |                     -0.001 |                    0.713 |                  0.711 |               -0.002 |
| baseline__raw     | cb04__fare_and_fare_per_family | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | cb04__fare_and_fare_per_family | svc           |                          0.827 |                        0.823 |                     -0.004 |                    0.76  |                  0.754 |               -0.006 |
| baseline__raw     | cb04__fare_and_fare_per_family | decision_tree |                          0.803 |                        0.811 |                      0.008 |                    0.702 |                  0.73  |                0.028 |
| baseline__raw     | cb04__fare_and_fare_per_family | random_forest |                          0.822 |                        0.82  |                     -0.002 |                    0.744 |                  0.739 |               -0.005 |
| baseline__raw     | cb04__fare_and_fare_per_family | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.724 |                0.003 |
| baseline__raw     | cb04__fare_and_fare_per_family | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.756 |               -0.002 |

##### Summary

| compare_group                  |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb04__fare_and_fare_per_family |                     -0.00114286 |                         -0.007 |                          0.008 |                0.00114286 |                   -0.008 |                    0.028 |

</details>

#### cb05__fare_and_fare_per_ticket

Combination exploring the use of Fare with Fare/Ticket, following the same idea as CB04. I'd expect results akin to Cb04, but am curious about its effect on Decision tree.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_negative
- Recommended for specific models:
- Notable secondary improvements in non-recommended models:
  - decision_tree: test_f1_mean: 0.012


##### Conclusion

Most models experienced either negligible changes or small performance losses. Decision Tree was the main exception: its accuracy increased by only 0.002, but its F1 score increased by 0.012, suggesting that the added representation affected its balance of survivor predictions more than its total number of correct predictions.

This is the second experiment suggesting that Decision Tree benefits from having multiple representations of the same underlying information. Although the evidence is still limited, the consistency across experiments makes this hypothesis increasingly plausible.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                  | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb05__fare_and_fare_per_ticket | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | cb05__fare_and_fare_per_ticket | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.739 |               -0.003 |
| baseline__raw     | cb05__fare_and_fare_per_ticket | svc           |                          0.827 |                        0.822 |                     -0.005 |                    0.76  |                  0.754 |               -0.006 |
| baseline__raw     | cb05__fare_and_fare_per_ticket | decision_tree |                          0.803 |                        0.805 |                      0.002 |                    0.702 |                  0.714 |                0.012 |
| baseline__raw     | cb05__fare_and_fare_per_ticket | random_forest |                          0.822 |                        0.815 |                     -0.007 |                    0.744 |                  0.73  |               -0.014 |
| baseline__raw     | cb05__fare_and_fare_per_ticket | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | cb05__fare_and_fare_per_ticket | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.759 |                0.001 |

##### Summary

| compare_group                  |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb05__fare_and_fare_per_ticket |                     -0.00185714 |                         -0.007 |                          0.002 |               -0.00114286 |                   -0.014 |                    0.012 |

</details>

#### cb06__all_fare_features

Combo experiment testing the effect of having all fare features together. The expected results is a configuration akin to CB04, roughly the same results or slightly worse, as too many representations of the same feature likely become redundant, diminishing its return. 

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.004
    - Secondary gains:
      - test_f1_mean: 0.025


##### Conclusion

Using all three Fare representations confirmed the pattern observed in the previous combination experiments. Most models either showed negligible changes or slight performance degradation, suggesting that the additional representations provide little information beyond what is already available through Fare itself.

Decision Tree remained the exception, achieving a meaningful improvement in F1 (+0.025) despite a modest accuracy gain (+0.004). This is the third consecutive experiment in which Decision Tree benefited from multiple representations of Fare, strengthening the hypothesis that Decision Tree configuration can exploit correlated representations more effectively than the other evaluated models.

However, the improvement was smaller than the one obtained with Fare + Fare_per_Family alone. This suggests that additional representations exhibit diminishing returns: once the most useful complementary information has been introduced, further correlated features become increasingly redundant.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb06__all_fare_features | logreg        |                          0.786 |                        0.786 |                      0     |                    0.713 |                  0.712 |               -0.001 |
| baseline__raw     | cb06__all_fare_features | knn           |                          0.809 |                        0.806 |                     -0.003 |                    0.742 |                  0.742 |                0     |
| baseline__raw     | cb06__all_fare_features | svc           |                          0.827 |                        0.82  |                     -0.007 |                    0.76  |                  0.752 |               -0.008 |
| baseline__raw     | cb06__all_fare_features | decision_tree |                          0.803 |                        0.807 |                      0.004 |                    0.702 |                  0.727 |                0.025 |
| baseline__raw     | cb06__all_fare_features | random_forest |                          0.822 |                        0.818 |                     -0.004 |                    0.744 |                  0.741 |               -0.003 |
| baseline__raw     | cb06__all_fare_features | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.72  |               -0.001 |
| baseline__raw     | cb06__all_fare_features | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.751 |               -0.007 |

##### Summary

| compare_group           |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb06__all_fare_features |                          -0.002 |                         -0.007 |                          0.004 |               0.000714286 |                   -0.008 |                    0.025 |

</details>

#### Overall conclusion

Neither Fare_per_family_member nor Fare_per_ticket_member consistently outperformed raw Fare across all models. Fare_per_ticket_member produced somewhat stronger individual results, suggesting that ticket groups may approximate shared fares better than family relationships, but its usefulness remained model-dependent.

For most models, combining raw Fare with one or more derived representations introduced redundancy without adding enough new information to improve performance. Raw Fare therefore remains the strongest general-purpose representation.

Decision Tree was the clear exception. It benefited in all three combination experiments, particularly when Fare was paired with Fare_per_family_member. However, adding Fare_per_ticket_member as a third representation did not improve on that result, indicating diminishing returns as the representations became increasingly redundant.

Overall, the Fare investigation shows that alternative representations can expose useful model-specific relationships, but more representations are not automatically better. Their usefulness depends both on the quality of the transformation and on the learning algorithm receiving it.

#### Findings

- Neither FamilySize nor TicketGroupSize provides a perfect estimate of the number of passengers covered by a fare.

- Fare_per_ticket_member was generally more useful than Fare_per_family_member, but neither derived feature consistently outperformed raw Fare.

- Raw Fare remains the most reliable general-purpose representation, while normalized Fare features are better treated as model-specific alternatives.

- Multiple representations of the same underlying information should not be assumed to outperform their individual components. They may provide useful split opportunities for some models while creating redundancy for others.

- Decision Tree consistently benefited from combining raw Fare with a normalized representation, but the same pattern did not extend to Random Forest, Extra Trees, or XGBoost.

- Adding a third Fare representation produced diminishing returns, suggesting that complementary information eventually gives way to redundancy.

#### hypotheses

- Decision Tree may benefit from receiving multiple representations of the same underlying feature because each representation provides different candidate split thresholds.

- The benefit appears to show diminishing returns once additional representations become strongly redundant. Future experiments in other feature domains are needed to determine whether this is a general Decision Tree pattern or specific to Fare.

#### Current recommendation

- Default for all models
  - Raw Fare

- Logistic Regression
  - Fare_per_family_member or Fare_per_ticket_member may be tested as alternatives, although gains were small.

- KNN
  - Fare_per_ticket_member

- Decision Tree
  - Fare + Fare_per_family_member

- SVC, Random Forest, Extra Trees, and XGBoost
  - Keep raw Fare; no derived or combined representation produced a meaningful improvement.


---

### Feature combination

#### Hypothesis

This investigation evaluates two complementary feature-engineering strategies.

The first is replacing multiple informative features with a single engineered representation. The second is augmenting the original feature set by keeping both the engineered feature and its source variables.

The objective is to determine whether engineered interactions replace existing information, complement it, or simply introduce redundancy.

#### Experiments performed:

#### fe12__sex_pclass

Proof of concept: Evaluate whether combining two informative features into a single feature can improve predictive performance or whether the models already learn this interaction naturally.

Sex and Pclass were chosen because they are among the strongest predictors in the dataset, increasing the likelihood that the experiment would produce a clear outcome rather than an ambiguous one.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.009
    - Secondary losses:
      - test_f1_mean: -0.02
  - extra_trees: test_accuracy_mean: 0.005
    - Secondary losses:
      - test_f1_mean: -0.019


##### Conclusion

Only Logistic Regression and Extra Trees showed a small improvement in accuracy after replacing Sex and Pclass with the combined Sex_Pclass feature. Most other models experienced a small decrease in accuracy.

More importantly, every model suffered a noticeable reduction in F1 score (roughly -0.02 or more, with KNN being the only minor exception). This suggests that, although the combined feature may slightly improve overall accuracy for some models, it also leads to poorer balance between precision and recall. Even though the Titanic competition is evaluated solely on accuracy, this trade-off makes the usefulness of this feature questionable.

A likely explanation is that most of these models are already capable of learning the interaction between Sex and Pclass. Replacing the original variables with a single combined feature removes flexibility from the model, preventing it from exploiting each variable independently and potentially reducing its ability to generalize.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group    | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe12__sex_pclass | logreg        |                          0.786 |                        0.795 |                      0.009 |                    0.713 |                  0.693 |               -0.02  |
| baseline__raw     | fe12__sex_pclass | knn           |                          0.809 |                        0.806 |                     -0.003 |                    0.742 |                  0.741 |               -0.001 |
| baseline__raw     | fe12__sex_pclass | svc           |                          0.827 |                        0.824 |                     -0.003 |                    0.76  |                  0.732 |               -0.028 |
| baseline__raw     | fe12__sex_pclass | decision_tree |                          0.803 |                        0.794 |                     -0.009 |                    0.702 |                  0.68  |               -0.022 |
| baseline__raw     | fe12__sex_pclass | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.717 |               -0.027 |
| baseline__raw     | fe12__sex_pclass | extra_trees   |                          0.804 |                        0.809 |                      0.005 |                    0.721 |                  0.702 |               -0.019 |
| baseline__raw     | fe12__sex_pclass | xgb           |                          0.826 |                        0.82  |                     -0.006 |                    0.758 |                  0.743 |               -0.015 |

##### Summary

| compare_group    |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe12__sex_pclass |                     -0.00185714 |                         -0.009 |                          0.009 |                -0.0188571 |                   -0.028 |                   -0.001 |

</details>

#### cb08__sex_pclass_features

Experiment exploring whether using combined features with their original ones leads to gains, or just adds noise.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.016


##### Conclusion

Unlike fe12, this experiment preserves both the original variables and the engineered interaction.

The results suggest that Sex_Pclass contains useful predictive information, but not enough to replace Sex and Pclass. Instead, keeping all three representations preserves the flexibility of the original variables while allowing some models to exploit the explicit interaction.

Logistic Regression showed the largest improvement (+0.016 accuracy) without the substantial F1 loss observed in fe12, indicating that simpler models benefit from receiving the engineered interaction explicitly. Most other models experienced only minor changes, suggesting they already learn this relationship internally.

Overall, Sex_Pclass is better viewed as a complementary feature than as a replacement for its source variables.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group             | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:--------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb08__pclass_sex_features | logreg        |                          0.786 |                        0.802 |                      0.016 |                    0.713 |                  0.713 |                0     |
| baseline__raw     | cb08__pclass_sex_features | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.738 |               -0.004 |
| baseline__raw     | cb08__pclass_sex_features | svc           |                          0.827 |                        0.827 |                      0     |                    0.76  |                  0.75  |               -0.01  |
| baseline__raw     | cb08__pclass_sex_features | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.702 |                0     |
| baseline__raw     | cb08__pclass_sex_features | random_forest |                          0.822 |                        0.817 |                     -0.005 |                    0.744 |                  0.73  |               -0.014 |
| baseline__raw     | cb08__pclass_sex_features | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.698 |               -0.023 |
| baseline__raw     | cb08__pclass_sex_features | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.753 |               -0.005 |

##### Summary

| compare_group             |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:--------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb08__pclass_sex_features |                      0.00157143 |                         -0.005 |                          0.016 |                    -0.008 |                   -0.023 |                        0 |

</details>

#### Overall conclusion

Three feature interaction studies (Family, Age, and Sex/Pclass) revealed that the usefulness of engineered feature interactions depends not only on the interaction itself, but also on how it is incorporated into the feature set.

Replacing informative variables with engineered interactions frequently reduced model flexibility and degraded performance. In contrast, augmenting the original representation sometimes produced modest but consistent improvements, particularly for simpler models such as Logistic Regression.

These experiments suggest that feature engineering should be viewed as a strategy for enriching the feature space rather than necessarily replacing existing variables.

#### Findings

- Replacing informative features with engineered interactions often reduces model flexibility and degrades performance.

- Engineered interaction features are generally more useful as complementary representations than as direct replacements.

- Logistic Regression consistently benefited from explicit engineered representations.

- Most tree-based models already learn many feature interactions internally and therefore gained less from explicit interaction features.

- Whether an engineered feature should replace or augment its source variables depends on both the feature and the learning algorithm.

#### Open hypotheses

- Does the benefit of explicit interaction features decrease as model complexity increases?

- Would automatically learned interaction features outperform manually designed ones?

- Which feature interactions genuinely introduce new information, and which merely duplicate information already available to the model?

- Can feature importance or SHAP analysis explain why Logistic Regression consistently benefits more from engineered representations than tree-based models?

#### Current recommendation

- Logistic Regression
    - Use the combined and original representations together (Sex, Pclass, and Sex_Pclass).
    - The combined representation consistently improved performance without the large F1 degradation observed when replacing the original variables.

- All other models
    - Prefer the original Sex and Pclass features.
    - The explicit interaction provides little additional benefit and may unnecessarily increase feature redundancy.

---

### Feature investigation closure

The feature-combination investigation marks the end of the exploratory
feature-engineering phase of this case study.

Additional experiments are still possible, particularly interactions between
features from unrelated domains. However, the experiments performed so far
already demonstrate the major behaviors this study set out to investigate:
features may be useful as replacements, complementary representations, or
model-specific transformations, while additional combinations frequently show
diminishing returns or introduce redundancy.

At this point, further feature exploration is expected to provide relatively
small additional learning compared with the effort required.

The project therefore moves from exploration to model construction. The
findings from the feature investigations will now be used to assemble a
model-specific feature configuration for each algorithm. These configurations
will be evaluated before the strongest candidates proceed to hyperparameter
tuning.

## Final experiments

The exploratory experiments evaluated individual feature-engineering ideas in
isolation. The final experiments test whether the most promising
model-specific modifications continue to provide value when combined into a
single configuration.

This stage therefore serves as the bridge between feature investigation and
model optimization.

### Selecting candidate features

Final configurations are constructed from the baseline configuration together
with the strongest acceptable experiment from each feature domain.

Because the Titanic competition evaluates accuracy, accuracy is used as the
primary selection metric. F1 is retained as a secondary guardrail to avoid
selecting configurations whose accuracy gains are accompanied by substantial
losses in classification balance.

An experiment is considered a candidate when:

- its cross-validation accuracy improves by at least **+0.003** relative to the
  baseline;
- its F1 score does not decrease by more than **-0.010**;
- it represents a canonical feature-engineering experiment rather than an
  ablation.

At most one experiment is selected from each feature domain.

A helper function applies these rules to the stored experiment results and
returns the strongest qualifying candidate for each domain. The resulting
recommendations are then reviewed manually against the corresponding feature
investigation before the final configuration is assembled.

Diagnostic ablations are excluded from automatic candidate selection because their purpose is to measure the marginal contribution of an existing feature rather than propose an alternative feature representation. Their findings may still influence manual decisions when assembling the final configurations.

| Model               | Selected experiment candidates                                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | `fe05__title`, `cb03__age_imputed_title_Pclass_and_bins`, `cb08__pclass_sex_features`, `fe01__family`, `fe04__cabin_features`, `fe08__fare_per_family_member` |
| KNN                 | `fe05__title`, `fe03__deck`, `fe10__fare_per_ticket_member`                                                                                                   |
| SVC                 | `fe05__title`, `fe09__ticket_group_size`, `cb02__age_imputed_title_and_bins`                                                                                  |
| Decision Tree       | `fe05__title`, `cb04__fare_and_fare_per_family`, `fe11__age_bin`, `cb07__family_features`                                                                     |
| Random Forest       | `fe11__age_bin`, `fe05__title`                                                                                                                                |
| Extra Trees         | `fe05__title`, `cb03__age_imputed_title_Pclass_and_bins`, `fe02__has_cabin`, `cb07__family_features`, `fe09__ticket_group_size`                               |
| XGBoost             | `fe05__title`, `fe04__cabin_features`, `cb07__family_features`                                                                                                |
These experiments identify candidate modifications rather than complete configurations. Their feature-engineering and preprocessing requirements are manually reconciled before the final experiments are executed.

## Lessons learned

- Recovering missing information (cabin, Age Imputation).
- finding hidden information (title).
- Changing representation of existing information (age_bin)

Hypothesis ✓ Confirmed
Hypothesis ✗ Rejected
Hypothesis ~ Partially confirmed