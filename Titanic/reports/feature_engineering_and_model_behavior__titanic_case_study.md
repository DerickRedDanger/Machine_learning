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
| experiment                                       | model_name    |   test_accuracy_mean |   test_f1_mean |
|:-------------------------------------------------|:--------------|---------------------:|---------------:|
| fe05__title__xgb                                 | xgb           |                0.836 |          0.772 |
| cb05__fare_and_fare_per_ticket_full_context__xgb | xgb           |                0.834 |          0.772 |
| fe05__title__svc                                 | svc           |                0.834 |          0.771 |
| fe09__ticket_group_size_full_context__svc        | svc           |                0.833 |          0.772 |
| fe05__title__random_forest                       | random_forest |                0.832 |          0.768 |
| fe04__cabin_features__xgb                        | xgb           |                0.832 |          0.767 |
| cb02__age_imputed_title_and_bins__svc            | svc           |                0.831 |          0.767 |
| cb03__age_imputed_title_pclass_and_bins__svc     | svc           |                0.831 |          0.767 |
| fe09__ticket_group_size_fitted__svc              | svc           |                0.831 |          0.767 |
| fe06__age_imputation_title__svc                  | svc           |                0.829 |          0.764 |

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

#### cb03__age_imputed_title_pclass_and_bins

This experiment combines Title-and-Pclass-based Age imputation from FE07 with the continuous-and-binned Age representation explored in the previous combination experiments.

Missing Age values are estimated from passengers sharing both Title and Pclass. The resulting Age is retained as a continuous feature while also being transformed into the ordinal Age_bin representation.

This tests whether combining a more context-specific Age estimate with multiple representations provides additional predictive value.

<details>
<summary>Conclusion</summary>


##### Interpretation

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


##### Conclusion

Title-and-Pclass-imputed Age combined with Age_bin is a strong model-specific Age representation, particularly for Logistic Regression, where it produces the best Age-related result observed so far.

Extra Trees also benefits substantially from the combined representation, reinforcing the evidence that continuous and ordinal Age provide complementary predictive structure for this model. Random Forest and SVC show smaller positive effects.

The experiment further demonstrates that increasing preprocessing complexity is not universally beneficial: the same representation that substantially helps Logistic Regression and Extra Trees is detrimental to KNN and provides no advantage to Decision Tree or XGBoost.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | logreg        |                          0.786 |                        0.806 |                      0.02  |                    0.713 |                  0.735 |                0.022 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | knn           |                          0.809 |                        0.798 |                     -0.011 |                    0.742 |                  0.73  |               -0.012 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | random_forest |                          0.822 |                        0.828 |                      0.006 |                    0.744 |                  0.754 |                0.01  |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | extra_trees   |                          0.804 |                        0.82  |                      0.016 |                    0.721 |                  0.747 |                0.026 |
| baseline__raw     | cb03__age_imputed_title_pclass_and_bins | xgb           |                          0.826 |                        0.824 |                     -0.002 |                    0.758 |                  0.756 |               -0.002 |

##### Summary

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
`Ticket` may contain information about passengers travelling as part of the same
group. A `TicketGroupSize` feature was therefore created by counting passengers
sharing the same ticket.

#### Prediction context

Unlike most engineered features in this study, however, `TicketGroupSize` is
population-dependent: its value changes according to which other passengers
are available when the feature is constructed.

For example, a passenger belonging to a four-person ticket group may appear to
have a group size of one if the other members are absent from the population
used to construct the feature. This makes the information boundary used to
calculate `TicketGroupSize` part of the feature's meaning.

To investigate this, three strategies were compared:

- **Batch context:** group size is calculated independently from the rows in
  each transformed batch. During cross-validation, training and validation
  folds therefore observe their own ticket groups independently.

- **Fitted:** ticket counts are learned from the training fold and mapped onto
  validation rows. Tickets not observed during fitting are treated as singleton
  groups (`TicketGroupSize = 1`).

- **Full prediction context:** ticket counts are calculated using the complete
  feature-only population available at prediction time. For this Titanic case
  study, this consists of the training population together with the supplied
  prediction/test population. No target information from the prediction
  population is used.

<details>
<summary>Comparison of TicketGroupSize strategies</summary>

| Model | Batch ΔAcc | Batch ΔF1 | Fitted ΔAcc | Fitted ΔF1 | Full context ΔAcc | Full context ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | +0.003 | +0.003 | +0.001 | 0.000 | -0.003 | -0.003 |
| KNN | -0.010 | -0.014 | 0.000 | -0.001 | 0.000 | +0.005 |
| SVC | -0.003 | 0.000 | +0.004 | +0.007 | +0.006 | +0.012 |
| Decision Tree | -0.002 | +0.004 | +0.002 | +0.007 | +0.005 | +0.014 |
| Random Forest | -0.012 | -0.011 | -0.006 | -0.004 | -0.006 | -0.005 |
| Extra Trees | -0.007 | -0.006 | -0.002 | +0.001 | +0.002 | +0.004 |
| XGBoost | -0.001 | 0.000 | 0.000 | +0.001 | -0.004 | -0.004 |

</details>

<details>
<summary>Individual experiment results</summary>

#### fe09__ticket_group_size_batch
<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                 | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe09__ticket_group_size_batch | logreg        |                          0.786 |                        0.789 |                      0.003 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | fe09__ticket_group_size_batch | knn           |                          0.809 |                        0.799 |                     -0.01  |                    0.742 |                  0.728 |               -0.014 |
| baseline__raw     | fe09__ticket_group_size_batch | svc           |                          0.827 |                        0.824 |                     -0.003 |                    0.76  |                  0.76  |                0     |
| baseline__raw     | fe09__ticket_group_size_batch | decision_tree |                          0.803 |                        0.801 |                     -0.002 |                    0.702 |                  0.706 |                0.004 |
| baseline__raw     | fe09__ticket_group_size_batch | random_forest |                          0.822 |                        0.81  |                     -0.012 |                    0.744 |                  0.733 |               -0.011 |
| baseline__raw     | fe09__ticket_group_size_batch | extra_trees   |                          0.804 |                        0.797 |                     -0.007 |                    0.721 |                  0.715 |               -0.006 |
| baseline__raw     | fe09__ticket_group_size_batch | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.758 |                0     |

##### Summary

| compare_group                 |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe09__ticket_group_size_batch |                     -0.00457143 |                         -0.012 |                          0.003 |               -0.00342857 |                   -0.014 |                    0.004 |

</details>

#### fe09__ticket_group_size_fitted

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - svc: test_accuracy_mean: 0.004

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                  | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe09__ticket_group_size_fitted | logreg        |                          0.786 |                        0.787 |                      0.001 |                    0.713 |                  0.713 |                0     |
| baseline__raw     | fe09__ticket_group_size_fitted | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.741 |               -0.001 |
| baseline__raw     | fe09__ticket_group_size_fitted | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | fe09__ticket_group_size_fitted | decision_tree |                          0.803 |                        0.805 |                      0.002 |                    0.702 |                  0.709 |                0.007 |
| baseline__raw     | fe09__ticket_group_size_fitted | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.74  |               -0.004 |
| baseline__raw     | fe09__ticket_group_size_fitted | extra_trees   |                          0.804 |                        0.802 |                     -0.002 |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | fe09__ticket_group_size_fitted | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.759 |                0.001 |

##### Summary

| compare_group                  |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe09__ticket_group_size_fitted |                    -0.000142857 |                         -0.006 |                          0.004 |                0.00157143 |                   -0.004 |                    0.007 |

</details>

#### fe09__ticket_group_size_full_context

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - svc: test_accuracy_mean: 0.006
    - Secondary gains:
      - test_f1_mean: 0.012
  - decision_tree: test_accuracy_mean: 0.005
    - Secondary gains:
      - test_f1_mean: 0.014

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                        | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe09__ticket_group_size_full_context | logreg        |                          0.786 |                        0.783 |                     -0.003 |                    0.713 |                  0.71  |               -0.003 |
| baseline__raw     | fe09__ticket_group_size_full_context | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.747 |                0.005 |
| baseline__raw     | fe09__ticket_group_size_full_context | svc           |                          0.827 |                        0.833 |                      0.006 |                    0.76  |                  0.772 |                0.012 |
| baseline__raw     | fe09__ticket_group_size_full_context | decision_tree |                          0.803 |                        0.808 |                      0.005 |                    0.702 |                  0.716 |                0.014 |
| baseline__raw     | fe09__ticket_group_size_full_context | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.739 |               -0.005 |
| baseline__raw     | fe09__ticket_group_size_full_context | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.725 |                0.004 |
| baseline__raw     | fe09__ticket_group_size_full_context | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.754 |               -0.004 |

##### Summary

| compare_group                        |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe09__ticket_group_size_full_context |                               0 |                         -0.006 |                          0.006 |                0.00328571 |                   -0.005 |                    0.014 |

</details>

</details>

#### Interpretation

The usefulness of `TicketGroupSize` depends substantially on how its population
context is defined.

The batch-context representation is generally the weakest. It reduces mean
accuracy and F1 across the tested models, with particularly negative effects
for KNN, Random Forest, and Extra Trees. This suggests that independently
fragmenting ticket groups between transformed batches produces a representation
that is poorly aligned with the underlying group information.

The fitted representation largely removes this penalty. Most models remain
close to baseline, while SVC improves by +0.004 accuracy and +0.007 F1 and
Decision Tree improves by +0.002 accuracy and +0.007 F1. This indicates that
TicketGroupSize can contain useful predictive information even under an
inductive setting, although its benefit remains model-specific.

Full prediction context produces the strongest useful effects. SVC improves by
+0.006 accuracy and +0.012 F1, while Decision Tree improves by +0.005 accuracy
and +0.014 F1. Extra Trees and KNN also show small F1 improvements. Logistic
Regression, Random Forest, and XGBoost instead perform worse than baseline,
showing that a more complete group representation does not make the feature
universally beneficial.

For SVC and Decision Tree in particular, performance improves progressively
from batch to fitted to full-context representations. This supports the
interpretation that these models can exploit ticket-group information, but
that the quality and completeness of the group representation matter.

#### Ticket groups vs family groups

`TicketGroupSize` was initially expected to behave similarly to the engineered
family features, since both can act as proxies for the number of passengers
travelling together. Their observed model effects, however, differ substantially.

<details>
<summary>Family features vs full-context TicketGroupSize</summary>

| Model | Family ΔAcc | Family ΔF1 | Ticket ΔAcc | Ticket ΔF1 |
|---|---:|---:|---:|---:|
| Logistic Regression | +0.004 | +0.006 | -0.003 | -0.003 |
| KNN | -0.008 | -0.009 | 0.000 | +0.005 |
| SVC | +0.001 | +0.003 | +0.006 | +0.012 |
| Decision Tree | +0.004 | +0.012 | +0.005 | +0.014 |
| Random Forest | -0.003 | -0.001 | -0.006 | -0.005 |
| Extra Trees | +0.002 | +0.005 | +0.002 | +0.004 |
| XGBoost | +0.001 | +0.003 | -0.004 | -0.004 |

</details>




Although the two representations overlap conceptually, the contrasting model
responses indicate that they are not interchangeable proxies. Family features
describe the passenger's recorded family structure (`SibSp` and `Parch`),
whereas shared tickets describe a different grouping relationship. The latter
may capture travelling groups that do not correspond directly to the recorded
family unit.

This distinction is especially visible for Logistic Regression and SVC:
family features improve Logistic Regression while full-context TicketGroupSize
hurts it, whereas SVC benefits considerably more from TicketGroupSize than from
the family representation.

The comparison therefore suggests that family structure and ticket-group
structure provide partially distinct information about passenger relationships
rather than alternative measurements of the same underlying feature.

#### Conclusion

`TicketGroupSize` is a useful example of a relational feature whose meaning
depends on the population available when it is constructed.

Batch-local construction is generally unreliable because ticket groups are
fragmented by the evaluation batches. Fitted counts provide a valid inductive
alternative and modest model-specific gains, while complete prediction context
can strengthen the signal when that population is legitimately available.

For this dataset, SVC and Decision Tree benefit most from a more complete
ticket-group representation. The feature remains detrimental or neutral for
several other models, so it should be selected according to both the model and
the intended prediction context rather than treated as a universally useful
engineered feature.

Full-context results should also be interpreted specifically under the assumed
deployment setting: they represent performance when the complete feature-only
prediction population is available, rather than ordinary prediction of
independent future observations.

#### Findings

- `TicketGroupSize` is strongly dependent on the population used to construct it. The three context strategies produced meaningfully different model behavior despite representing the same underlying feature concept.
- Batch-local `TicketGroupSize` was generally detrimental, with mean changes of -0.0046 accuracy and -0.0034 F1. Fragmenting ticket groups between independently transformed batches appears to weaken the representation.
- Fitted `TicketGroupSize` was substantially more stable, remaining near baseline overall while providing useful gains for SVC (+0.004 accuracy, +0.007 F1) and smaller gains for Decision Tree (+0.002 accuracy, +0.007 F1).
- Full-context `TicketGroupSize` produced the strongest model-specific gains, particularly for SVC (+0.006 accuracy, +0.012 F1) and Decision Tree (+0.005 accuracy, +0.014 F1).
- For SVC and Decision Tree, performance improved progressively from batch → fitted → full context, suggesting that these models benefit from increasingly complete ticket-group information.
- More complete context does not make `TicketGroupSize` universally useful. Full context remained detrimental for Logistic Regression, Random Forest, and XGBoost.
- `TicketGroupSize` and the engineered family representation are not interchangeable proxies. Their contrasting effects, particularly for Logistic Regression and SVC, indicate that ticket groups and recorded family groups capture partially distinct passenger relationships.

#### Hypotheses

- Batch-local counting weakens `TicketGroupSize` because the same underlying travel group can receive different representations depending on how its members are distributed between batches.
- Fitted counting preserves a more stable historical group signal, explaining why it removes much of the degradation observed with batch-local counting.
- Full prediction context provides the most complete representation of the actual ticket groups and may therefore expose relational structure that is partially lost under fitted or batch-local strategies.
- SVC and Decision Tree may be particularly sensitive to the quality of this group representation, allowing them to exploit TicketGroupSize once the signal becomes sufficiently stable.
- FamilySize and TicketGroupSize likely represent different forms of social structure: FamilySize describes recorded family relationships, while TicketGroupSize more broadly represents a shared travel/booking group. Their overlap is therefore only partial.
- The usefulness of full-context TicketGroupSize may propagate more strongly into features that mathematically depend on it, particularly `Fare/TicketGroupSize`, where an incomplete group count directly changes the scale of the resulting feature.

#### Current recommendation

- Do not use batch-local `TicketGroupSize` as the preferred representation. Its overall behavior is unstable and generally worse than the alternatives.
- Keep fitted `TicketGroupSize` as the default inductive representation when predictions must generalize to observations without access to the complete prediction population.
- Consider full-context `TicketGroupSize` when the complete feature-only prediction population is legitimately available. Its results are especially promising for SVC and Decision Tree, but should be interpreted specifically under this context-aware deployment assumption.
- Do not treat TicketGroupSize as a replacement for FamilySize. The two representations appear to capture complementary rather than equivalent relationship information.
- Carry all three TicketGroupSize strategies into the `Fare/TicketGroupSize` investigation to test how context semantics propagate through a derived ratio.
- Defer testing FamilySize and TicketGroupSize together until model-specific feature selection provides a reason to investigate their potential complementarity.

---

### Fare

`Fare` may not represent an individual passenger's effective fare. Multiple
passengers travelling together may share a booking or ticket, making the raw
value partly dependent on group structure.

Two previously engineered group representations provide different ways to
estimate this structure:

- `FamilySize` represents the passenger's recorded family group.
- `TicketGroupSize` represents passengers sharing the same ticket.

The Ticket investigation showed that these representations are not
interchangeable and that `TicketGroupSize` additionally depends on the
population context used to construct it.

The Fare investigation therefore evaluates three related questions:

1. Can group-normalized Fare provide a useful alternative to raw `Fare`?
2. Are raw and normalized Fare complementary?
3. Can Family- and Ticket-normalized Fare provide complementary representations
   when used together?

For Ticket-based normalization, the batch, fitted, and full-prediction-context
strategies are evaluated separately.

---

#### Normalized Fare

Raw `Fare` was replaced by a normalized Fare calculated using either
`FamilySize` or `TicketGroupSize`.

Because an incomplete TicketGroupSize changes the denominator rather than
merely the value of an independent feature, differences between Ticket context
strategies may propagate into the meaning and scale of the resulting Fare
representation.

<details>
<summary>Comparison of normalized Fare representations</summary>

| Model | Fare/Family ΔAcc | ΔF1 | Fare/Ticket Batch ΔAcc | ΔF1 | Fitted ΔAcc | ΔF1 | Full Context ΔAcc | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | +0.003 | +0.004 | +0.001 | +0.002 | +0.002 | +0.003 | +0.003 | +0.004 |
| KNN | -0.005 | -0.007 | -0.004 | -0.008 | -0.004 | -0.008 | +0.011 | +0.013 |
| SVC | -0.001 | -0.001 | -0.005 | -0.008 | -0.001 | 0.000 | 0.000 | +0.001 |
| Decision Tree | +0.004 | +0.009 | -0.011 | +0.004 | -0.007 | +0.005 | -0.001 | +0.012 |
| Random Forest | -0.006 | -0.012 | -0.013 | -0.012 | -0.004 | -0.003 | +0.001 | -0.002 |
| Extra Trees | +0.002 | +0.001 | -0.005 | -0.004 | 0.000 | +0.001 | +0.001 | -0.001 |
| XGBoost | -0.003 | -0.006 | -0.015 | -0.020 | -0.006 | -0.009 | +0.001 | +0.002 |

</details>


<details>
<summary>Comparison of normalized Fare representations</summary>

| Model | Fare/Family ΔAcc | ΔF1 | Fare/Ticket Batch ΔAcc | ΔF1 | Fitted ΔAcc | ΔF1 | Full Context ΔAcc | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | +0.003 | +0.004 | +0.001 | +0.002 | +0.002 | +0.003 | +0.003 | +0.004 |
| KNN | -0.005 | -0.007 | -0.004 | -0.008 | -0.004 | -0.008 | +0.011 | +0.013 |
| SVC | -0.001 | -0.001 | -0.005 | -0.008 | -0.001 | 0.000 | 0.000 | +0.001 |
| Decision Tree | +0.004 | +0.009 | -0.011 | +0.004 | -0.007 | +0.005 | -0.001 | +0.012 |
| Random Forest | -0.006 | -0.012 | -0.013 | -0.012 | -0.004 | -0.003 | +0.001 | -0.002 |
| Extra Trees | +0.002 | +0.001 | -0.005 | -0.004 | 0.000 | +0.001 | +0.001 | -0.001 |
| XGBoost | -0.003 | -0.006 | -0.015 | -0.020 | -0.006 | -0.009 | +0.001 | +0.002 |

</details>


<details>
<summary>Individual experiment results</summary>

#### fe08__fare_per_family_member

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003
  - decision_tree: test_accuracy_mean: 0.004


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

##### Summary

| compare_group                |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe08__fare_per_family_member |                    -0.000857143 |                         -0.006 |                          0.004 |               -0.00171429 |                   -0.012 |                    0.009 |

</details>


#### fe10__fare_per_ticket_member_batch

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_negative
- Recommended for specific models:

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                      | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-----------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe10__fare_per_ticket_member_batch | logreg        |                          0.786 |                        0.787 |                      0.001 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | svc           |                          0.827 |                        0.822 |                     -0.005 |                    0.76  |                  0.752 |               -0.008 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | decision_tree |                          0.803 |                        0.792 |                     -0.011 |                    0.702 |                  0.706 |                0.004 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | random_forest |                          0.822 |                        0.809 |                     -0.013 |                    0.744 |                  0.732 |               -0.012 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | extra_trees   |                          0.804 |                        0.799 |                     -0.005 |                    0.721 |                  0.717 |               -0.004 |
| baseline__raw     | fe10__fare_per_ticket_member_batch | xgb           |                          0.826 |                        0.811 |                     -0.015 |                    0.758 |                  0.738 |               -0.02  |

##### Summary

| compare_group                      |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe10__fare_per_ticket_member_batch |                     -0.00742857 |                         -0.015 |                          0.001 |               -0.00657143 |                    -0.02 |                    0.004 |

</details>

#### fe10__fare_per_ticket_member_fitted

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_negative
- Recommended for specific models:

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                       | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe10__fare_per_ticket_member_fitted | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | svc           |                          0.827 |                        0.826 |                     -0.001 |                    0.76  |                  0.76  |                0     |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | decision_tree |                          0.803 |                        0.796 |                     -0.007 |                    0.702 |                  0.707 |                0.005 |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | random_forest |                          0.822 |                        0.818 |                     -0.004 |                    0.744 |                  0.741 |               -0.003 |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | fe10__fare_per_ticket_member_fitted | xgb           |                          0.826 |                        0.82  |                     -0.006 |                    0.758 |                  0.749 |               -0.009 |

##### Summary

| compare_group                       |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe10__fare_per_ticket_member_fitted |                     -0.00285714 |                         -0.007 |                          0.002 |               -0.00157143 |                   -0.009 |                    0.005 |

</details>

#### fe10__fare_per_ticket_member_full_context

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_positive
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003
  - knn: test_accuracy_mean: 0.011
    - Secondary gains:
      - test_f1_mean: 0.013
- Notable secondary improvements in non-recommended models:
  - decision_tree: test_f1_mean: 0.012

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                             | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe10__fare_per_ticket_member_full_context | logreg        |                          0.786 |                        0.789 |                      0.003 |                    0.713 |                  0.717 |                0.004 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | knn           |                          0.809 |                        0.82  |                      0.011 |                    0.742 |                  0.755 |                0.013 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | svc           |                          0.827 |                        0.827 |                      0     |                    0.76  |                  0.761 |                0.001 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | decision_tree |                          0.803 |                        0.802 |                     -0.001 |                    0.702 |                  0.714 |                0.012 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | random_forest |                          0.822 |                        0.823 |                      0.001 |                    0.744 |                  0.742 |               -0.002 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.72  |               -0.001 |
| baseline__raw     | fe10__fare_per_ticket_member_full_context | xgb           |                          0.826 |                        0.827 |                      0.001 |                    0.758 |                  0.76  |                0.002 |

##### Summary

| compare_group                             |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe10__fare_per_ticket_member_full_context |                      0.00228571 |                         -0.001 |                          0.011 |                0.00414286 |                   -0.002 |                    0.013 |

</details>

</details>


#### Interpretation

Fare normalization is highly dependent on both the definition of passenger
grouping and the model consuming the resulting feature.

`Fare/FamilySize` is model-specific rather than generally beneficial. Logistic
Regression improves modestly, while Decision Tree receives the clearest
benefit (+0.004 accuracy, +0.009 F1). Several other models instead lose
performance.

Ticket-based normalization shows a stronger dependency on population context.
Batch-local normalization performs worst overall, while fitted counts reduce
much of its negative effect. Full prediction context reverses this pattern and
is the only tested normalization with positive mean changes in both accuracy
and F1.

KNN provides the clearest example. `Fare/TicketGroupSize` is detrimental under
both batch and fitted construction (-0.004 accuracy, -0.008 F1), but improves
by +0.011 accuracy and +0.013 F1 under full prediction context.

The separation between Ticket strategies is larger for normalized Fare than
for `TicketGroupSize` alone. An incomplete group count does not merely alter a
relational feature: when used as a denominator, it changes the scale of the
resulting Fare representation. Population semantics can therefore propagate
into downstream engineered features.

Family- and Ticket-normalized Fare also continue to produce different model
responses, supporting the earlier finding that family and ticket groups are
not interchangeable definitions of passenger grouping.

#### Interim conclusion

Replacing raw Fare with a normalized representation is not universally
beneficial. Family normalization provides useful model-specific information,
particularly for Decision Tree, while Ticket normalization depends strongly on
the completeness of the ticket-group representation.

Full-context `Fare/TicketGroupSize` is the strongest standalone Fare
normalization tested, producing a substantial improvement for KNN and no large
detriment for the remaining models.

---

#### Raw and normalized Fare

Normalization may preserve different information rather than provide a
complete replacement for raw `Fare`. Raw Fare was therefore retained alongside
each normalized representation to test whether the two forms are
complementary.


<details>
<summary>Comparison of raw + normalized Fare representations</summary>

| Model | Fare + Family ΔAcc | ΔF1 | Fare + Ticket Batch ΔAcc | ΔF1 | Fitted ΔAcc | ΔF1 | Full Context ΔAcc | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | -0.001 | -0.002 | +0.002 | +0.002 | +0.002 | +0.002 | +0.003 | +0.005 |
| KNN | -0.007 | -0.008 | -0.005 | -0.009 | -0.007 | -0.007 | +0.004 | +0.010 |
| SVC | -0.004 | -0.006 | -0.012 | -0.017 | -0.005 | -0.006 | -0.002 | -0.002 |
| Decision Tree | +0.008 | +0.028 | -0.012 | +0.004 | -0.006 | +0.009 | +0.008 | +0.032 |
| Random Forest | -0.005 | -0.008 | -0.013 | -0.013 | -0.005 | -0.004 | -0.002 | -0.003 |
| Extra Trees | +0.001 | -0.002 | -0.005 | -0.008 | +0.002 | 0.000 | +0.002 | +0.001 |
| XGBoost | -0.009 | -0.009 | -0.007 | -0.005 | -0.002 | -0.001 | +0.008 | +0.014 |

</details>

<details>
<summary>Individual experiment results</summary>

#### cb04__fare_and_fare_per_family

_Description pending._

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.008
    - Secondary gains:
      - test_f1_mean: 0.028


##### Conclusion

_Conclusion pending._

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
| baseline__raw     | cb04__fare_and_fare_per_family | random_forest |                          0.822 |                        0.817 |                     -0.005 |                    0.744 |                  0.736 |               -0.008 |
| baseline__raw     | cb04__fare_and_fare_per_family | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.719 |               -0.002 |
| baseline__raw     | cb04__fare_and_fare_per_family | xgb           |                          0.826 |                        0.817 |                     -0.009 |                    0.758 |                  0.749 |               -0.009 |

##### Summary

| compare_group                  |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb04__fare_and_fare_per_family |                     -0.00242857 |                         -0.009 |                          0.008 |                    -0.001 |                   -0.009 |                    0.028 |

</details>

#### cb05__fare_and_fare_per_ticket_batch


<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_negative
- Recommended for specific models:

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                        | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | knn           |                          0.809 |                        0.804 |                     -0.005 |                    0.742 |                  0.733 |               -0.009 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | svc           |                          0.827 |                        0.815 |                     -0.012 |                    0.76  |                  0.743 |               -0.017 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | decision_tree |                          0.803 |                        0.791 |                     -0.012 |                    0.702 |                  0.706 |                0.004 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | random_forest |                          0.822 |                        0.809 |                     -0.013 |                    0.744 |                  0.731 |               -0.013 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | extra_trees   |                          0.804 |                        0.799 |                     -0.005 |                    0.721 |                  0.713 |               -0.008 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_batch | xgb           |                          0.826 |                        0.819 |                     -0.007 |                    0.758 |                  0.753 |               -0.005 |

##### Summary

| compare_group                        |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb05__fare_and_fare_per_ticket_batch |                     -0.00742857 |                         -0.013 |                          0.002 |               -0.00657143 |                   -0.017 |                    0.004 |

</details>

#### cb05__fare_and_fare_per_ticket_fitted

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_negative
- Recommended for specific models:

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                         | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:--------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.735 |               -0.007 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | svc           |                          0.827 |                        0.822 |                     -0.005 |                    0.76  |                  0.754 |               -0.006 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | decision_tree |                          0.803 |                        0.797 |                     -0.006 |                    0.702 |                  0.711 |                0.009 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | random_forest |                          0.822 |                        0.817 |                     -0.005 |                    0.744 |                  0.74  |               -0.004 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.721 |                0     |
| baseline__raw     | cb05__fare_and_fare_per_ticket_fitted | xgb           |                          0.826 |                        0.824 |                     -0.002 |                    0.758 |                  0.757 |               -0.001 |

##### Summary

| compare_group                         |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:--------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb05__fare_and_fare_per_ticket_fitted |                          -0.003 |                         -0.007 |                          0.002 |                    -0.001 |                   -0.007 |                    0.009 |

</details>

#### cb05__fare_and_fare_per_ticket_full_context

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_positive
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.003
  - knn: test_accuracy_mean: 0.004
    - Secondary gains:
      - test_f1_mean: 0.01
  - decision_tree: test_accuracy_mean: 0.008
    - Secondary gains:
      - test_f1_mean: 0.032
  - xgb: test_accuracy_mean: 0.008
    - Secondary gains:
      - test_f1_mean: 0.014

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                               | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:--------------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | logreg        |                          0.786 |                        0.789 |                      0.003 |                    0.713 |                  0.718 |                0.005 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | knn           |                          0.809 |                        0.813 |                      0.004 |                    0.742 |                  0.752 |                0.01  |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.758 |               -0.002 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | decision_tree |                          0.803 |                        0.811 |                      0.008 |                    0.702 |                  0.734 |                0.032 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | random_forest |                          0.822 |                        0.82  |                     -0.002 |                    0.744 |                  0.741 |               -0.003 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.722 |                0.001 |
| baseline__raw     | cb05__fare_and_fare_per_ticket_full_context | xgb           |                          0.826 |                        0.834 |                      0.008 |                    0.758 |                  0.772 |                0.014 |

##### Summary

| compare_group                               |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:--------------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb05__fare_and_fare_per_ticket_full_context |                           0.003 |                         -0.002 |                          0.008 |                0.00814286 |                   -0.003 |                    0.032 |

</details>

</details>


#### Interpretation

Retaining raw `Fare` does not generally rescue weak normalized Fare
representations. Batch-local and fitted `Fare/TicketGroupSize` remain negative
overall, suggesting that their earlier weakness cannot primarily be explained
by information lost when raw Fare was replaced.

Full-context Ticket normalization behaves differently. Decision Tree improves
from -0.001 accuracy / +0.012 F1 with normalized Fare alone to +0.008 / +0.032
when raw Fare is retained. XGBoost similarly changes from +0.001 / +0.002 to
+0.008 / +0.014.

KNN shows the opposite interaction. Full-context normalized Fare alone produces
+0.011 accuracy / +0.013 F1, while adding raw Fare reduces the gain to
+0.004 / +0.010. The normalized representation therefore appears more useful
as a replacement for raw Fare for KNN.

Decision Tree also benefits strongly from `Fare + Fare/FamilySize`
(+0.008 accuracy, +0.028 F1), suggesting that raw and meaningful normalized
Fare representations can provide complementary information to this model.

#### Interim conclusion

Raw and normalized Fare are complementary only for specific combinations of
model and group definition. Restoring raw Fare does not repair the poor
behavior of batch-local or fitted Ticket normalization, while Decision Tree
and XGBoost show substantial complementarity with specific normalized
representations.

---

#### Multiple Fare representations

Because FamilySize and TicketGroupSize appear to represent different passenger
relationships, both normalized Fare representations were supplied together to
test whether their information could be complementary.

Two configurations were examined:

- `Fare + Fare/FamilySize + Fare/TicketGroupSize`
- `Fare/FamilySize + Fare/TicketGroupSize`, excluding raw Fare

The second configuration was introduced as a follow-up after the first
produced an unexpected Decision Tree interaction.

<details>
<summary>Comparison of all Fare representations</summary>

| Model | All Fare Batch ΔAcc | ΔF1 | Fitted ΔAcc | ΔF1 | Full Context ΔAcc | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | -0.001 | -0.003 | -0.003 | -0.004 | +0.001 | +0.002 |
| KNN | -0.008 | -0.008 | -0.004 | -0.002 | +0.002 | +0.007 |
| SVC | -0.010 | -0.013 | -0.011 | -0.014 | -0.009 | -0.012 |
| Decision Tree | +0.011 | +0.039 | +0.011 | +0.040 | +0.005 | +0.028 |
| Random Forest | -0.007 | -0.007 | +0.001 | +0.003 | -0.012 | -0.015 |
| Extra Trees | -0.004 | -0.004 | +0.002 | +0.002 | -0.004 | -0.005 |
| XGBoost | -0.004 | -0.004 | -0.001 | 0.000 | -0.007 | -0.008 |

</details>

<details>
<summary>Comparison of raw + normalized Fare representations</summary>

#### cb06__all_fare_features_batch

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.011
    - Secondary gains:
      - test_f1_mean: 0.039

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                 | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb06__all_fare_features_batch | logreg        |                          0.786 |                        0.785 |                     -0.001 |                    0.713 |                  0.71  |               -0.003 |
| baseline__raw     | cb06__all_fare_features_batch | knn           |                          0.809 |                        0.801 |                     -0.008 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | cb06__all_fare_features_batch | svc           |                          0.827 |                        0.817 |                     -0.01  |                    0.76  |                  0.747 |               -0.013 |
| baseline__raw     | cb06__all_fare_features_batch | decision_tree |                          0.803 |                        0.814 |                      0.011 |                    0.702 |                  0.741 |                0.039 |
| baseline__raw     | cb06__all_fare_features_batch | random_forest |                          0.822 |                        0.815 |                     -0.007 |                    0.744 |                  0.737 |               -0.007 |
| baseline__raw     | cb06__all_fare_features_batch | extra_trees   |                          0.804 |                        0.8   |                     -0.004 |                    0.721 |                  0.717 |               -0.004 |
| baseline__raw     | cb06__all_fare_features_batch | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.754 |               -0.004 |

##### Summary

| compare_group                 |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb06__all_fare_features_batch |                     -0.00328571 |                          -0.01 |                          0.011 |                         0 |                   -0.013 |                    0.039 |

</details>

#### cb06__all_fare_features_fitted

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.011
    - Secondary gains:
      - test_f1_mean: 0.04

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                  | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb06__all_fare_features_fitted | logreg        |                          0.786 |                        0.783 |                     -0.003 |                    0.713 |                  0.709 |               -0.004 |
| baseline__raw     | cb06__all_fare_features_fitted | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.74  |               -0.002 |
| baseline__raw     | cb06__all_fare_features_fitted | svc           |                          0.827 |                        0.816 |                     -0.011 |                    0.76  |                  0.746 |               -0.014 |
| baseline__raw     | cb06__all_fare_features_fitted | decision_tree |                          0.803 |                        0.814 |                      0.011 |                    0.702 |                  0.742 |                0.04  |
| baseline__raw     | cb06__all_fare_features_fitted | random_forest |                          0.822 |                        0.823 |                      0.001 |                    0.744 |                  0.747 |                0.003 |
| baseline__raw     | cb06__all_fare_features_fitted | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.723 |                0.002 |
| baseline__raw     | cb06__all_fare_features_fitted | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.758 |                0     |

##### Summary

| compare_group                  |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb06__all_fare_features_fitted |                    -0.000714286 |                         -0.011 |                          0.011 |                0.00357143 |                   -0.014 |                     0.04 |

</details>

#### cb06__all_fare_features_full_context

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.005
    - Secondary gains:
      - test_f1_mean: 0.028

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                        | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb06__all_fare_features_full_context | logreg        |                          0.786 |                        0.787 |                      0.001 |                    0.713 |                  0.715 |                0.002 |
| baseline__raw     | cb06__all_fare_features_full_context | knn           |                          0.809 |                        0.811 |                      0.002 |                    0.742 |                  0.749 |                0.007 |
| baseline__raw     | cb06__all_fare_features_full_context | svc           |                          0.827 |                        0.818 |                     -0.009 |                    0.76  |                  0.748 |               -0.012 |
| baseline__raw     | cb06__all_fare_features_full_context | decision_tree |                          0.803 |                        0.808 |                      0.005 |                    0.702 |                  0.73  |                0.028 |
| baseline__raw     | cb06__all_fare_features_full_context | random_forest |                          0.822 |                        0.81  |                     -0.012 |                    0.744 |                  0.729 |               -0.015 |
| baseline__raw     | cb06__all_fare_features_full_context | extra_trees   |                          0.804 |                        0.8   |                     -0.004 |                    0.721 |                  0.716 |               -0.005 |
| baseline__raw     | cb06__all_fare_features_full_context | xgb           |                          0.826 |                        0.819 |                     -0.007 |                    0.758 |                  0.75  |               -0.008 |

##### Summary

| compare_group                        |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb06__all_fare_features_full_context |                     -0.00342857 |                         -0.012 |                          0.005 |              -0.000428571 |                   -0.015 |                    0.028 |

</details>

</details>

Batch and fitted Ticket representations produce an unexpected result for
Decision Tree. Despite performing poorly in simpler Fare configurations, both
produce the strongest Decision Tree result once raw Fare and Family-normalized
Fare are simultaneously available:

- batch: +0.011 accuracy / +0.039 F1
- fitted: +0.011 accuracy / +0.040 F1
- full context: +0.005 accuracy / +0.028 F1

The fitted result reproducing the batch result makes it less likely that the
batch result is merely an isolated configuration anomaly. It instead suggests
that Decision Tree is exploiting an interaction between the available Fare
representations.

Importantly, the full-context representation no longer performs best. A
representation that is more useful in isolation is therefore not necessarily
the representation that contributes the most additional information when
other related features are already available.

#### Follow-up: removing raw Fare

To determine whether the Decision Tree improvement came from combining the two
normalized representations or depended on raw Fare, both normalized Fare
features were tested without raw `Fare`.

<details>
<summary>Normalized Fare combinations without raw Fare</summary>

| Model | Batch ΔAcc | ΔF1 | Fitted ΔAcc | ΔF1 | Full Context ΔAcc | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | +0.002 | +0.003 | +0.002 | +0.003 | 0.000 | +0.001 |
| KNN | -0.003 | -0.002 | -0.002 | +0.001 | +0.001 | +0.005 |
| SVC | -0.007 | -0.008 | -0.007 | -0.008 | -0.003 | -0.004 |
| Decision Tree | +0.003 | +0.012 | +0.003 | +0.013 | +0.001 | +0.004 |
| Random Forest | -0.014 | -0.019 | -0.005 | -0.008 | -0.003 | -0.006 |
| Extra Trees | -0.002 | -0.003 | +0.003 | +0.003 | +0.004 | 0.000 |
| XGBoost | -0.011 | -0.014 | -0.004 | -0.005 | -0.003 | -0.003 |

</details>

<details>
<summary>Comparison of raw + normalized Fare representations</summary>

#### cb09__fare_per_family_and_ticket_batch

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.003
    - Secondary gains:
      - test_f1_mean: 0.012


</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                          | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | knn           |                          0.809 |                        0.806 |                     -0.003 |                    0.742 |                  0.74  |               -0.002 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | svc           |                          0.827 |                        0.82  |                     -0.007 |                    0.76  |                  0.752 |               -0.008 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.714 |                0.012 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | random_forest |                          0.822 |                        0.808 |                     -0.014 |                    0.744 |                  0.725 |               -0.019 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | extra_trees   |                          0.804 |                        0.802 |                     -0.002 |                    0.721 |                  0.718 |               -0.003 |
| baseline__raw     | cb09__fare_per_family_and_ticket_batch | xgb           |                          0.826 |                        0.815 |                     -0.011 |                    0.758 |                  0.744 |               -0.014 |

##### Summary

| compare_group                          |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb09__fare_per_family_and_ticket_batch |                     -0.00457143 |                         -0.014 |                          0.003 |               -0.00442857 |                   -0.019 |                    0.012 |

</details>

#### cb09__fare_per_family_and_ticket_fitted

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.003
    - Secondary gains:
      - test_f1_mean: 0.013
  - extra_trees: test_accuracy_mean: 0.003

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | logreg        |                          0.786 |                        0.788 |                      0.002 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | knn           |                          0.809 |                        0.807 |                     -0.002 |                    0.742 |                  0.743 |                0.001 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | svc           |                          0.827 |                        0.82  |                     -0.007 |                    0.76  |                  0.752 |               -0.008 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.715 |                0.013 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | random_forest |                          0.822 |                        0.817 |                     -0.005 |                    0.744 |                  0.736 |               -0.008 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | extra_trees   |                          0.804 |                        0.807 |                      0.003 |                    0.721 |                  0.724 |                0.003 |
| baseline__raw     | cb09__fare_per_family_and_ticket_fitted | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.753 |               -0.005 |

##### Summary

| compare_group                           |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb09__fare_per_family_and_ticket_fitted |                     -0.00142857 |                         -0.007 |                          0.003 |              -0.000142857 |                   -0.008 |                    0.013 |

</details>

#### cb09__fare_per_family_and_ticket_full_context

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - extra_trees: test_accuracy_mean: 0.004

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                                 | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | logreg        |                          0.786 |                        0.786 |                      0     |                    0.713 |                  0.714 |                0.001 |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | knn           |                          0.809 |                        0.81  |                      0.001 |                    0.742 |                  0.747 |                0.005 |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | svc           |                          0.827 |                        0.824 |                     -0.003 |                    0.76  |                  0.756 |               -0.004 |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | decision_tree |                          0.803 |                        0.804 |                      0.001 |                    0.702 |                  0.706 |                0.004 |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | random_forest |                          0.822 |                        0.819 |                     -0.003 |                    0.744 |                  0.738 |               -0.006 |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | extra_trees   |                          0.804 |                        0.808 |                      0.004 |                    0.721 |                  0.721 |                0     |
| baseline__raw     | cb09__fare_per_family_and_ticket_full_context | xgb           |                          0.826 |                        0.823 |                     -0.003 |                    0.758 |                  0.755 |               -0.003 |

##### Summary

| compare_group                                 |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb09__fare_per_family_and_ticket_full_context |                    -0.000428571 |                         -0.003 |                          0.004 |              -0.000428571 |                   -0.006 |                    0.005 |

</details>

</details>

Removing raw Fare substantially reduces the Decision Tree gains. The two
normalized representations alone therefore do not reproduce the effect seen
when all three representations are available.

For Decision Tree, `Fare + Fare/FamilySize` appears to form an important base
combination. Adding batch or fitted Ticket-normalized Fare provides a further
+0.003 accuracy and approximately +0.011–0.012 F1 relative to that
configuration, while adding the full-context representation provides no F1
improvement and reduces accuracy by 0.003.

This suggests that the value of a Fare representation is conditional on the
other representations available to the model. A feature that performs poorly
alone may still provide complementary partitioning information when combined
with related features.

#### Similarity diagnostic

A preliminary feature-similarity diagnostic supports the hypothesis that
full-context Ticket normalization overlaps more strongly with Family
normalization.

| Ticket strategy | Group-size equality | Group-size MAE | Fare Pearson | Fare Spearman | Exact Fare equality |
|---|---:|---:|---:|---:|---:|
| Batch | 0.587 | 0.922 | 0.777 | 0.586 | 0.587 |
| Fitted | 0.542 | 0.844 | 0.845 | 0.672 | 0.542 |
| Full context | 0.827 | 0.307 | 0.840 | 0.866 | 0.827 |

Under full prediction context, `FamilySize` and `TicketGroupSize` are exactly
equal for approximately 82.7% of validation passengers, compared with 58.7%
under batch-local construction and 54.2% under fitted construction. The mean
absolute difference between the group-size representations also falls to
0.307 under full context.

Because both normalized features use the same Fare numerator, this results in
`Fare/FamilySize` and `Fare/TicketGroupSize` being exactly equal for the same
82.7% of passengers under full context. Their rank similarity also increases
substantially, with Spearman correlation rising from 0.586 for batch and 0.672
for fitted to 0.866 under full context.

Pearson correlation does not increase monotonically: fitted (0.845) is
slightly higher than full context (0.840). The evidence therefore does not
support a general claim that full-context normalization is more correlated
under every measure. Instead, it supports a more specific form of structural
overlap: full-context Ticket normalization produces substantially more exact
agreement with Family normalization and a more similar ordering of passenger
Fare values.

This provides a plausible explanation for the Decision Tree interaction seen
in CB06. Once `Fare/FamilySize` is available, full-context
`Fare/TicketGroupSize` may provide fewer distinct candidate partitions than
the batch or fitted representations. The latter are less complete estimates
of the passenger's full ticket group, but may consequently provide more
complementary threshold structure to the constrained tree.

This diagnostic supports the redundancy/complementarity hypothesis but does
not establish that it caused the observed performance difference. Confirming
the mechanism would require examining feature usage and split behavior across
the fitted trees.

#### Findings

- Fare normalization is strongly dependent on both the definition of passenger
  grouping and the model using the resulting representation. Neither
  `Fare/FamilySize` nor `Fare/TicketGroupSize` is a generally superior
  replacement for raw `Fare`.

- The population semantics of `TicketGroupSize` become more consequential when
  the feature is used as the denominator of another feature. Batch-local
  `Fare/TicketGroupSize` was broadly detrimental, fitted normalization reduced
  much of this negative effect, and full-context normalization was the only
  Ticket-normalized Fare representation with positive mean changes in both
  accuracy and F1 when used as a replacement for raw Fare.

- KNN showed the clearest standalone benefit from full-context Ticket
  normalization. Replacing raw Fare with full-context `Fare/TicketGroupSize`
  improved accuracy by +0.011 and F1 by +0.013, while the batch and fitted
  versions were detrimental. This indicates that the usefulness of a derived
  feature can depend not only on its formula, but also on the population used
  to construct one of its inputs.

- Restoring raw `Fare` did not generally rescue the weak batch or fitted
  Ticket-normalized representations. Their poor standalone behavior therefore
  cannot be explained simply by information lost when raw Fare was replaced.

- Raw and normalized Fare can nevertheless be complementary for specific
  models. Decision Tree benefited from `Fare + Fare/FamilySize`
  (+0.008 accuracy, +0.028 F1), while full-context
  `Fare/TicketGroupSize` became substantially more useful to Decision Tree and
  XGBoost when raw Fare was retained.

- Combining all three Fare representations exposed a different interaction for
  Decision Tree. `Fare + Fare/FamilySize + Fare/TicketGroupSize` produced
  +0.011 accuracy / +0.039 F1 with batch Ticket counts and
  +0.011 / +0.040 with fitted counts, compared with +0.005 / +0.028 under
  full prediction context.

- CB09 showed that this Decision Tree effect does not come simply from combining
  the two normalized Fare representations. Removing raw Fare reduced the
  corresponding gains to +0.003 / +0.012 for batch, +0.003 / +0.013 for
  fitted, and +0.001 / +0.004 for full context. Raw Fare is therefore an
  important part of the interaction observed in CB06.

- The feature-similarity diagnostic found substantial structural overlap between
  Family and full-context Ticket representations. `FamilySize` and
  full-context `TicketGroupSize` were exactly equal for 82.7% of validation
  passengers, compared with 58.7% for batch and 54.2% for fitted Ticket counts.
  Their mean absolute group-size difference also fell from 0.922 (batch) and
  0.844 (fitted) to 0.307 under full context.

- Because both normalized Fare features share the same Fare numerator, this
  denominator agreement resulted in `Fare/FamilySize` and
  `Fare/TicketGroupSize` being exactly equal for the same 82.7% of passengers
  under full context. Their Spearman correlation also increased substantially:
  0.586 for batch, 0.672 for fitted, and 0.866 for full context.

- The similarity increase is not universal across every measure. Pearson
  correlation was 0.777 for batch, 0.845 for fitted, and 0.840 for full
  context. The diagnostic therefore supports increased exact and rank-based
  overlap under full context rather than a general claim that full-context
  normalization is more correlated according to every similarity measure.

- FamilySize and TicketGroupSize are therefore neither interchangeable nor
  completely independent representations. With incomplete Ticket information
  they frequently describe different group structures, while complete Ticket
  context causes the two representations to coincide for a large majority of
  passengers.

- More complete feature semantics do not necessarily imply greater marginal
  predictive value. Full-context Ticket normalization was generally the
  strongest standalone Ticket-based Fare representation, yet it contributed
  less than batch or fitted normalization to Decision Tree once raw Fare and
  Family-normalized Fare were already available.

#### Hypotheses

- The poor behavior of batch-local `Fare/TicketGroupSize` when used alone is
  likely related to instability in its denominator. A passenger's normalized
  Fare can change according to which same-ticket passengers happen to be
  present in the prediction batch, altering both the scale and meaning of the
  feature.

- Full prediction context produces a more complete estimate of the passenger's
  ticket group and therefore a more stable per-ticket Fare representation.
  This may explain why full-context normalization performs substantially better
  than batch or fitted normalization when the Ticket-normalized feature is used
  alone, particularly for KNN.

- KNN may benefit from full-context Fare normalization because replacing raw
  Fare with a group-relative value changes the geometry of the feature space.
  The improvement disappearing or shrinking when raw Fare is restored is
  consistent with this possibility, but the current experiments do not
  directly establish the mechanism.

- Decision Tree appears to benefit from having access to both absolute and
  relative Fare information. Raw `Fare` preserves absolute price, while
  `Fare/FamilySize` and `Fare/TicketGroupSize` provide alternative
  group-relative representations. The strong reduction observed in CB09
  suggests that raw Fare acts as an important component of this combination
  rather than the two normalized representations being sufficient by
  themselves.

- The similarity diagnostic provides a plausible explanation for why batch and
  fitted Ticket normalization add more value than full-context normalization in
  the three-representation Decision Tree configuration. Under full context,
  `Fare/TicketGroupSize` is identical to `Fare/FamilySize` for 82.7% of
  validation passengers and has substantially greater rank similarity.
  Consequently, it may offer the constrained tree fewer distinct useful
  thresholds once `Fare/FamilySize` is already available.

- Batch and fitted Ticket normalization are less complete representations of
  the passenger's full ticket group, but their greater difference from
  `Fare/FamilySize` may create additional candidate partitions for the
  Decision Tree. In this specific feature combination, this additional
  complementarity may be more useful than the greater semantic completeness of
  the full-context representation.

- This explanation is supported by the measured feature overlap and the
  controlled CB09 comparison, but it is not a demonstrated causal mechanism.
  Confirming it would require examining feature usage, thresholds, and
  fold-level tree structures. Such analysis is not currently necessary for
  feature selection unless this interaction becomes important to the final
  model configuration.

- More broadly, these experiments suggest that feature quality should not be
  treated as an intrinsic property of a representation. Its usefulness depends
  on the model's inductive structure, the other representations already
  available, and—in the case of population-dependent features—the information
  context available at prediction time.

#### Current recommendation

Fare representation should remain model-specific rather than adopting a single
engineered form across all models.

- **Logistic Regression:** retain raw Fare as the default. Normalized
  representations produced only small improvements that do not currently
  justify additional complexity or context requirements.

- **KNN:** carry full-context `Fare/TicketGroupSize` forward as a
  context-dependent candidate, preferably replacing rather than supplementing
  raw Fare. If full prediction context is unavailable, retain raw Fare.

- **SVC:** retain raw Fare. Fare normalization and representation stacking did
  not provide a compelling improvement.

- **Decision Tree:** carry
  `Fare + Fare/FamilySize + fitted Fare/TicketGroupSize` forward as the primary
  candidate. It produced the strongest Fare-domain result
  (+0.011 accuracy, +0.040 F1) while retaining inductive prediction semantics.
  `Fare + Fare/FamilySize` remains a useful simpler alternative if the extra
  representation does not survive final feature combination.

- **Random Forest:** retain raw Fare for now. Engineered Fare representations
  provided little consistent benefit. Whether Fare itself should ultimately be
  retained remains dependent on the final Age configuration.

- **Extra Trees:** retain raw Fare for now. Small gains from some normalized
  combinations are not yet compelling enough to justify their added complexity
  or context assumptions.

- **XGBoost:** carry
  `Fare + full-context Fare/TicketGroupSize` forward as a context-dependent
  candidate (+0.008 accuracy, +0.014 F1). Retain raw Fare when full prediction
  context is unavailable.

These are candidates for final feature-combination testing rather than final
model configurations. In particular, full-context representations should only
be retained when the intended prediction setting makes the required population
context available.

---

### Feature Combination

Two individually informative features can sometimes be combined into a new
categorical representation, allowing their joint values to be treated as a
single feature.

This investigation was designed as a small proof of concept: if two important
features are available, can combining them into a single representation improve
model performance, even when there is no specific domain reason requiring the
combination?

`Sex` and `Pclass` were selected because they showed the strongest relationships
with the target among the raw features. This made them useful candidates for
producing a clear result rather than choosing weaker features whose effect might
be too small to interpret.

Two representations were tested:

- **FE12:** replace `Sex` and `Pclass` with `Sex_Pclass`.
- **CB08:** retain `Sex` and `Pclass` and add `Sex_Pclass`.

CB08 was originally introduced as a follow-up after the earlier FE12 results
suggested that replacing the source features caused substantial F1 losses. It
tests whether the combined representation becomes more useful when the models
retain access to its original components.

#### Experiments:

#### fe12__sex_pclass

<details>
<summary>Interpretation</summary>

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.016
  - extra_trees: test_accuracy_mean: 0.006
    - Secondary losses:
      - test_f1_mean: -0.019

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group    | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:-----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe12__sex_pclass | logreg        |                          0.786 |                        0.802 |                      0.016 |                    0.713 |                  0.713 |                0     |
| baseline__raw     | fe12__sex_pclass | knn           |                          0.809 |                        0.806 |                     -0.003 |                    0.742 |                  0.735 |               -0.007 |
| baseline__raw     | fe12__sex_pclass | svc           |                          0.827 |                        0.828 |                      0.001 |                    0.76  |                  0.747 |               -0.013 |
| baseline__raw     | fe12__sex_pclass | decision_tree |                          0.803 |                        0.798 |                     -0.005 |                    0.702 |                  0.692 |               -0.01  |
| baseline__raw     | fe12__sex_pclass | random_forest |                          0.822 |                        0.814 |                     -0.008 |                    0.744 |                  0.71  |               -0.034 |
| baseline__raw     | fe12__sex_pclass | extra_trees   |                          0.804 |                        0.81  |                      0.006 |                    0.721 |                  0.702 |               -0.019 |
| baseline__raw     | fe12__sex_pclass | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.751 |               -0.007 |

##### Summary

| compare_group    |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe12__sex_pclass |                     0.000857143 |                         -0.008 |                          0.016 |                -0.0128571 |                   -0.034 |                        0 |

</details>

#### cb08__sex_pclass_features

<details>
<summary>Interpretation</summary>

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.016
  - extra_trees: test_accuracy_mean: 0.003
    - Secondary losses:
      - test_f1_mean: -0.021


</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group             | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:--------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb08__sex_pclass_features | logreg        |                          0.786 |                        0.802 |                      0.016 |                    0.713 |                  0.713 |                0     |
| baseline__raw     | cb08__sex_pclass_features | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.738 |               -0.004 |
| baseline__raw     | cb08__sex_pclass_features | svc           |                          0.827 |                        0.827 |                      0     |                    0.76  |                  0.75  |               -0.01  |
| baseline__raw     | cb08__sex_pclass_features | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.702 |                0     |
| baseline__raw     | cb08__sex_pclass_features | random_forest |                          0.822 |                        0.815 |                     -0.007 |                    0.744 |                  0.729 |               -0.015 |
| baseline__raw     | cb08__sex_pclass_features | extra_trees   |                          0.804 |                        0.807 |                      0.003 |                    0.721 |                  0.7   |               -0.021 |
| baseline__raw     | cb08__sex_pclass_features | xgb           |                          0.826 |                        0.819 |                     -0.007 |                    0.758 |                  0.746 |               -0.012 |

##### Summary

| compare_group             |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:--------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb08__sex_pclass_features |                     0.000714286 |                         -0.007 |                          0.016 |               -0.00885714 |                   -0.021 |                        0 |

</details>

</details>

#### Findings

- Combining two informative features can improve predictive performance, but
  the effect is strongly model-dependent.

- Logistic Regression provides the clearest positive result. Replacing `Sex`
  and `Pclass` with `Sex_Pclass` improved accuracy by +0.016 without changing
  F1. Keeping all three representations produced the same reported
  +0.016 accuracy and unchanged F1.

- The Logistic Regression result demonstrates that a combined representation
  can expose useful predictive structure even when it is constructed entirely
  from information already available to the model. In this case, retaining the
  source features provided no additional measurable benefit at the reported
  precision.

- Decision Tree showed the opposite behavior. Replacing `Sex` and `Pclass`
  reduced accuracy by 0.005 and F1 by 0.010, while adding `Sex_Pclass`
  alongside them produced no measurable change from baseline. For this model,
  preserving the original representations was more useful than replacing them
  with the combined feature.

- KNN and SVC showed no meaningful accuracy improvement and experienced small
  F1 losses. Random Forest was negatively affected under both configurations.

- Extra Trees gained some accuracy from the combined representation, but the
  gain was accompanied by an approximately 0.02 F1 loss and therefore does not
  provide a compelling trade-off.

- XGBoost also showed no useful improvement from either representation.

- The experiment therefore answers the original proof-of-concept question
  positively, but conditionally: combining important features is a potentially
  useful feature-engineering technique, not a transformation that should be
  expected to improve every model.

#### Hypotheses

- A combined categorical feature changes how the relationship between its
  source variables is presented to the model. Rather than requiring the model
  to derive useful combinations from separate inputs, relevant combinations
  become directly represented as feature values.

- Models differ in how much they benefit from this explicit representation.
  Logistic Regression may benefit because a combined categorical feature makes
  joint structure directly available to a linear model that would otherwise
  represent the source features primarily through separate effects.

- Models capable of naturally learning conditional relationships, particularly
  tree-based models, may gain less from an explicit combination because similar
  relationships can already be constructed through their own decision
  structure.

- Replacing the source variables can also remove useful flexibility. The
  Decision Tree result illustrates this distinction: adding the combined
  representation caused no measurable change, while forcing the combined
  representation to replace its source features reduced performance.

- The usefulness of feature combination should therefore be treated as an
  empirical question. Strong individual features can make reasonable
  candidates for combination, but their importance alone does not guarantee
  that their combined representation will be useful.

#### Current recommendation

Feature combination is worth retaining as a feature-engineering technique to
consider when important predictors are identified, particularly when an
explicit joint representation may expose structure that is difficult for the
chosen model to learn directly.

It should be tested rather than applied automatically, and replacement of the
source features should be evaluated separately from adding the combined
representation.

For the current Titanic models, carry `Sex_Pclass` forward only for Logistic
Regression. FE12 and CB08 produce the same reported performance, so the simpler
replacement representation is the current candidate. Retain the original
`Sex` and `Pclass` representation for the remaining models.

No further Sex/Pclass combination experiments are currently warranted. The
purpose of this investigation was to determine whether feature combination can
be useful; the experiments provide sufficient evidence that it can be, while
also demonstrating its model dependence and potential costs.

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

| Domain                     | LogReg                                                 | KNN                                                              | SVC                                                                  | Decision Tree                                                        | Random Forest                                       | Extra Trees                                  | XGBoost                                               |
| -------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------- |
| **Title**                  | **Carry:** Title                                       | **Carry:** Title                                                 | **Carry:** Title                                                     | **Carry:** Title                                                     | **Carry:** Title                                    | **Carry:** Title                             | **Carry:** Title                                      |
| **Age**                    | **Carry:** CB03 — Title+Pclass imputed Age + AgeBin    | **Raw:** Age                                                     | **Carry:** CB02 — Title-imputed Age + AgeBin                         | **Carry:** FE11 — AgeBin replacing Age                               | **Carry:** CB03 — Title+Pclass imputed Age + AgeBin | **Carry:** CB02 — Title-imputed Age + AgeBin | **Alt.:** FE06 — Title-imputed Age; otherwise raw Age |
| **Family**                 | **Carry:** CB07 — SibSp + Parch + FamilySize + IsAlone | **Raw:** SibSp + Parch                                           | **Raw:** SibSp + Parch                                               | **Carry:** CB07 — raw + FamilySize + IsAlone                         | **Raw:** SibSp + Parch                              | **Raw:** SibSp + Parch                       | **Alt.:** CB07 — raw + FamilySize + IsAlone           |
| **Cabin**                  | **Carry:** FE04 — HasCabin + Deck                      | **Carry:** FE03 — Deck                                           | **Raw:** no Cabin feature                                            | **Raw:** no Cabin feature                                            | **Raw:** no Cabin feature                           | **Alt.:** FE02 — HasCabin                    | **Carry:** FE04 — HasCabin + Deck                     |
| **Ticket**                 | **Alt.:** fitted/batch TicketGroupSize                 | **Raw**; full-context only as weak Alt.                          | **Carry:** fitted TicketGroupSize; **Context:** full-context version | **Carry:** fitted TicketGroupSize; **Context:** full-context version | **Raw**                                             | **Raw**; full-context weak Alt.              | **Raw**                                               |
| **Fare**                   | **Raw:** Fare                                          | **Context:** replace Fare with full-context Fare/TicketGroupSize | **Raw:** Fare                                                        | **Carry:** Fare + Fare/FamilySize + fitted Fare/TicketGroupSize      | **Raw:** Fare, with **Fare removal to test**        | **Raw:** Fare                                | **Context:** Fare + full-context Fare/TicketGroupSize |
| **Sex/Pclass combination** | **Carry:** FE12 — Sex_Pclass replacing Sex + Pclass    | **Raw:** Sex + Pclass                                            | **Raw:** Sex + Pclass                                                | **Raw:** Sex + Pclass                                                | **Raw:** Sex + Pclass                               | **Raw:** Sex + Pclass                        | **Raw:** Sex + Pclass                                 |


## Feature Selection

### Strategy

Feature selection was performed independently for each model, using the feature-engineering experiments to identify promising representations.

Rather than exhaustively testing every feature combination, the process used two complementary stages:

**Forward Selection**

1. Start from a promising baseline configuration, generally the raw features with Title engineering.
2. Incrementally test feature representations supported by earlier experiments.
3. Compare each candidate against the current accepted configuration, prioritizing **cross-validation accuracy**, with **F1 as a secondary metric**.
4. Carry promising or approximately neutral representations forward when further interactions are worth investigating.
5. Reject clearly detrimental additions, while testing alternative representations when justified by previous evidence.

**Backward Pruning**

1. Start from the completed forward-selection candidate.
2. Remove individual patches or feature groups to measure their contribution within the assembled configuration.
3. Retain removals that improve or preserve performance; restore features whose removal causes meaningful deterioration.
4. Investigate specific feature interactions or alternative representations when the results justify additional experiments.
5. Select a final configuration based on predictive performance, representation complexity, and inference requirements.

**Evaluation and interpretation**

- Reported results are cross-validation means. Differences in accuracy and F1 are expressed as absolute changes, not relative percentages.
- Small improvements are treated cautiously; they do not necessarily establish a reliable generalization advantage.
- Feature contributions are **conditional on the surrounding representation**. A feature that performs poorly in isolation may become useful when combined with others, and vice versa.
- Fitted, batch, and full-context TicketGroupSize approaches are distinguished where relevant. A patch may use TicketGroupSize internally without exposing it as a separate model input.
- Cross-validation results used to guide feature selection are not independent final performance estimates. Final generalization should be assessed using data not involved in selection.

The objective was to identify **compact, interpretable feature configurations supported by experimental evidence**, rather than simply maximizing the number of engineered features or exhaustively searching all possible combinations.


### Logistic Regression

#### Starting Point

Feature selection started from the raw feature set with **FE05 Title engineering**, which had produced a substantial improvement during the earlier feature-engineering experiments.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.786 | 0.713 |
| **FS baseline — Title FE05** | **0.825** | **0.765** |

The objective was to determine which additional engineered representations complemented Title, then remove redundant or detrimental features through Backward Pruning.

#### Forward Selection

Previously promising feature representations were introduced incrementally. Small or neutral gains could be retained provisionally to investigate potential interactions with later features.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | Title FE05 | 0.825 | 0.765 | — | — | Reference |
| FS01 | + Age CB03 | 0.828 | 0.767 | +0.003 | +0.002 | Carry; CB02 alternative |
| FS02 | + Family FE01 | 0.829 | 0.769 | +0.001 | +0.002 | Carry provisionally |
| FS03 | + Cabin FE04 | 0.834 | 0.777 | +0.005 | +0.008 | Carry |
| FS04 | + TicketGroupSize FE09 fitted | 0.835 | 0.778 | +0.001 | +0.001 | Carry provisionally |
| FS05 | + Fare CB05 fitted | 0.835 | 0.778 | 0.000 | 0.000 | Carry for pruning |
| FS06 | + Sex × Pclass FE12 | 0.833 | 0.770 | -0.002 | -0.008 | Reject |

*Deltas compare each test with the preceding accepted configuration. Additional tests within the same stage are discussed below.*

**Notable decisions**

- **Age:** CB03 (Title + Pclass-imputed continuous Age and Age bins) improved accuracy by 0.003 and F1 by 0.002. CB02 achieved 0.827 / 0.769, producing a small accuracy–F1 trade-off. CB03 was carried forward because accuracy was the primary metric, while CB02 was preserved as an alternative candidate for later investigation.
- **Family:** CB07 produced no accuracy gain and +0.002 F1. FE01, which had also performed well during feature engineering, improved accuracy by 0.001 and F1 by 0.002. FE01 was provisionally selected for further testing.
- **Cabin:** FE04 improved accuracy by 0.005 and F1 by 0.008, the largest marginal improvement after the starting baseline. Given this comparatively strong result and FE04's earlier performance, additional Cabin variants were deprioritized in favor of exploring the remaining domains.
- **Ticket:** FE09 batch and fitted both improved accuracy and F1 by 0.001; full-context preserved accuracy and improved F1 by 0.001. Fitted semantics were preferred because they matched the best observed performance without requiring full prediction context.
- **Fare:** FE08 reduced accuracy/F1 by 0.003/0.004, while CB05 full-context reduced both by 0.002. CB05 fitted was neutral and was retained only to investigate its contribution during pruning.
- **Sex × Pclass:** Both FE12 and CB08 reduced performance. FE12 produced 0.833 / 0.770, while CB08 had slightly larger losses. Both were rejected.

**Forward candidate:** Title FE05 + Age CB03 + Family FE01 + Cabin FE04 + Ticket FE09 fitted + Fare CB05 fitted.

**Result:** 0.835 accuracy / 0.778 F1.

#### Backward Pruning

Backward Pruning tested whether the provisionally retained representations remained useful in the completed forward candidate.

Accepted removals were carried into subsequent experiments. Rejected removals were evaluated against the retained configuration.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Forward candidate | — | 0.835 | 0.778 | — | — | Reference |
| PR01 | Remove engineered Family | 0.835 | 0.779 | 0.000 | +0.001 | Remove |
| PR02 | Remove FE09 TicketGroupSize | 0.837 | 0.782 | +0.002 | +0.003 | Remove |
| PR03 | Remove Fare domain | **0.841** | **0.787** | +0.004 | +0.005 | Remove |
| PR04 | Remove CB03 Age engineering | 0.824 | 0.768 | -0.017 | -0.019 | Keep CB03 |
| PR05 | Remove Cabin FE04 | 0.832 | 0.772 | -0.009 | -0.015 | Keep Cabin |
| PR06 | Remove Title FE05 | 0.804 | 0.734 | -0.037 | -0.053 | Keep Title |
| PR07 | Remove raw SibSp + Parch | 0.800 | 0.737 | -0.041 | -0.050 | Keep raw Family |
| PR08 | Remove all Age representations | 0.820 | 0.762 | -0.021 | -0.025 | Keep Age |

*Deltas compare each experiment with its reference configuration: the latest accepted candidate, not necessarily the preceding table row. Additional PR08 tests are discussed below.*

**Key pruning findings**

**Family and TicketGroupSize**

Removing FE01's engineered Family representation preserved accuracy and slightly improved F1. Removing FE09 TicketGroupSize afterward improved accuracy by another 0.002 and F1 by 0.003.

However, removing raw SibSp and Parch from the later pruned configuration caused substantial losses (−0.041 accuracy / −0.050 F1).

Thus, FamilySize and IsAlone were unnecessary in the final representation, while the original family variables remained valuable.

**Fare — conditional representation effects**

Fare produced one of the most informative pruning results:

| Raw Fare | Fare / TicketGroupSize | Accuracy | F1 |
|---|---|---:|---:|
| Yes | Yes | 0.837 | 0.782 |
| Yes | No | 0.832 | 0.775 |
| No | Yes | 0.836 | 0.783 |
| **No** | **No** | **0.841** | **0.787** |

Removing only Fare/TicketGroupSize reduced performance, while removing raw Fare alone produced a small accuracy–F1 trade-off. Removing both achieved the best result.

The engineered Fare representation therefore had conditional value, but the complete Fare domain was detrimental to the selected configuration.

**Age**

CB03 became considerably more important after pruning. Removing it reduced accuracy by 0.017 and F1 by 0.019, compared with its initial forward-selection gains of only +0.003/+0.002.

Because CB02 had been preserved as a competitive alternative during FS01, it was reconsidered in the reduced configuration. Replacing CB03 with CB02 achieved 0.832 accuracy / 0.777 F1, supporting CB03 as the stronger representation in the final context.

PR08 then investigated whether the model needed all its remaining Age information:

- **Remove all Age representations:** 0.820 accuracy / 0.762 F1 (−0.021 / −0.025).
- **Remove continuous Age while retaining Age bins:** 0.818 accuracy / 0.755 F1 (−0.023 / −0.032).

Both configurations performed worse than the retained candidate, supporting CB03's combination of continuous and binned Age.

**Cabin and Title**

Removing Cabin FE04 reduced accuracy by 0.009 and F1 by 0.015.

Removing Title FE05 caused the largest engineered-feature pruning loss: −0.037 accuracy and −0.053 F1.

Both representations were retained.

#### Final Selection

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.786 | 0.713 |
| FS baseline — Title FE05 | 0.825 | 0.765 |
| Forward candidate | 0.835 | 0.778 |
| **Final pruned candidate** | **0.841** | **0.787** |

**Selected representations**

- **FE05 — Title**
- **CB03 — Age:** Title + Pclass-based imputation, continuous Age, and Age bins
- **FE04 — Cabin**
- **Raw SibSp and Parch**, alongside remaining raw predictors not explicitly removed

**Removed or rejected representations:** FE01 FamilySize/IsAlone, FE09 TicketGroupSize, raw Fare, engineered Fare/TicketGroupSize, and Sex × Pclass.

**Improvement over raw baseline:** +0.055 accuracy / +0.074 F1.

The final configuration also improved upon the forward candidate by +0.006 accuracy and +0.009 F1 while using fewer representations.

#### Interpretation

Logistic Regression demonstrated that feature contributions can change substantially depending on the surrounding feature set.

Three findings were particularly informative:

1. **Marginal contributions changed after pruning.** CB03 Age initially provided only a small gain, but became substantially more important in the reduced configuration.
2. **Original and engineered representations were not interchangeable.** FamilySize/IsAlone could be removed, while retaining SibSp and Parch was important. Keeping the original variables may allow independent linear coefficients that the compressed FamilySize representation cannot express in the same way.
3. **Redundant or competing representations could reduce performance.** The Fare experiments showed that individually useful representations did not necessarily improve the model when retained together.

These experiments establish the observed predictive differences, but do not prove the underlying mechanisms.

Overall, Logistic Regression benefited from both combining complementary representations and removing unnecessary ones. Backward Pruning was particularly valuable: it improved performance beyond the forward candidate while producing a smaller final feature set.

### Support Vector Classifier (SVC)

#### Starting Point

Feature selection started from the raw feature set with **FE05 Title engineering**, which had produced the strongest SVC result during the earlier feature-engineering experiments.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.827 | 0.760 |
| **FS baseline — Title FE05** | **0.834** | **0.771** |

Unlike Logistic Regression, relatively few engineered representations had shown clear benefits for SVC. The selection process therefore prioritized the most promising candidates while exploring alternative representations when earlier results justified doing so.

#### Forward Selection

Feature representations were tested incrementally against the latest accepted configuration. Small improvements could be carried provisionally, while detrimental additions were rejected.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | Title FE05 | 0.834 | 0.771 | — | — | Reference |
| FS01 | + Age CB02 | 0.835 | 0.773 | +0.001 | +0.002 | Carry; CB03 alternative |
| FS02 | + TicketGroupSize FE09 full-context | 0.834 | 0.773 | -0.001 | 0.000 | Reject |
| FS03 | + Family CB07 | 0.833 | 0.770 | -0.002 | -0.003 | Reject |
| FS04 | + Fare FE10 full-context | **0.836** | **0.774** | +0.001 | +0.001 | Carry |
| FS05 | + Cabin FE04 | 0.834 | 0.772 | -0.002 | -0.002 | Reject |
| FS06 | + Sex × Pclass CB08 | 0.832 | 0.765 | -0.004 | -0.009 | Reject |

*Deltas compare each test against the latest accepted configuration. Additional experiments within each stage are discussed below.*

**Notable decisions**

- **Age:** CB02 and CB03 both achieved 0.835 accuracy / 0.773 F1, improving the Title baseline by +0.001/+0.002. CB02 was carried forward because it offered a simpler representation, while CB03 remained a competitive alternative with equivalent measured performance.

- **Ticket:** FE09 fitted reduced performance to 0.829 / 0.768, while FE09 full-context reached 0.834 / 0.773. Although TicketGroupSize had performed well in earlier isolated experiments, neither approach improved the assembled Title + Age configuration. FE09 was rejected as an additional model input.

- **Family:** CB07 had previously shown a small positive result and was tested to investigate whether Family engineering became useful alongside Title and Age. It reduced accuracy by 0.002 and F1 by 0.003, so it was rejected.

- **Fare:** FE10 fitted reduced accuracy and F1 by 0.003, while FE08 Fare/FamilySize reduced them by 0.001 and 0.002. FE10 full-context was the only tested Fare representation to improve both metrics (+0.001/+0.001). It was carried forward provisionally, despite the small gain, to determine whether the improvement survived pruning.

- **Cabin:** Because earlier Cabin representations had produced similarly weak results, FE04 was selected as a representative test of the broader Cabin information. Its addition reduced accuracy and F1 by 0.002, so further Cabin exploration was deprioritized.

- **Sex × Pclass:** CB08 reduced accuracy by 0.004 and F1 by 0.009. FE12 performed worse (−0.005/−0.014). Neither representation justified retention.

**Forward candidate:** Title FE05 + Age CB02 + Fare FE10 full-context, alongside the remaining raw predictors.

**Result:** 0.836 accuracy / 0.774 F1.

#### Backward Pruning

Backward Pruning evaluated whether the retained engineered representations remained useful in the completed forward candidate.

Because most additions had already been rejected during Forward Selection, pruning focused on testing whether the selected Age, Fare, and Title representations could be simplified or removed.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Forward candidate | — | **0.836** | **0.774** | — | — | Reference |
| PR01 | Remove entire Fare domain | 0.834 | 0.770 | -0.002 | -0.004 | Keep Fare |
| PR02 | Remove continuous/imputed Age, retain bins | 0.833 | 0.769 | -0.003 | -0.005 | Keep complete CB02 |
| PR03 | Remove Title FE05 | 0.831 | 0.767 | -0.005 | -0.007 | Keep Title |

*Deltas compare each experiment against the complete forward candidate. Additional PR02 experiments are discussed below.*

**Key pruning findings**

**Fare — full-context improvement versus simplicity**

The result without FE10 was already available from Forward Selection. Pruning therefore investigated whether removing Fare entirely could improve performance.

| Fare representation | Accuracy | F1 |
|---|---:|---:|
| Raw Fare | 0.835 | 0.773 |
| **Raw Fare + FE10 full-context** | **0.836** | **0.774** |
| No Fare | 0.834 | 0.770 |

Removing the entire Fare domain reduced accuracy by 0.002 and F1 by 0.004, supporting the retention of Fare information.

FE10 full-context provided only +0.001 accuracy / +0.001 F1 beyond raw Fare. Although this was the best measured configuration, the simpler raw-Fare alternative remained competitive because it avoided FE10's full-context TicketGroupSize requirement.

**Age — complementary representations**

PR02 investigated whether the complete CB02 representation was necessary:

- **Remove continuous Title-imputed Age, retain bins:** 0.833 accuracy / 0.769 F1 (−0.003/−0.005).
- **Remove the entire Age domain:** 0.834 accuracy / 0.771 F1 (−0.002/−0.003).

Both performed worse than the forward candidate.

Notably, retaining only Age bins performed slightly worse than removing Age altogether. This suggests that the bins were not independently beneficial in the tested configuration, while the combined continuous and binned representation performed best.

CB02 was therefore retained without modification.

**Title**

Removing Title FE05 reduced accuracy by 0.005 and F1 by 0.007, the largest pruning loss among the selected engineered domains.

This contribution was smaller than Title's original improvement over raw SVC (+0.007/+0.011), suggesting that its marginal predictive value decreased after Age and Fare engineering were introduced.

Because CB02 also uses Title internally for Age imputation, some information overlap is plausible. However, the experiments do not establish the underlying mechanism.

Title remained useful as an explicit model input and was retained.

#### Final Selection

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.827 | 0.760 |
| FS baseline — Title FE05 | 0.834 | 0.771 |
| **Final selected candidate** | **0.836** | **0.774** |

**Selected representations**

- **FE05 — Title**
- **CB02 — Age:** Title-based imputation, continuous Age, and Age bins
- **Raw Fare + FE10 — Fare/TicketGroupSize:** Full-context semantics
- Remaining raw predictors not explicitly removed or replaced

**Rejected representations:** FE09 TicketGroupSize as an additional input, Family engineering, Cabin engineering, and Sex × Pclass.

**Improvement over raw baseline:** +0.009 accuracy / +0.014 F1.

**Alternative configuration:** Title FE05 + Age CB02 + raw Fare, without FE10, achieved 0.835 accuracy / 0.773 F1. It remains a competitive inductive alternative when the full prediction context is unavailable or undesirable.

#### Interpretation

SVC showed comparatively limited benefits from the explored feature-engineering representations.

Three findings stood out:

1. **Forward Selection performed most of the filtering.** TicketGroupSize, Family, Cabin, and Sex × Pclass failed to improve the assembled configuration. Every retained representation survived Backward Pruning.
2. **Age benefited from combining representations.** Continuous Title-imputed Age and Age bins performed better together than either the bins alone or the complete removal of Age.
3. **Predictive gains had practical trade-offs.** FE10 full-context produced the best measured performance, but its advantage over raw Fare alone was only +0.001 accuracy / +0.001 F1, making the simpler alternative worth preserving.

SVC's limited improvement is consistent with the possibility that its existing feature representation already captured much of the useful predictive structure. However, the experiments do not establish whether this reflects the model's capabilities or the particular engineered features investigated.

Overall, SVC favored a conservative feature configuration. Forward Selection identified a small set of useful representations, and Backward Pruning confirmed their contribution without producing additional improvements.

The final candidate reached **0.836 accuracy / 0.774 F1**, compared with 0.827 / 0.760 for the raw baseline.

### Random Forest

#### Starting Point

Feature selection started from the raw feature set with **FE05 Title engineering**, which had produced the strongest general-purpose Random Forest result during earlier feature-engineering experiments.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.822 | 0.744 |
| **FS baseline — Title FE05** | **0.832** | **0.768** |

Random Forest had shown limited benefits from most engineered representations. However, earlier Age and Fare experiments suggested that their contributions might depend on the surrounding features.

Forward Selection therefore prioritized these domains, while Backward Pruning investigated whether removing or replacing existing representations could produce a better configuration.

#### Forward Selection

Previously promising Age and Fare representations were tested against the Title baseline.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | Title FE05 | **0.832** | **0.768** | — | — | Reference |
| FS01 | + Age CB03 | 0.831 | 0.764 | -0.001 | -0.004 | Reject |
| FS02 | + Fare CB06 fitted | 0.832 | 0.766 | 0.000 | -0.002 | Reject |

*Deltas compare each test against the Title baseline. Additional experiments within each stage are discussed below.*

**Notable decisions**

- **Age:** CB03 and CB02 were tested because they had previously been among Random Forest's more promising Age representations. CB03 reached 0.831 accuracy / 0.764 F1, while CB02 reached 0.828 / 0.760. Neither improved the Title baseline, so both were rejected as additions to the current configuration.

- **Fare:** FE10 full-context reduced accuracy by 0.004 and F1 by 0.009. CB06 fitted preserved accuracy but reduced F1 by 0.002. Neither provided a reason to increase representation complexity, so both were rejected.

**Forward candidate:** Title FE05 with the remaining raw features.

**Result:** 0.832 accuracy / 0.768 F1.

Unlike Logistic Regression and SVC, Random Forest did not benefit from assembling additional engineered representations during Forward Selection.

#### Backward Pruning

Although Forward Selection found no beneficial additions, earlier Age × Fare experiments had suggested that removing Fare could improve some Random Forest configurations.

This justified investigating feature removal and reconsidering CB03 under different representation contexts.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Forward candidate | Title FE05 | 0.832 | 0.768 | — | — | Reference |
| PR01 | Remove raw Fare | 0.832 | 0.767 | 0.000 | -0.001 | No improvement |
| PR02 | Add CB03 without Fare | 0.829 | 0.766 | -0.003 | -0.002 | Reject |
| PR03 | Remove Title and Fare, retain CB03 | **0.834** | 0.762 | +0.002 | -0.006 | Select for accuracy |

*Deltas compare each configuration against the original Title-based forward candidate, allowing the alternative representations to be evaluated against the same reference.*

**Key pruning findings**

**Fare — context-dependent contribution**

Removing raw Fare from the Title baseline preserved accuracy but slightly reduced F1:

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Title + raw Fare | 0.832 | 0.768 |
| Title without Fare | 0.832 | 0.767 |

Fare was therefore not generally detrimental to Random Forest. Its earlier negative effects appeared to depend on the other representations present.

**Age — reconsidering a rejected candidate**

CB03 had been rejected during Forward Selection because it reduced performance when added to Title.

However, earlier experiments suggested that CB03 might behave differently without Fare. It was therefore reconsidered after Fare removal.

Adding CB03 while retaining Title but excluding Fare reached 0.829 accuracy / 0.766 F1, remaining below the original Title baseline.

Removing Fare alone was consequently insufficient to make CB03 beneficial.

**Title, Age, and Fare — combined representation change**

A final targeted experiment reproduced the previously promising configuration using CB03 without raw Fare or Title as an explicit predictor.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Title baseline | 0.832 | **0.768** |
| Title + CB03 + Fare | 0.831 | 0.764 |
| Title + CB03, without Fare | 0.829 | 0.766 |
| **CB03, without Title or Fare** | **0.834** | 0.762 |

The last configuration achieved the highest Random Forest accuracy observed during Feature Selection.

This revealed an important conditional relationship:

- CB03 reduced performance when added to the Title baseline.
- Removing Fare alone did not make CB03 beneficial.
- Removing both Fare and Title as explicit predictors allowed the CB03 configuration to achieve higher accuracy.

The result also reproduced the earlier Age/Fare ablation finding, supporting its consistency within the existing experimental setup.

**Title remained involved indirectly:** CB03 still used Title alongside Pclass for Age imputation, even though Title was no longer supplied as a separate model input.

#### Final Selection

Random Forest produced two competitive configurations with different metric priorities.

| Configuration | Accuracy | F1 | Characteristic |
|---|---:|---:|---|
| Title FE05 + otherwise raw features | 0.832 | **0.768** | Higher F1 |
| **Age CB03, without Title or Fare** | **0.834** | 0.762 | Higher accuracy |

Because accuracy was the primary selection metric, the CB03 configuration was selected as the final candidate.

**Selected representations**

- **CB03 — Age:** Title + Pclass-based imputation, continuous Age, and Age bins
- Remaining raw predictors, excluding Fare and Title as a separate input

**Removed or rejected representations:** Title as an explicit predictor, raw Fare, additional Fare engineering, and Age CB02.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.822 | 0.744 |
| FS baseline — Title FE05 | 0.832 | **0.768** |
| **Final accuracy-oriented candidate** | **0.834** | 0.762 |

**Improvement over raw baseline:** +0.012 accuracy / +0.018 F1.

Relative to the Title baseline, the final configuration improved accuracy by 0.002 while reducing F1 by 0.006.

**Alternative configuration:** Title FE05 with the otherwise raw feature set remains a competitive candidate when F1 is prioritized or a simpler feature representation is preferred.

#### Interpretation

Random Forest benefited more from selecting between alternative feature representations than from accumulating additional engineered features.

Three findings stood out:

1. **Useful representations did not necessarily combine well.** Title and CB03 Age each supported competitive configurations, but combining them reduced performance relative to the Title baseline.
2. **Feature removal changed the value of other representations.** CB03 achieved its highest accuracy only after both Fare and Title were removed as explicit predictors.
3. **Feature construction and model inputs are distinct.** Title remained useful for CB03's Age imputation even though it was excluded as a separate predictor.

These results indicate that feature contributions were conditional on the surrounding representation. They do not establish the underlying mechanism or imply that Random Forest generally performs better without Fare or Title.

Unlike Logistic Regression, where Backward Pruning simplified a larger forward candidate, Random Forest's pruning stage identified a different feature configuration altogether.

The final candidate reached **0.834 accuracy / 0.762 F1**, while the Title-based alternative retained a slightly higher F1 of 0.768.

The accuracy difference was small, so both configurations remain meaningful candidates for later hyperparameter optimization and independent evaluation.

### K-Nearest Neighbors (KNN)

#### Starting Point

Feature selection started from the raw feature set with **FE05 Title engineering**, which had produced one of KNN's strongest results during feature engineering.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.809 | 0.742 |
| **FS baseline — Title FE05** | **0.822** | **0.760** |

Unlike SVC and Random Forest, KNN had shown promising results from several engineered representations, particularly Fare/TicketGroupSize and Deck. Direct TicketGroupSize had also shown smaller, context-dependent benefits.

Forward Selection prioritized these representations, followed by Family engineering. Backward Pruning would then test whether their contributions remained useful in combination.

#### Forward Selection

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | Title FE05 | 0.822 | 0.760 | — | — | Reference |
| FS01 | + Fare FE10 full-context | 0.831 | 0.771 | +0.009 | +0.011 | Carry |
| FS02 | + Cabin FE03 Deck | 0.832 | 0.768 | +0.001 | -0.003 | Carry provisionally |
| FS03 | + Ticket FE09 full-context | **0.834** | **0.776** | +0.002 | +0.008 | Carry |
| FS04 | + Family FE01 | 0.829 | 0.771 | -0.005 | -0.005 | Reject |

*Deltas compare each test with the latest accepted configuration. Other experiments within each stage are discussed below.*

**Notable decisions**

- **Fare:** FE10 full-context was prioritized because of its strong earlier results. It improved accuracy by 0.009 and F1 by 0.011, making it a clear candidate for retention.

- **Cabin:** FE03 Deck improved accuracy by 0.001 but reduced F1 by 0.003. Despite this mixed result, **Deck was carried provisionally** to investigate possible interactions with later features. The broader FE04 Cabin representation performed worse (0.827 / 0.764), so FE03 was preferred.

- **Ticket:** Adding FE09 full-context improved accuracy by 0.002 and F1 by 0.008. This was particularly interesting because FE10 already used TicketGroupSize indirectly to normalize Fare. Adding FE09 showed that TicketGroupSize could provide additional information as a separate feature. The improvement suggested that TicketGroupSize could provide additional information when supplied as a separate model input.

- **Family:** Earlier Family × Ticket experiments had suggested a possible benefit from combining these domains, justifying another test despite Family's generally weak isolated results. FE01 reduced accuracy and F1 by 0.005; CB07 reduced both by 0.006. Neither was carried forward.

**Forward candidate:** Title FE05 + Fare FE10 full-context + Cabin FE03 Deck + Ticket FE09 full-context.

**Result:** 0.834 accuracy / 0.776 F1.

#### Backward Pruning

Backward Pruning evaluated whether each retained representation remained useful after the complete forward assembly.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Forward candidate | — | **0.834** | **0.776** | — | — | Reference |
| PR01 | Remove Deck FE03 | 0.827 | 0.771 | -0.007 | -0.005 | Keep Deck |
| PR02 | Remove Fare FE10 | 0.828 | 0.766 | -0.006 | -0.010 | Keep FE10 |
| PR03 | Remove Title FE05 | 0.825 | 0.766 | -0.009 | -0.010 | Keep Title |
| Existing FS02 | Remove TicketGroupSize FE09 | 0.832 | 0.768 | -0.002 | -0.008 | Keep FE09 |

*All deltas compare against the complete forward candidate. The existing FS02 result provides the comparison without FE09, so no separate pruning experiment was necessary.*

**Deck × TicketGroupSize interaction**

Deck's contribution changed substantially after TicketGroupSize was introduced.

During Forward Selection, Deck improved accuracy by only 0.001 while reducing F1 by 0.003. However, removing Deck from the completed candidate reduced accuracy by 0.007 and F1 by 0.005.

The experiments provided all four combinations needed to examine this relationship:

| Deck | Direct FE09 TicketGroupSize | Accuracy | F1 |
|---|---|---:|---:|
| No | No | 0.831 | 0.771 |
| Yes | No | 0.832 | 0.768 |
| No | Yes | 0.827 | 0.771 |
| **Yes** | **Yes** | **0.834** | **0.776** |

*All configurations retain Title FE05 and Fare FE10 full-context.*

Without Deck, adding TicketGroupSize reduced accuracy by 0.004 and left F1 unchanged. With Deck, it improved accuracy by 0.002 and F1 by 0.008.

The resulting **interaction contrasts were +0.006 accuracy and +0.008 F1**, showing that the measured contribution of one representation depended on the presence of the other.

This was the strongest reason to retain Deck despite its initially mixed result. Had we rejected it immediately during Forward Selection, we would not have discovered this combination through the planned sequence.

The experiments establish the predictive interaction, but not its mechanism. A plausible explanation is that Deck and TicketGroupSize jointly improve KNN's neighborhood structure; confirming that would require examining distances or nearest-neighbor composition.

**Family without Deck**

Because Family engineering had performed poorly in the completed candidate, a targeted experiment tested whether its contribution changed when Deck was absent.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Complete forward candidate | **0.834** | **0.776** |
| Without Deck | 0.827 | 0.771 |
| Without Deck, with Family FE01 | 0.829 | 0.774 |
| With Deck and Family FE01 | 0.829 | 0.771 |

Without Deck, FE01 improved accuracy by 0.002 and F1 by 0.003 relative to the corresponding configuration without Family engineering.

However, replacing Deck with FE01 remained inferior to the complete candidate (−0.005 accuracy / −0.002 F1). The result revealed another context-dependent contribution but did not justify retaining Family engineering.

**Fare, Title, and TicketGroupSize**

The remaining pruning results supported retaining all three representations:

- **Fare FE10:** Removing it reduced accuracy by 0.006 and F1 by 0.010. Direct TicketGroupSize therefore did not replace the value of normalized Fare.
- **Title FE05:** Removing it reduced accuracy by 0.009 and F1 by 0.010, the largest accuracy loss among the pruning tests.
- **Ticket FE09:** The earlier FS02 configuration already showed that removing FE09 reduced accuracy by 0.002 and F1 by 0.008.

No removal improved the complete forward candidate.

#### Final Selection

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.809 | 0.742 |
| FS baseline — Title FE05 | 0.822 | 0.760 |
| **Final selected candidate** | **0.834** | **0.776** |

**Selected representations**

- **FE05 — Title**
- **FE10 — Fare/TicketGroupSize:** Full-context semantics
- **FE03 — Deck**
- **FE09 — TicketGroupSize:** Full-context semantics, included as a separate model input
- Remaining raw predictors not explicitly replaced or removed

**Rejected representations:** FE04 broader Cabin engineering and FE01/CB07 Family engineering.

**Improvement over raw baseline:** +0.025 accuracy / +0.034 F1.

Relative to the Title baseline, the final configuration improved accuracy by 0.012 and F1 by 0.016.

Unlike Logistic Regression, KNN did not benefit from removing any of the provisionally retained representations.

**Inference requirement:** Both FE10 and FE09 use full-context TicketGroupSize. The final configuration therefore depends on the availability and appropriate use of prediction-population context. It should not be treated as interchangeable with a purely fitted, inductive implementation.

#### Interpretation

KNN benefited from combining engineered representations, with **Deck × TicketGroupSize** providing the clearest example of a conditional contribution.

Three findings stood out:

1. **Provisional retention revealed a useful interaction.** Deck's initial accuracy gain was small and accompanied by an F1 loss. Its contribution became substantially more favorable after TicketGroupSize was introduced.
2. **Positive contributions depended on the surrounding features.** Family FE01 improved the configuration without Deck but reduced performance when Deck was present. A locally beneficial addition was not necessarily part of the best overall configuration.
3. **Related representations were not interchangeable.** Fare/TicketGroupSize and TicketGroupSize both contributed to the completed candidate, despite sharing underlying ticket information.

These findings are consistent with KNN's sensitivity to feature representation: adding or removing features changes distances between observations and can alter which neighbors influence predictions. However, the experiments measured predictive outcomes rather than directly investigating distance geometry, so this remains an explanatory hypothesis.

Overall, KNN demonstrated why Forward Selection and Backward Pruning are complementary. Forward Selection assembled a promising configuration, including a provisionally retained feature, while Backward Pruning confirmed its value and exposed the interaction responsible for its improved contribution.

The final candidate achieved **0.834 accuracy / 0.776 F1**, improving on the raw baseline without hyperparameter tuning.

### XGBoost (XGB)

#### Starting Point

Feature selection started from the raw feature set with **FE05 Title engineering**, which had produced one of XGBoost's strongest improvements during the earlier feature-engineering experiments.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.826 | 0.758 |
| **FS baseline — Title FE05** | **0.836** | **0.772** |

Earlier experiments had also identified promising results from Cabin, Fare, and Family × TicketGroupSize combinations. The objective was to determine whether these benefits remained useful alongside Title and whether additional representations could complement one another.

#### Forward Selection

Feature selection started from the raw XGBoost configuration with **FE05 Title**, which had produced one of the strongest improvements during feature engineering.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.826 | 0.758 |
| **Starting configuration — FE05 Title** | **0.836** | **0.772** |

Earlier experiments had also identified promising Cabin, Fare, and Family × TicketGroupSize representations. Forward Selection investigated whether these benefits remained useful alongside Title and whether additional representations complemented one another.

##### Experiment summary

| Stage | Representative test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | FE05 Title | 0.836 | 0.772 | — | — | Reference |
| FS01 | + FE04 Cabin | 0.831 | 0.766 | −0.005 | −0.006 | Reject |
| FS02 | + CB07 Family | 0.831 | 0.767 | −0.005 | −0.005 | Reject |
| FS03 | + FE09 TicketGroupSize fitted | 0.837 | 0.774 | +0.001 | +0.002 | Carry; full-context alternative |
| FS04 | + CB07 Family with FE09 fitted | 0.836 | 0.774 | −0.001 | 0.000 | Reject Family |
| FS05 | + FE08 FarePerFamilyMember with FE09 fitted | 0.837 | 0.775 | 0.000 | +0.001 | Retain as alternative |
| FS06 | + CB01 Age with FE09 fitted, without FE08 | **0.838** | **0.776** | +0.001 | +0.002 | Carry; final forward candidate |

*FS01–FS03 compare against the Title baseline, except FS03's FE09 addition, which also uses that baseline. FS04 and FS05 compare against Title + FE09 fitted (FS03). FS06 compares against FS03 without FE08. Additional configurations within each stage are discussed below.*

##### Notable decisions

**Cabin — FS01**

FE04 Cabin was tested because it had previously improved XGBoost during feature engineering. However, adding it to Title reduced accuracy by 0.005 and F1 by 0.006.

Its earlier benefit did not transfer to the stronger Title-based configuration, so FE04 was rejected.

**Family — FS02 and FS04**

CB07 Family initially reduced accuracy and F1 by 0.005 when added to Title.

However, earlier feature-engineering experiments had suggested a possible Family × TicketGroupSize interaction. Family was therefore reconsidered during FS04 alongside both FE09 alternatives.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Title + FE09 fitted | **0.837** | 0.774 |
| + CB07 Family | 0.836 | 0.774 |
| Title + FE09 full-context | **0.836** | **0.777** |
| + CB07 Family | 0.829 | 0.768 |

Family failed to improve either TicketGroupSize configuration. The earlier positive interaction therefore did not provide a useful gain in the tested Title-based configurations.

CB07 was rejected.

**TicketGroupSize — FS03**

FE09 was tested with fitted and full-context semantics.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Title baseline | 0.836 | 0.772 |
| **+ FE09 fitted** | **0.837** | 0.774 |
| + FE09 full-context | 0.836 | **0.777** |

Both alternatives offered small improvements but favored different metrics.

**FE09 fitted was carried forward** because it achieved higher accuracy, our primary metric, and supported simpler inductive inference. Full-context remained a competitive alternative because of its higher F1 and was reconsidered after Age engineering.

**Fare — FS05**

Fare engineering was investigated using CB05 and FE10 under fitted and full-context TicketGroupSize semantics. FE08 FarePerFamilyMember was also explored.

CB05 fitted was particularly detrimental, reaching 0.823 accuracy / 0.759 F1. Other Fare representations produced mixed or negligible improvements.

The strongest candidate was **FE08 combined with FE09 fitted**:

| Configuration | Accuracy | F1 |
|---|---:|---:|
| FS03 — Title + FE09 fitted | 0.837 | 0.774 |
| FS05 — Title + FE09 fitted + FE08 | 0.837 | **0.775** |

FE08 preserved accuracy and improved F1 by 0.001. Although small, this was a positive marginal contribution, so the configuration remained a reasonable alternative rather than being conclusively rejected.

Its compatibility with Age engineering was subsequently investigated through additional FS06 experiments.

**Age — FS06**

Age engineering was tested after FE09 fitted had been selected as the provisional reference.

| Configuration | Accuracy | F1 |
|---|---:|---:|
| FS03 — Title + FE09 fitted | 0.837 | 0.774 |
| + FE06 Title-based Age imputation | 0.832 | 0.767 |
| **+ CB01 continuous Age and Age bins** | **0.838** | **0.776** |

FE06 reduced accuracy by 0.005 and F1 by 0.007.

CB01 improved accuracy by 0.001 and F1 by 0.002, so it was carried forward.

**Additional FS06 tests — retaining FE08**

Follow-up experiments investigated whether the small initial benefit from FE08 survived when Age engineering was added.

| Configuration | Accuracy | F1 | Δ Acc. vs FS05 | Δ F1 vs FS05 |
|---|---:|---:|---:|---:|
| FS05 — Title + FE09 fitted + FE08 | 0.837 | 0.775 | — | — |
| + FE06 Age | 0.836 | 0.774 | −0.001 | −0.001 |
| + CB01 Age | 0.833 | 0.769 | −0.004 | −0.006 |
| **Title + FE09 fitted + CB01, without FE08** | **0.838** | **0.776** | — | — |

*The additional FS06 experiment reports use FS03 as their reference. The deltas above are recalculated against FS05 to isolate the marginal effect of adding Age while FE08 is present.*

FE08 had initially improved F1 slightly, but neither Age representation improved the configuration when FE08 was retained.

Most notably, CB01 was beneficial without FE08 (+0.001 accuracy / +0.002 F1) but detrimental with FE08 (−0.004 accuracy / −0.006 F1).

This indicates a **negative conditional interaction between FE08 and CB01** in the tested configurations. It does not establish the underlying mechanism or imply that FE08 is universally detrimental.

The configuration without FE08 achieved higher accuracy and F1 while using fewer engineered representations, so FE08 was excluded from the completed forward candidate.

##### Forward Selection result

**Selected configuration:**

- **FE05 — Title**
- **FE09 — TicketGroupSize**, fitted semantics
- **CB01 — Age**, continuous Age and Age bins
- Remaining raw predictors, except where explicitly modified or replaced

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Starting configuration — Title FE05 | 0.836 | 0.772 |
| **Completed forward candidate** | **0.838** | **0.776** |

Forward Selection improved accuracy by 0.002 and F1 by 0.004 over the Title baseline.

The main finding was not simply that CB01 and FE09 provided small individual gains. Their usefulness depended on the surrounding feature representation: CB01 complemented FE09 but performed worse when FE08 was also retained.

The completed candidate was carried into Backward Pruning to verify whether all three selected representations remained necessary.

#### Backward Pruning

Backward Pruning evaluated whether the selected representations remained useful after completing Forward Selection.

The candidate contained FE05 Title, FE09 TicketGroupSize fitted, and CB01 Age engineering. Each representation was tested for removal to determine whether it still contributed to the assembled configuration.

| Stage | Selected test | Accuracy | F1 | Δ Acc. | Δ F1 | Decision |
|---|---|---:|---:|---:|---:|---|
| Forward candidate | Title + FE09 fitted + CB01 | **0.838** | **0.776** | — | — | Reference |
| PR01 | Remove FE09 TicketGroupSize | 0.833 | 0.768 | −0.005 | −0.008 | Keep FE09 |
| PR02 | Remove FE05 Title | 0.826 | 0.758 | −0.012 | −0.018 | Keep Title |
| FS03 reference | Remove CB01 Age | 0.837 | 0.774 | −0.001 | −0.002 | Keep CB01 |

*Deltas compare each configuration against the completed forward candidate. The CB01 removal result was already available from FS03, so an additional pruning experiment was unnecessary.*

**Notable findings**

**Title**

Removing FE05 Title caused the largest deterioration, reducing accuracy by 0.012 and F1 by 0.018.

The resulting scores (0.826 / 0.758) matched the raw XGBoost baseline. Although this does not mean the configurations produced identical predictions, it shows that FE09 and CB01 did not improve aggregate cross-validation performance beyond the raw baseline when Title was absent.

Title was therefore retained.

**Age × TicketGroupSize interaction**

FE09 initially improved accuracy by only 0.001 and F1 by 0.002 when added to Title. However, removing FE09 from the completed candidate reduced accuracy by 0.005 and F1 by 0.008.

The available experiments allow us to examine how the contributions of CB01 and FE09 depended on one another:

| CB01 Age | FE09 TicketGroupSize fitted | Accuracy | F1 |
|---|---|---:|---:|
| No | No | 0.836 | 0.772 |
| No | Yes | 0.837 | 0.774 |
| Yes | No | 0.833 | 0.768 |
| **Yes** | **Yes** | **0.838** | **0.776** |

*All configurations retain FE05 Title.*

Without CB01, adding FE09 improved accuracy by 0.001 and F1 by 0.002. With CB01, the same addition improved accuracy by 0.005 and F1 by 0.008.

The resulting **interaction contrasts were +0.004 accuracy and +0.006 F1**.

CB01 showed the same conditional behavior: it reduced performance when FE09 was absent but improved both metrics when FE09 was present.

These results support a predictive interaction between CB01 and FE09 within the tested configurations. They do not establish the underlying mechanism or guarantee that the interaction will generalize.

Both representations were retained.

**Fitted versus full-context TicketGroupSize**

During FS03, FE09 full-context had remained an alternative because it produced higher F1 than fitted semantics when combined with Title alone.

After CB01 was introduced, the two approaches were compared again:

| FE09 semantics | Accuracy | F1 |
|---|---:|---:|
| **Fitted** | **0.838** | **0.776** |
| Full-context | 0.834 | 0.774 |

Full-context reduced accuracy by 0.004 and F1 by 0.002.

The earlier F1 advantage of full-context FE09 therefore did not persist after Age engineering was added.

**Fitted FE09 was retained**, providing both better measured performance and simpler inductive inference requirements.

No removal or alternative improved the completed forward candidate.

#### Final Selection

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.826 | 0.758 |
| FS baseline — FE05 Title | 0.836 | 0.772 |
| **Final selected candidate** | **0.838** | **0.776** |

**Selected representations**

- **FE05 — Title**
- **FE09 — TicketGroupSize:** Fitted semantics
- **CB01 — Age:** Continuous Age and Age bins
- Remaining raw predictors, except where explicitly modified or replaced

**Rejected or excluded representations:** Cabin FE04, Family CB07, Fare engineering including FE08, FE06 Age imputation, and FE09 full-context.

**Improvement over raw baseline:** +0.012 accuracy / +0.018 F1.

Relative to the Title baseline, the final configuration improved accuracy by 0.002 and F1 by 0.004.

No selected representation was removed during Backward Pruning.

The final candidate uses fitted TicketGroupSize semantics, avoiding the additional prediction-population context required by the full-context alternative.

#### Interpretation

XGBoost showed a selective response to feature engineering. Several representations that had performed well during earlier isolated experiments failed to improve the stronger Title-based configuration.

Four findings stood out:

1. **Title remained the most influential engineered representation.** Removing FE05 from the completed candidate reduced accuracy and F1 to the raw baseline values. Its contribution was considerably larger than the gains from the later additions.

2. **Age and TicketGroupSize had complementary, conditional contributions.** CB01 reduced performance without FE09 but improved both metrics when FE09 was present. Likewise, FE09 became more valuable after CB01 was introduced. Their positive interaction helped justify retaining both despite their small individual forward-selection gains.

3. **A positive marginal gain did not guarantee compatibility with later features.** FE08 FarePerFamilyMember initially preserved accuracy and improved F1 by 0.001. However, additional FS06 experiments showed that retaining FE08 alongside CB01 produced worse results than using CB01 without FE08. This supported excluding FE08 from the final configuration, while preserving the finding that it had been slightly beneficial in the earlier context.

4. **Inference semantics affected both performance and practicality.** FE09 full-context initially offered higher F1 with Title alone, but fitted FE09 outperformed it on both metrics after CB01 was added. The selected configuration therefore achieved better measured performance without full-context inference requirements.

These findings are consistent with the possibility that XGBoost already captures some useful nonlinear relationships and interactions from the original predictors, limiting the additional value of explicit feature engineering. However, the experiments do not directly establish this mechanism.

The observed differences are also based on cross-validation results used during feature selection. Small gains should therefore be interpreted cautiously rather than as independent evidence of improved generalization.

Overall, XGBoost benefited from **a compact set of complementary representations rather than accumulating every feature with a positive isolated result**.

The final candidate achieved **0.838 accuracy / 0.776 F1**, improving upon the raw baseline while retaining fitted, inductive TicketGroupSize semantics.

The main lesson was the importance of evaluating engineered features in their assembled context: both the positive Age × TicketGroupSize interaction and the unfavorable FE08 × Age combination would be missed by considering only isolated feature contributions.

## Extra Trees — Feature Selection

### Starting point

Feature selection began with **FE05 (Title)**, which was the second-best individual feature-engineering candidate for Extra Trees. Although Age CB02 had performed better in isolated feature-engineering experiments, Title was selected as the starting point to investigate whether combining representations could produce stronger results.

| Configuration | CV Accuracy | CV F1 |
|---|---:|---:|
| Raw baseline | 0.804 | 0.721 |
| FS baseline — Title FE05 | 0.820 | 0.754 |
| **Final selection — PR06** | **0.831** | **0.767** |

The final configuration improved accuracy by **0.027** and F1 by **0.046** over the raw baseline.

### Forward selection

**FS01 — Age**

Both Age CB02 (Title-imputed Age + bins) and CB03 (Title + Pclass-imputed Age + bins) improved the Title baseline by approximately **+0.002 accuracy / +0.002 F1**, reaching 0.822 / 0.756.

CB02 was retained because it had performed better in the earlier isolated Age experiments and offered a simpler representation.

**FS02 — Fare**

Fare CB09 introduced Fare-per-family and Fare-per-ticket representations, with TicketGroupSize computed using either fitted or full-prediction-context semantics.

- **Fitted:** 0.826 accuracy / 0.763 F1, improving the previous configuration by +0.004 / +0.007.
- **Full-context:** 0.823 accuracy / 0.758 F1, improving the previous configuration by +0.001 / +0.002.

The fitted variant was selected for subsequent forward experiments because it provided the stronger improvement at this stage.

**FS03 — Cabin**

Adding Cabin features to Title + Age CB02 + Fare CB09 fitted reduced performance:

- FE02 (HasCabin): −0.007 accuracy / −0.011 F1.
- FE04 (Cabin features): −0.009 accuracy / −0.013 F1.

Both were rejected. Cabin engineering did not provide a useful improvement in the tested assembled configurations.

**FS04 — Family**

Two Family representations were evaluated:

- FE01: unchanged accuracy, −0.002 F1.
- CB07: unchanged accuracy and F1.

CB07 was provisionally retained despite its neutral result, allowing potential interactions with Ticket features to be investigated. Unlike FE01, CB07 preserved the original SibSp and Parch inputs.

**FS05 — Ticket**

Direct TicketGroupSize (FE09) was added to the configuration containing Title, Age CB02, Fare CB09, and Family CB07. Fare CB09 and FE09 used matching TicketGroupSize semantics in each experiment.

- **Fitted Fare CB09 + fitted FE09:** 0.824 / 0.760.
- **Full-context Fare CB09 + full-context FE09:** 0.829 / 0.766.

The full-context combination became the best observed configuration at this stage and was carried into backward pruning.

This result does not establish that full-context TicketGroupSize is universally preferable. The comparison changes the TicketGroupSize semantics used by both Fare CB09 and FE09, and the earlier Fare-only experiment had favored fitted semantics.

### Backward pruning

Pruning investigated whether the selected representations remained useful once other engineered features were available.

| Experiment | Change | Accuracy | F1 |
|---|---|---:|---:|
| FS05 | Full forward-selected configuration | 0.829 | 0.766 |
| PR01 | Remove Family CB07 | 0.829 | 0.766 |
| PR02 | Remove Title FE05 from PR01 | 0.818 | 0.745 |
| PR03 | Remove Ticket FE09 from FS05 | 0.829 | 0.766 |
| PR04 | Remove Age CB02 from PR01 | 0.823 | 0.757 |
| PR05 | Remove Fare CB09 from PR01 | 0.828 | 0.765 |
| **PR06** | **Add Family CB07 to PR05** | **0.831** | **0.767** |
| PR07 | Remove Ticket FE09 from PR06 | 0.828 | 0.765 |

#### Title and Age remained important

Removing Title in PR02 reduced accuracy by **0.011** and F1 by **0.021** relative to PR01.

This occurred even though Age CB02 continued to use Title internally for Age imputation. Title therefore retained useful predictive information when exposed directly to Extra Trees.

Removing Age CB02 in PR04 reduced accuracy by **0.006** and F1 by **0.009** relative to PR01. This was a larger deterioration than the improvement observed when Age was first introduced, indicating that its measured contribution depended on the surrounding features.

Both representations were retained.

#### Family, Fare, and Ticket showed conditional redundancy

Removing Family from FS05 produced no measurable change. Removing TicketGroupSize while retaining Family and Fare also produced no measurable change.

However, removing Fare from the Family-free configuration caused only a small decline, from 0.829 / 0.766 to 0.828 / 0.765. Adding Family back **without Fare** then improved performance to **0.831 / 0.767**.

This provides a useful contrast:

| Context | Effect of Family CB07 on accuracy / F1 |
|---|---|
| Fare CB09 full-context + Ticket FE09 full-context | 0.000 / 0.000 |
| No Fare CB09 + Ticket FE09 full-context | +0.003 / +0.002 |

Family was neutral when Fare-derived representations were available, but beneficial when Fare was absent.

This is consistent with **conditional redundancy**: Family and Fare-derived features may supply overlapping information, making Family's measured contribution dependent on the remaining feature set. The experiments demonstrate the performance pattern, rather than proving a specific internal mechanism.

Finally, PR07 removed TicketGroupSize from PR06. Performance declined by **0.003 accuracy / 0.002 F1**, confirming a small measured benefit from retaining it in the final selected configuration.

### Final configuration

**Selected: PR06 — Title + Age CB02 + Family CB07 + Ticket FE09 full-context**

| Metric | Raw baseline | Title baseline | PR06 |
|---|---:|---:|---:|
| Accuracy | 0.804 | 0.820 | **0.831** |
| F1 | 0.721 | 0.754 | **0.767** |

Retained engineered representations:

- **FE05 — Title:** Direct categorical information extracted from passenger names.
- **CB02 — Age:** Title-based Age imputation and Age bins.
- **CB07 — Family:** Engineered family representations while retaining raw SibSp and Parch.
- **FE09 — TicketGroupSize:** Direct full-context TicketGroupSize.

Fare CB09 and Cabin engineering were excluded.

The selected configuration requires **full prediction context** to construct TicketGroupSize consistently. This inference dependency must be preserved when using the model.

**Practical alternative: PR07 — Title + Age CB02 + Family CB07**

| Metric | PR06 (selected) | PR07 (alternative) |
|---|---:|---:|
| Accuracy | **0.831** | 0.828 |
| F1 | **0.767** | 0.765 |
| Full prediction context required | Yes | No |

PR07 removes the full-context TicketGroupSize requirement while sacrificing only a small amount of observed cross-validation performance. It is therefore a useful alternative when simpler, inductive inference is preferred.

The differences are small, and the repeated use of cross-validation results for feature selection means the selected scores should not be treated as independent estimates of generalization performance.

### Conclusions and model behavior

**Verdict:** Feature engineering improved Extra Trees over both its raw and Title-only baselines, but additional representations were not consistently beneficial.

Key observations:

1. **Title remained valuable as a direct feature.** Using Title indirectly for Age imputation did not eliminate the benefit of supplying Title directly to the model.
2. **Age engineering became more important in the assembled configuration.** Its removal caused a larger loss than its initial forward-selection gain.
3. **Cabin engineering was detrimental in the tested combinations.** Both HasCabin and the broader Cabin feature set reduced performance during forward selection.
4. **Fare and Family showed conditional redundancy.** Family was neutral alongside Fare but became beneficial once Fare was removed.
5. **TicketGroupSize offered a small final gain.** Its contribution depended on the other available representations, and its full-context requirement created a practical inference trade-off.
6. **Backward pruning was essential.** The strongest final configuration was not simply the accumulation of every feature retained during forward selection.

**Overall conclusion:** Extra Trees benefited from a compact combination of Title, Age, Family, and Ticket representations. Its most informative behavior was the context-dependent usefulness of Family and Fare: a feature that appeared unnecessary in one configuration became useful after another representation was removed.

PR06 was selected as the best observed configuration, while PR07 was retained as a simpler alternative. Further combinations were not pursued because the remaining performance differences were small relative to the additional experimentation and complexity.

## Decision Tree — Feature Selection

### Overview

**Objective:** Identify a feature configuration that improves Decision Tree classification performance while minimizing unnecessary feature engineering and inference complexity.

**Selection criteria:**
- Primary metric: Cross-validation accuracy.
- Secondary metric: Cross-validation F1.
- Prefer simpler, inductive configurations when performance is comparable.
- Retain neutral features provisionally when there is a specific interaction worth investigating.

### Baseline and final results

| Configuration | Accuracy | F1 |
|---|---:|---:|
| Raw baseline | 0.803 | 0.702 |
| FS baseline — Title FE05 | 0.823 | 0.756 |
| **Final — Title FE05 + Family CB07 + Cabin FE02** | **0.827** | **0.765** |

**Improvement over raw baseline:** +0.024 accuracy and +0.063 F1.

**Improvement over Title baseline:** +0.004 accuracy and +0.009 F1.

The final configuration preserves the highest reported accuracy among the leading candidates while achieving the best reported F1 of the selected configurations. It also avoids the full-prediction-context dependency introduced by some Fare and TicketGroupSize patches.

---

### Forward selection

The forward-selection phase started from the raw Decision Tree configuration with Title FE05 added. Each domain was explored incrementally, carrying forward promising or potentially informative representations.

#### FS01 — Fare

**Reference:** Title FE05 — accuracy 0.823, F1 0.756.

| Fare patch | Accuracy | F1 | Δ Accuracy | Δ F1 |
|---|---:|---:|---:|---:|
| CB06 fitted | 0.820 | 0.761 | -0.003 | +0.005 |
| CB06 batch | 0.820 | 0.760 | -0.003 | +0.004 |
| **CB06 full-context** | **0.827** | **0.763** | **+0.004** | **+0.007** |

**Decision:** Carry forward Fare CB06 full-context.

**Interpretation:** Fare engineering improved performance when its TicketGroupSize-dependent features used full prediction context. Fitted and batch variants improved F1 but reduced accuracy.

The ranking differed from earlier isolated Fare experiments, demonstrating that the preferred representation can depend on the features already present.

**Important:** CB06 full-context uses full-context TicketGroupSize internally to derive Fare-related features. Removing Ticket FE09 does not remove this dependency while CB06 remains.

#### FS02 — Age

**Reference:** Title FE05 + Fare CB06 full-context — accuracy 0.827, F1 0.763.

| Age patch | Accuracy | F1 | Δ Accuracy | Δ F1 |
|---|---:|---:|---:|---:|
| FE11 — Age bins | 0.819 | 0.753 | -0.008 | -0.010 |
| FE06 — Title-based Age imputation | 0.826 | 0.757 | -0.001 | -0.006 |
| CB02 — Title-imputed Age + bins | 0.826 | 0.757 | -0.001 | -0.006 |

**Decision:** Reject additional Age engineering.

**Interpretation:** None of the tested Age representations improved upon the current configuration. Raw Age remained available through the baseline features.

Although Age engineering showed promise in earlier isolated experiments, that benefit did not carry over to the selected Title + Fare configuration.

#### FS03 — Family

**Reference:** Title FE05 + Fare CB06 full-context — accuracy 0.827, F1 0.763.

| Family patch | Accuracy | F1 | Δ Accuracy | Δ F1 |
|---|---:|---:|---:|---:|
| CB07 — Raw family features + FamilySize + IsAlone | 0.827 | 0.763 | 0 | 0 |
| FE01 — FamilySize + IsAlone replacing SibSp/Parch | 0.827 | 0.763 | 0 | 0 |

**Decision:** Carry forward CB07 provisionally.

**Interpretation:** Both Family representations were neutral at the reported precision. CB07 was retained because it preserves the original SibSp and Parch information alongside the engineered features, allowing further investigation of interactions with TicketGroupSize.

Its retention was exploratory, not evidence of an independent performance gain.

#### FS04 — Ticket

**Reference:** Title FE05 + Fare CB06 full-context — accuracy 0.827, F1 0.763.

All FS04 configurations also included Family CB07. The Fare CB06 and Ticket FE09 variants used matching TicketGroupSize semantics.

| Ticket/Fare context | Accuracy | F1 | Δ Accuracy | Δ F1 |
|---|---:|---:|---:|---:|
| Full-context | 0.826 | 0.764 | -0.001 | +0.001 |
| Fitted | 0.813 | 0.754 | -0.014 | -0.009 |
| Batch | 0.813 | 0.753 | -0.014 | -0.010 |

**Decision:** Carry forward the full-context configuration provisionally; reject fitted and batch combinations.

**Interpretation:** Adding Family CB07 and Ticket FE09 full-context produced an effectively neutral trade-off relative to FS01.

Fitted and batch configurations performed considerably worse. However, those comparisons changed both the Ticket patch and the TicketGroupSize semantics inside Fare CB06, so their losses cannot be attributed solely to TicketGroupSize.

Direct Ticket FE09 was retained temporarily to investigate whether its contribution would change after removing other patches.

#### FS05 — Cabin

**Reference:** FS04 — Title FE05 + Fare CB06 full-context + Family CB07 + Ticket FE09 full-context; accuracy 0.826, F1 0.764.

| Cabin patch | Accuracy | F1 | Δ Accuracy | Δ F1 |
|---|---:|---:|---:|---:|
| FE04 — Cabin features | 0.826 | 0.764 | 0 | 0 |
| FE02 — HasCabin | 0.826 | 0.764 | 0 | 0 |

**Decision:** Carry forward Cabin FE04 provisionally.

**Interpretation:** Both Cabin representations were neutral in the complete forward-selected configuration.

FE04 was initially retained to explore whether the broader Cabin representation would become useful when other feature groups were removed.

### Forward-selection conclusion

The best accuracy observed during forward selection was **0.827**, achieved by Title FE05 + Fare CB06 full-context at FS01.

Later additions of Family, Ticket, and Cabin did not produce a convincing improvement over that result.

However, their neutral behavior did not establish that they were universally unnecessary. Backward pruning was used to investigate whether overlapping representations were concealing useful contributions.

---

### Backward pruning

**Starting configuration:** Title FE05 + Fare CB06 full-context + Family CB07 + Ticket FE09 full-context + Cabin FE04.

This configuration achieved 0.826 accuracy and 0.764 F1, identical to FS04 at the reported precision.

All pruning results below were originally reported against FS04. The interpretations additionally compare configurations that differ by specific patches, where such comparisons are available.

#### PR01 — Remove Fare CB06

| Retained engineered patches | Accuracy | F1 |
|---|---:|---:|
| Title + Fare CB06 + Family + Ticket + Cabin FE04 | 0.826 | 0.764 |
| Title + Family + Ticket + Cabin FE04 | **0.827** | 0.763 |
| Title + Family + Ticket (without Fare or Cabin) | 0.820 | 0.757 |

**Interpretation:**

Removing Fare CB06 while retaining Cabin FE04 preserved accuracy and F1 approximately.

However, removing both Fare CB06 and Cabin FE04 caused a larger decline.

The results revealed conditional usefulness:

- With Fare present, Cabin FE04 had no measured benefit.
- Without Fare, Cabin FE04 improved accuracy by 0.007 and F1 by 0.006.
- With Cabin present, removing Fare changed accuracy by +0.001 and F1 by -0.001.
- Without Cabin, removing Fare reduced accuracy by 0.006 and F1 by 0.007.

**Conclusion:** Fare CB06 and Cabin FE04 appeared partially substitutable in this configuration. Their combined use offered no measurable advantage over retaining either representation alone.

This justified exploring separate Fare-based and Cabin-based pruning branches.

#### PR02 — Remove Title FE05

| Retained engineered patches | Accuracy | F1 |
|---|---:|---:|
| Fare CB06 + Family + Ticket + Cabin FE04 | 0.809 | 0.735 |
| Fare CB06 + Family + Ticket | 0.809 | 0.735 |
| Family + Ticket + Cabin FE04 | 0.806 | 0.715 |

For comparison, the corresponding configurations retaining Title achieved approximately 0.826–0.827 accuracy and 0.763–0.764 F1.

**Interpretation:** Removing Title caused substantial performance losses in every tested branch.

Neither Fare nor Cabin, separately or together, compensated for the absence of Title engineering.

The Fare-based configuration performed better than the Cabin-only configuration when Title was absent, especially in F1. Nevertheless, both were inferior to their Title-containing counterparts.

**Conclusion:** Retain Title FE05. It was the most consistently valuable engineered representation in Decision Tree feature selection.

#### PR03 — Remove Ticket FE09

The corrected primary PR03 experiments used Cabin FE04. An earlier run accidentally used FE02; those results are preserved separately because they provided a useful Cabin-representation comparison.

**Corrected PR03 results:**

| Retained engineered patches | Accuracy | F1 | Full-context dependency |
|---|---:|---:|---|
| Title + Fare CB06 + Family + Cabin FE04 | 0.825 | 0.761 | Yes |
| Title + Family + Cabin FE04 | **0.827** | **0.763** | No |
| Title + Fare CB06 + Family | **0.827** | **0.763** | Yes |

**Earlier FE02 comparison:**

| Retained engineered patches | Accuracy | F1 |
|---|---:|---:|
| Title + Fare CB06 + Family + Cabin FE02 | 0.826 | 0.762 |
| Title + Family + Cabin FE02 | **0.827** | **0.765** |

**Interpretation:**

Direct Ticket FE09 was not necessary to maintain competitive performance in the reduced configurations.

The corrected Cabin-only and Fare-only branches both achieved 0.827 accuracy and 0.763 F1.

The earlier FE02-based Cabin branch achieved the same accuracy but slightly higher F1 (0.765).

Crucially, removing both Fare CB06 and Ticket FE09 eliminated the full-context TicketGroupSize dependency. Removing FE09 alone did not, because CB06 still used full-context TicketGroupSize internally.

**Conclusion:** Direct Ticket FE09 was unnecessary for the leading reduced configurations. The inductive Cabin branch became especially attractive because its performance matched the full-context Fare branch.

#### PR04 — Remove Family CB07

The leading inductive Cabin branch was tested with Family removed, using both Cabin representations.

| Retained engineered patches | Accuracy | F1 |
|---|---:|---:|
| Title + Family CB07 + Cabin FE04 | **0.827** | **0.763** |
| Title + Cabin FE04 | 0.826 | 0.758 |
| Title + Family CB07 + Cabin FE02 | **0.827** | **0.765** |
| Title + Cabin FE02 | 0.826 | 0.760 |

**Interpretation:**

Removing Family CB07 reduced accuracy by 0.001 and F1 by 0.005 with either Cabin representation.

This is notable because Family CB07 had been neutral during forward selection when Fare CB06 was present.

Family became more useful after Fare and TicketGroupSize were removed.

Additionally, Cabin FE02 consistently achieved 0.002 higher F1 than FE04, with equal accuracy, in both Family-containing and Family-removed configurations.

**Conclusion:** Retain Family CB07 in the Cabin branch. Prefer the simpler Cabin FE02 representation over FE04, based on its repeated small F1 advantage.

For the Fare branch, removing Family CB07 returned to the previously tested FS01 configuration, which achieved 0.827 accuracy and 0.763 F1. Therefore, Family was not needed to maintain performance in that branch.

#### PR05 — Remove Cabin FE02

**Reference:** Title FE05 + Family CB07 + Cabin FE02 — accuracy 0.827, F1 0.765.

| Retained engineered patches | Accuracy | F1 |
|---|---:|---:|
| Title + Family CB07 + Cabin FE02 | **0.827** | **0.765** |
| Title + Family CB07 | 0.824 | 0.761 |
| **Difference after removing Cabin** | **-0.003** | **-0.004** |

**Interpretation:**

Removing Cabin FE02 reduced both metrics.

Although the decline was modest, it supported retaining Cabin alongside Title and Family.

Together with PR04, this showed that both Family CB07 and Cabin FE02 contributed to the selected reduced configuration.

**Conclusion:** Retain Cabin FE02. No further feature pruning was justified by the available results.

---

### Feature interaction findings

The pruning phase revealed several important relationships.

| Feature or relationship | Observation | Interpretation |
|---|---|---|
| Title FE05 | Removing Title caused substantial losses | Consistently useful across tested branches |
| Fare CB06 ↔ Cabin | Cabin became useful after Fare removal | Conditional redundancy or substitutability |
| Family CB07 | Neutral with Fare, beneficial in the reduced Cabin branch | Conditional usefulness |
| Ticket FE09 | Removing Ticket preserved competitive performance | Not necessary in the leading configurations |
| Cabin FE02 vs FE04 | FE02 repeatedly gained +0.002 F1 in reduced configurations | Simpler Cabin representation favored |
| Full-context engineering | Competitive, but unnecessary for the selected configuration | Inductive alternative preferred |

These observations are based on cross-validation mean scores. They describe measured relationships between configurations, not proven causal mechanisms or statistically established improvements.

In particular, equal aggregate scores do not establish identical predictions, and small metric differences may be sensitive to the training folds or model randomness.

### Final configuration

**Selected:** Title FE05 + Family CB07 + Cabin FE02.

| Feature group | Selected patch | Purpose |
|---|---|---|
| Title | `fe05__title_patch` | Extract passenger title information |
| Family | `cb07__family_features_patch` | Retain SibSp/Parch and add FamilySize/IsAlone |
| Cabin | `fe02__has_cabin_patch` | Represent whether Cabin information is available |

The configuration retains the baseline raw features as governed by the patch system and does not use additional Age, Fare, or Ticket engineering.

**Final cross-validation results:**

| Metric | Raw baseline | Title baseline | Final | Δ vs raw |
|---|---:|---:|---:|---:|
| Accuracy | 0.803 | 0.823 | **0.827** | **+0.024** |
| F1 | 0.702 | 0.756 | **0.765** | **+0.063** |

**Why this configuration was selected:**

- Ties the highest reported accuracy among the leading candidates.
- Achieves the highest reported F1 among those candidates.
- Requires fewer engineered representations than the complete forward-selected configuration.
- Avoids the prediction-cohort dependency of full-context TicketGroupSize.
- Has targeted pruning comparisons supporting retention of Title, Family, and Cabin.

**Competitive alternative:** Title FE05 + Fare CB06 full-context — accuracy 0.827, F1 0.763.

This alternative should remain documented because future hyperparameter tuning could change the relative performance of the feature sets.

### Final conclusion

Decision Tree benefited most consistently from Title engineering. Fare CB06 full-context provided a useful early improvement, but subsequent pruning revealed that a simpler combination of Title, Family, and Cabin could match its accuracy while avoiding full-context inference.

The main finding was **conditional feature usefulness**: Family and Cabin appeared neutral when overlapping Fare representations were present, but became beneficial after those representations were removed.

Backward pruning therefore contributed more than model simplification. It revealed relationships that forward selection alone would have missed.

**Final selection: Title FE05 + Family CB07 + Cabin FE02 — accuracy 0.827, F1 0.765.**

Feature selection is complete for Decision Tree. The selected configuration is the primary candidate for later hyperparameter tuning, with Title + Fare CB06 full-context retained as an alternative.


## Lessons learned

- Recovering missing information (cabin, Age Imputation).
- finding hidden information (title).
- Changing representation of existing information (age_bin)

Hypothesis ✓ Confirmed
Hypothesis ✗ Rejected
Hypothesis ~ Partially confirmed