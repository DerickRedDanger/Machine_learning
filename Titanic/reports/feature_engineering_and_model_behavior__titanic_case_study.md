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


## Lessons learned

- Recovering missing information (cabin, Age Imputation).
- finding hidden information (title).
- Changing representation of existing information (age_bin)

Hypothesis ✓ Confirmed
Hypothesis ✗ Rejected
Hypothesis ~ Partially confirmed