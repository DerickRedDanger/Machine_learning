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

### cb07__family_features

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

Since Sex, Pclass and Fare already capture part of this information, I initially expected Title to provide only a modest improvement. Nevertheless, because it combines multiple characteristics into a single feature, it was worth investigating. Additionally, Title may prove useful for imputing missing Age values.

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

Title feature produced the largest improvements so far and it extended to all models, meaning this is a good feature to use in future datasets.

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

Age have 20% of its values missing, imputing with median is decent, but not nescesarely the best. This experiments test how good imputing age with group by title is, and how this would affect the accuracy of the models.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_positive
- Recommended for specific models:
  - decision_tree: test_accuracy_mean: 0.005
  - random_forest: test_accuracy_mean: 0.003

##### Conclusion

Overall, imputing by title lead to gains, even if only slightly.

The relatively small improvements suggest that either the missing proportion is too small to have a large impact, or that median imputation was already providing a reasonable approximation.

Interesting how Xgb and Extra tree had no change in accuracy and barely any in f1. They didn't care about the changes in age? Or they already got the information they needed from the other features?

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group              | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe06__age_imputation_title | logreg        |                          0.786 |                        0.787 |                      0.001 |                    0.713 |                  0.712 |               -0.001 |
| baseline__raw     | fe06__age_imputation_title | knn           |                          0.809 |                        0.808 |                     -0.001 |                    0.742 |                  0.739 |               -0.003 |
| baseline__raw     | fe06__age_imputation_title | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe06__age_imputation_title | decision_tree |                          0.803 |                        0.808 |                      0.005 |                    0.702 |                  0.711 |                0.009 |
| baseline__raw     | fe06__age_imputation_title | random_forest |                          0.822 |                        0.825 |                      0.003 |                    0.744 |                  0.749 |                0.005 |
| baseline__raw     | fe06__age_imputation_title | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | fe06__age_imputation_title | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.756 |               -0.002 |

##### Summary

| compare_group              |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe06__age_imputation_title |                      0.00142857 |                         -0.001 |                          0.005 |                0.00171429 |                   -0.003 |                    0.009 |

</details>

#### fe07__age_imputation_title_pclass

Testing the effect of imputing using both Title and Pclass, expecting better results than imputing with title alone.

<details>
<summary>Conclusion</summary>

##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.013
    - Secondary gains:
      - test_f1_mean: 0.012
  - random_forest: test_accuracy_mean: 0.004

##### Conclusion

Imputing Age using Title and Pclass produced better results than imputing by Title alone. 

The strongest improvement was observed in Logistic Regression, suggesting that the more refined age estimates create relationships that are easier for linear models to exploit. 

The overall gains remain small, likely because Age is only missing for approximately 20% of passengers. Nevertheless, the experiment indicates that Pclass provides useful contextual information when estimating missing ages.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                     | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe07__age_imputation_title_pclass | logreg        |                          0.786 |                        0.799 |                      0.013 |                    0.713 |                  0.725 |                0.012 |
| baseline__raw     | fe07__age_imputation_title_pclass | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.737 |               -0.005 |
| baseline__raw     | fe07__age_imputation_title_pclass | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe07__age_imputation_title_pclass | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | fe07__age_imputation_title_pclass | random_forest |                          0.822 |                        0.826 |                      0.004 |                    0.744 |                  0.75  |                0.006 |
| baseline__raw     | fe07__age_imputation_title_pclass | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | fe07__age_imputation_title_pclass | xgb           |                          0.826 |                        0.828 |                      0.002 |                    0.758 |                  0.761 |                0.003 |

##### Summary

| compare_group                     |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe07__age_imputation_title_pclass |                           0.002 |                         -0.007 |                          0.013 |                0.00242857 |                   -0.005 |                    0.012 |

</details>

#### Age representation:

#### fe11__age_bin

Test whether changing the representation of Age changes what the models learn.

These bin were made based from previous exploratory experiments, while its order are meant to represent the survival rate of the passenger based on its age, rather than the age itself.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: model_specific_mixed
- Recommended for specific models:
  - decision_tree: 0.006
  - random_forest: 0.011
  - extra_trees: 0.015
- Notable secondary improvements:
  - decision_tree: test_f1_mean: 0.02
  - random_forest: test_f1_mean: 0.015
  - extra_trees: test_f1_mean: 0.016


##### Conclusion

Age bin's results were significant, Logreg suffered slightly, Knn and Xgb had small accuracy gains while taking some losses in f1. Scv had small gains.

But compared to them, Decision, random and extra tree had significant gains on both accuracy and F1. This' likely because the binning facilitated the creation of their's threshold, while the binning themselves apparently align well with the survival behavior.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe11__age_bin   | logreg        |                          0.786 |                        0.783 |                     -0.003 |                    0.713 |                  0.71  |               -0.003 |
| baseline__raw     | fe11__age_bin   | knn           |                          0.809 |                        0.81  |                      0.001 |                    0.742 |                  0.737 |               -0.005 |
| baseline__raw     | fe11__age_bin   | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe11__age_bin   | decision_tree |                          0.803 |                        0.809 |                      0.006 |                    0.702 |                  0.722 |                0.02  |
| baseline__raw     | fe11__age_bin   | random_forest |                          0.822 |                        0.833 |                      0.011 |                    0.744 |                  0.759 |                0.015 |
| baseline__raw     | fe11__age_bin   | extra_trees   |                          0.804 |                        0.819 |                      0.015 |                    0.721 |                  0.737 |                0.016 |
| baseline__raw     | fe11__age_bin   | xgb           |                          0.826 |                        0.827 |                      0.001 |                    0.758 |                  0.752 |               -0.006 |

##### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe11__age_bin   |                      0.00471429 |                         -0.003 |                          0.015 |                0.00585714 |                   -0.006 |                     0.02 |

</details>

#### Age representation combination

#### cb01__age_and_bins

Combo using both raw Age and Age_bin.

This experiment explores whether the original continuous Age feature and its binned version provide complementary information, or mostly overlap.

It also serves as a proof of concept for combo experiments: testing whether the effect of combining features can be predicted from their individual effects, or whether combinations behave differently enough to require separate evaluation.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.008
  - extra_trees: test_accuracy_mean: 0.014
    - Secondary gains:
      - test_f1_mean: 0.024


##### Conclusion

Using both continuous Age and Age_bin provides little universal benefit. Most models perform similarly to using one representation alone, while Decision Tree and Random Forest lose much of the benefit obtained from Age_bin by itself. Logistic Regression is the primary exception, suggesting that the continuous and ordinal representations contain complementary information for linear models.

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
| baseline__raw     | cb01__age_and_bins | random_forest |                          0.822 |                        0.824 |                      0.002 |                    0.744 |                  0.748 |                0.004 |
| baseline__raw     | cb01__age_and_bins | extra_trees   |                          0.804 |                        0.818 |                      0.014 |                    0.721 |                  0.745 |                0.024 |
| baseline__raw     | cb01__age_and_bins | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.757 |               -0.001 |

##### Summary

| compare_group      |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:-------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb01__age_and_bins |                      0.00242857 |                         -0.007 |                          0.014 |                0.00457143 |                   -0.008 |                    0.024 |

</details>


#### cb02__age_imputed_title_and_bins

Experiment testing the effect of imputing raw age with Title before binning it.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.005
  - svc: test_accuracy_mean: 0.004
  - decision_tree: test_accuracy_mean: 0.005
  - extra_trees: test_accuracy_mean: 0.015
    - Secondary gains:
      - test_f1_mean: 0.025


##### Conclusion

Replacing median-imputed Age with Title-imputed Age changes which models benefit from the combined representation but does not fundamentally alter the overall pattern. Combining continuous and ordinal Age remains primarily useful for selected models rather than as a universal strategy.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                    | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb02__age_imputed_title_and_bins | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.716 |                0.003 |
| baseline__raw     | cb02__age_imputed_title_and_bins | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.734 |               -0.008 |
| baseline__raw     | cb02__age_imputed_title_and_bins | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | cb02__age_imputed_title_and_bins | decision_tree |                          0.803 |                        0.808 |                      0.005 |                    0.702 |                  0.711 |                0.009 |
| baseline__raw     | cb02__age_imputed_title_and_bins | random_forest |                          0.822 |                        0.824 |                      0.002 |                    0.744 |                  0.748 |                0.004 |
| baseline__raw     | cb02__age_imputed_title_and_bins | extra_trees   |                          0.804 |                        0.819 |                      0.015 |                    0.721 |                  0.746 |                0.025 |
| baseline__raw     | cb02__age_imputed_title_and_bins | xgb           |                          0.826 |                        0.828 |                      0.002 |                    0.758 |                  0.759 |                0.001 |

##### Summary

| compare_group                    |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb02__age_imputed_title_and_bins |                      0.00414286 |                         -0.004 |                          0.015 |                0.00585714 |                   -0.008 |                    0.025 |

</details>

#### cb03__age_imputed_title_Pclass_and_bins

Experiment akin to Cb02, but imputing age with both title and pclass.

<details>
<summary>Conclusion</summary>


##### Interpretation

- Verdict: mixed
- Recommended for specific models:
  - logreg: test_accuracy_mean: 0.02
    - Secondary gains:
      - test_f1_mean: 0.022
  - svc: test_accuracy_mean: 0.004
  - random_forest: test_accuracy_mean: 0.003
  - extra_trees: test_accuracy_mean: 0.015
    - Secondary gains:
      - test_f1_mean: 0.026


##### Conclusion

Using the most accurate Age imputation together with Age_bin produced the strongest combination results of the three experiments. Logistic Regression showed the largest improvement of the entire Age investigation (+0.020 accuracy), while Extra Trees consistently benefited from the richer representation. Decision Tree, however, gained no advantage, reinforcing that models benefiting from Age_bin alone do not necessarily benefit from combining multiple Age representations.

</details>

<details>
<summary>Experiment details</summary>

##### Comparison vs baseline__raw

| reference_group   | compare_group                           | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | logreg        |                          0.786 |                        0.806 |                      0.02  |                    0.713 |                  0.735 |                0.022 |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | knn           |                          0.809 |                        0.799 |                     -0.01  |                    0.742 |                  0.732 |               -0.01  |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | svc           |                          0.827 |                        0.831 |                      0.004 |                    0.76  |                  0.767 |                0.007 |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | random_forest |                          0.822 |                        0.825 |                      0.003 |                    0.744 |                  0.75  |                0.006 |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | extra_trees   |                          0.804 |                        0.819 |                      0.015 |                    0.721 |                  0.747 |                0.026 |
| baseline__raw     | cb03__age_imputed_title_Pclass_and_bins | xgb           |                          0.826 |                        0.825 |                     -0.001 |                    0.758 |                  0.758 |                0     |

##### Summary

| compare_group                           |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| cb03__age_imputed_title_Pclass_and_bins |                      0.00442857 |                          -0.01 |                           0.02 |                0.00685714 |                    -0.01 |                    0.026 |

</details>

#### Accidental Fare ablation review

<details>
<summary>Ablation details</summary>


A later review found that Fare had been omitted from the original cb01–cb03configurations. The corrected versions are used as the canonical Age
combination experiments.

The original runs were retained as Fare ablations because they revealed a
consistent, model-specific interaction between Fare and the engineered Age
representations.

Although these configurations were created unintentionally, they effectively form a controlled Fare ablation, allowing the marginal contribution of Fare to be evaluated within progressively richer Age feature sets.

**Accuracy effect of removing Fare**

| Model               |     cb01 |     cb02 |     cb03 | Pattern              |
| ------------------- | ---------: | ---------: | ---------: | -------------------- |
| Logistic Regression |     +0.001 |     **+0.004** | **-0.004** | Context-dependent    |
| KNN                 | **-0.007** |     -0.001 | **+0.009** | Reverses in cb03   |
| SVC                 |      0.000 |     +0.001 |      0.000 | Nearly irrelevant    |
| Decision Tree       |     +0.004 |     -0.001 |     +0.001 | Generally improves performance  |
| Random Forest       |     +0.004 |     +0.003 |     +0.003 | Consistently improves performance |
| Extra Trees         | **-0.003** | **-0.001** | **-0.002** | consistently reduces performance |
| XGBoost             |     +0.002 |     +0.004 |     +0.002 | Consistently improves performance |

**F1 effect of removing Fare**

| Model               |     cb01 |     cb02 |     cb03 | Pattern                               |
| ------------------- | ---------: | ---------: | ---------: | ------------------------------------- |
| Logistic Regression |      0.000 |     +0.004 | **-0.005** | Context-dependent                     |
| KNN                 | **-0.016** | **-0.009** |     -0.002 | reduction diminishes                    |
| SVC                 |     -0.002 |      0.000 |     -0.001 | Nearly irrelevant                     |
| Decision Tree       | **+0.017** |     +0.005 | **+0.015** | consistently improves performance                   |
| Random Forest       |     **+0.007** |     +0.003 |     +0.001 | consistently improves performance, but diminishing |
| Extra Trees         | **-0.007** | **-0.005** | **-0.006** | consistently reduces performance                  |
| XGBoost             |     -0.003 |     +0.003 |     -0.002 | Inconsistent and negligible           |

#### Fare ablation conclusion

The accidental omission of Fare from the original cb01–cb03 configurations
created an opportunity to examine Fare's contribution within several
Age-engineered feature sets.

Fare did not produce a consistent improvement across the evaluated models.Instead, its inclusion redistributed performance between models, benefiting some while reducing performance for others. Its inclusion slightly reduced mean accuracy in all three configurations, while its effect on mean F1 was small. However, the model-level results showed several stable patterns.

Extra Trees benefited from Fare in every configuration, while Decision Tree consistently lost F1 and Random Forest experienced small performance losses. SVC was nearly unaffected. KNN initially benefited from Fare, but the benefit decreased as the Age-imputation strategy became more informative and reversed in accuracy under Title-and-Pclass imputation. Logistic Regression benefited only in the most structured Age-imputation configuration.

These results suggest that Fare contains useful information, but much of that information overlaps with the surrounding feature set. Its marginal value is therefore strongly dependent on both the model and the way Age and socioeconomic information are represented.

</details>

#### Overall conclusion

Age proved to be one of the richest features investigated, requiring improvements in both data quality (imputation) and feature representation (continuous versus ordinal).

Improving the imputation method consistently produced modest gains, with Title + Pclass generally outperforming simpler strategies. Converting Age into an ordinal representation (Age_bin) produced the largest improvements for Decision Tree, Random Forest, and Extra Trees, suggesting that these models benefit from threshold-friendly representations aligned with passenger survival behavior.

Combining continuous Age with Age_bin produced a much more model-dependent outcome. Logistic Regression consistently benefited from having access to both representations, culminating in the strongest result when combined with Title + Pclass imputation. Extra Trees also benefited from the richer representation, whereas Decision Tree and Random Forest generally preferred Age_bin alone.

Overall, the investigation demonstrates that there is no universally optimal representation of Age. The best representation depends on both the learning algorithm and the surrounding feature set.

#### Findings

- Better Age imputation consistently improved performance, although the gains remained modest because only around 20% of Age values were missing.

- Age_bin was the single most beneficial Age transformation for tree-based models.

- Combining continuous and ordinal Age representations primarily benefited Logistic Regression and Extra Trees rather than all models.

- Better feature representations do not simply accumulate. Their usefulness depends on the model architecture and the surrounding feature set.

- Feature interactions can change the value of existing features, as illustrated by the accidental Fare ablation.

#### Current recommendation

- Logistic Regression
    - Use Age with Title + Pclass imputation together with Age_bin.
    - This combination produced the strongest Age-related improvement (+0.020 accuracy) and consistently outperformed using either representation alone.

- SVC
    - Use Age with Title imputation together with Age_bin.
    - The combined representation produced small but consistent improvements while remaining stable across all combination experiments.

- Decision Tree
    - Prefer Age_bin alone.
    - Binning Age consistently outperformed continuous Age, while combining both representations provided no additional benefit and often reduced performance.

- Random Forest
    - Prefer Age_bin alone.
    - The binned representation captured nearly all of the available improvement, with combination experiments providing only marginal gains.

- Extra Trees
    - Use Age with Title imputation together with Age_bin.
    - Extra Trees consistently benefited from the combined representation, producing the strongest and most stable improvements across the combination experiments.

- KNN
    - Prefer raw Age.
    - Neither Age binning nor the combined representations consistently improved performance over the original continuous feature.

- XGBoost
    - Prefer raw Age or simple Title + Pclass imputation.
    - More complex Age representations produced little additional benefit, indicating that the model already extracts most of the available information from the continuous feature.

#### Open questions

- Why do linear models appear to benefit from having both continuous and ordinal representations of Age?

- Why does Decision Tree prefer Age_bin alone while Extra Trees consistently benefits from combining Age and Age_bin?

- Would learned bins outperform manually designed ones?

- Would nonlinear transformations (log, spline, quantile) outperform fixed bins?

#### Practical takeaway

The optimal representation of Age depends on the learning algorithm. Tree-based models generally benefit from simpler ordinal representations, whereas some linear and ensemble methods can exploit complementary information from multiple Age representations. Consequently, feature engineering decisions should be validated for each model rather than assumed to generalize across algorithms.

---

### Ticket

#### Hypothesis

Multiple passengers share the same Ticket identifier, suggesting that tickets may represent travel groups rather than individuals. If passengers sharing a ticket remained together during boarding or evacuation, the number of passengers associated with a ticket may contain information beyond the family relationships captured by SibSp and Parch.

Unlike FamilySize, TicketGroupSize depends entirely on the passengers present in the current dataset. If members of the same ticket group are absent from the dataset, the feature underestimates the true group size. This makes it inherently dataset-dependent, and its effect on unseen data is uncertain.

#### Experiments performed:

#### fe09__ticket_group_size

TicketGroupSize counts the number of passengers sharing the same ticket. The objective is to determine whether actual travel groups contain more predictive information than family relationships alone.

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

#### cb08__pclass_sex_features

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

Leaderboard
| experiment                                   | model_name    |   test_accuracy_mean |   test_f1_mean |
|:---------------------------------------------|:--------------|---------------------:|---------------:|
| fe05__title__xgb                             | xgb           |                0.836 |          0.772 |
| fe05__title__svc                             | svc           |                0.834 |          0.771 |
| fe11__age_bin__random_forest                 | random_forest |                0.833 |          0.759 |
| ab02__age_imputed_title_and_bins__svc        | svc           |                0.832 |          0.767 |
| fe04__cabin_features__xgb                    | xgb           |                0.832 |          0.767 |
| ab02__age_imputed_title_and_bins__xgb        | xgb           |                0.832 |          0.762 |
| fe05__title__random_forest                   | random_forest |                0.832 |          0.768 |
| fe09__ticket_group_size__svc                 | svc           |                0.832 |          0.77  |
| cb02__age_imputed_title_and_bins__svc        | svc           |                0.831 |          0.767 |
| cb03__age_imputed_title_Pclass_and_bins__svc | svc           |                0.831 |          0.767 |

Comparison summary:
|    | reference_group   | compare_group             | model_name    |   test_accuracy_mean_delta |   test_f1_mean_delta |
|---:|:------------------|:--------------------------|:--------------|---------------------------:|---------------------:|
|  0 | fe12__sex_pclass  | cb08__pclass_sex_features | logreg        |                      0.007 |                0.02  |
|  1 | fe12__sex_pclass  | cb08__pclass_sex_features | knn           |                      0.003 |               -0.003 |
|  2 | fe12__sex_pclass  | cb08__pclass_sex_features | svc           |                      0.003 |                0.018 |
|  3 | fe12__sex_pclass  | cb08__pclass_sex_features | decision_tree |                      0.009 |                0.022 |
|  4 | fe12__sex_pclass  | cb08__pclass_sex_features | random_forest |                      0.001 |                0.013 |
|  5 | fe12__sex_pclass  | cb08__pclass_sex_features | extra_trees   |                     -0.004 |               -0.004 |
|  6 | fe12__sex_pclass  | cb08__pclass_sex_features | xgb           |                      0.005 |                0.01  |


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



## Lessons learned

- Recovering missing information (cabin, Age Imputation).
- finding hidden information (title).
- Changing representation of existing information (age_bin)

Hypothesis ✓ Confirmed
Hypothesis ✗ Rejected
Hypothesis ~ Partially confirmed