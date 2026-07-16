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

#### Fare
- investigate effects of scaling and transformations
- Fare seems to be the price paid per ticket. Divide by family and ticket groups to find individual fare

#### Ticket
- Investigate ticket groups

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

Small/negligible impact overall. LogReg improved slightly.
Apparently, most models already manage to extract the information from SibSp and Parch, making this feature engineering mostly unescessary.

One exception being LogReg, which had a small gain in accuracy. Was it unable to make the most of these features on it's own? or it just gained from receiving processed information?

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


#### Findings

- Most models appear capable of extracting the information contained in SibSp and Parch without explicit feature engineering. Explicitly constructing FamilySize and IsAlone therefore provides little additional information to all models but Logreg.

#### Current recommendation
- Logistic Regression
    - FamilySize + IsAlone.

- All other models
    - Raw SibSp and Parch features.

- Revisit after combination experiments are complete.

---

### Cabin

#### Hypothesis

Cabin contains a very large proportion of missing values, while most other features are relatively complete. This suggests that the missingness may not be entirely random, but instead related to some characteristic of the passengers or the data collection process.

One possibility is that passengers whose cabin is unknown are systematically different from those whose cabin is known. For example, because around 70% of passengers without cabin information did not survive, the absence of cabin information may itself contain predictive information.

This hypothesis is tested through the Has_Cabin feature.

Cabin's values always starts with a letter, which likely points to a location within the ship, like it's deck. Perhaps knowing the passenges' cabin position on the ship makes it easier to predict their survival.

This hypothesis is tested throught the Deck feature.

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

Negligible changes on it's own, likely on the level of noise. Surprisingly, LogReg had a small improvement. Going to explore the impact of Deck and Deck + has_cabin to find out whether they complement each other or if Deck's enough on it's own.

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

Testing the impact of the feature deck in the models. Expecting a higher impact then has_cabin, but not by a large margen, given that only 23% of the decks are know.

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

Results were surprising. Meanwhile the mean delta is essentially 0, this combination had greater influence on the models then deck or has_cabin. This suggests that the two features capture different aspects of the underlying information rather than simply encoding the same signal. 

Random forest was the model that suffered the most from it, One possible explanation is that Random Forest already extracts most of the available information from the existing variables, making the additional Cabin-derived features partially redundant.

Logreg continued to gain exactly 0.005 on all three attempts, meaning that it's likely getting the same information from all three approaches, initially telling me that just using one of them would do. But Fe04 also increased it's f1 by 0.011, meaning it's actually generalizing better when using both features together

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


### fe05__title

Title is expected to encode information related to the passenger's gender, age and social status. This experiment evaluates how much additional predictive power this feature provides beyond the variables already present in the baseline model.

<details>
<summary>Conclusion</summary>

#### Interpretation

- Verdict: strong_general
- Recommended for all models
- Mean delta: 0.016

#### Conclusion

Unlike the previous feature engineering experiments, Title produced consistent improvements across every tested model. The feature appears to summarize multiple passenger characteristics—including age, gender and social status—into a single, highly informative variable. Given the magnitude and consistency of the improvements, Title is a strong candidate for inclusion in future feature sets.

The addition of Title also reshaped the leaderboard. Title + XGBoost and Title + SVC moved into first and second place respectively, while Title + Random Forest climbed close behind Cabin Features + XGBoost.

One particularly interesting observation is that Logistic Regression improved by 0.039 accuracy, the largest gain obtained by any model from a single feature engineering experiment so far. Combined with its positive response to previous engineered features, this suggests that Logistic Regression benefits substantially from features that make informative relationships explicit, rather than requiring the model to infer them from the original variables.

</details>

<details>
<summary>Experiment details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe05__title     | logreg        |                          0.786 |                        0.825 |                      0.039 |                    0.713 |                  0.765 |                0.052 |
| baseline__raw     | fe05__title     | knn           |                          0.809 |                        0.822 |                      0.013 |                    0.742 |                  0.76  |                0.018 |
| baseline__raw     | fe05__title     | svc           |                          0.827 |                        0.834 |                      0.007 |                    0.76  |                  0.771 |                0.011 |
| baseline__raw     | fe05__title     | decision_tree |                          0.803 |                        0.823 |                      0.02  |                    0.702 |                  0.756 |                0.054 |
| baseline__raw     | fe05__title     | random_forest |                          0.822 |                        0.832 |                      0.01  |                    0.744 |                  0.768 |                0.024 |
| baseline__raw     | fe05__title     | extra_trees   |                          0.804 |                        0.82  |                      0.016 |                    0.721 |                  0.754 |                0.033 |
| baseline__raw     | fe05__title     | xgb           |                          0.826 |                        0.836 |                      0.01  |                    0.758 |                  0.772 |                0.014 |

#### Summary

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

## Lessons learned

Hypothesis ✓ Confirmed
Hypothesis ✗ Rejected
Hypothesis ~ Partially confirmed