# Titanic Notes

## All Models report

| experiment             | model_name    | status   | test_accuracy   | test_precision   | test_recall   | test_f1       |   fit_time | notes                                                |
|:-----------------------|:--------------|:---------|:----------------|:-----------------|:--------------|:--------------|-----------:|:-----------------------------------------------------|
| baseline_logreg        | logreg        | success  | 0.786 ± 0.018   | 0.736 ± 0.036    | 0.693 ± 0.038 | 0.713 ± 0.026 |      0.028 | Base logistic regression. Baseline for comparison.   |
| baseline_knn           | knn           | success  | 0.809 ± 0.021   | 0.775 ± 0.044    | 0.713 ± 0.038 | 0.742 ± 0.026 |      0.019 | Base kneighbors classifier. Baseline for comparison. |
| baseline_svc           | svc           | success  | 0.827 ± 0.015   | 0.813 ± 0.029    | 0.716 ± 0.045 | 0.76 ± 0.026  |      0.03  | Base SVC. Baseline for comparison.                   |
| baseline_decision_tree | decision_tree | success  | 0.803 ± 0.023   | 0.828 ± 0.03     | 0.617 ± 0.093 | 0.702 ± 0.055 |      0.024 | Base decision tree. Baseline for comparison.         |
| baseline_random_forest | random_forest | success  | 0.822 ± 0.02    | 0.824 ± 0.02     | 0.681 ± 0.062 | 0.744 ± 0.041 |      0.423 | Base random forest. Baseline for comparison.         |
| baseline_extra_trees   | extra_trees   | success  | 0.804 ± 0.012   | 0.791 ± 0.009    | 0.664 ± 0.041 | 0.721 ± 0.025 |      0.309 | Base extra trees. Baseline for comparison.           |
| baseline_xgb           | xgb           | success  | 0.826 ± 0.025   | 0.811 ± 0.033    | 0.713 ± 0.06  | 0.758 ± 0.041 |      0.594 | Base xgb. Baseline for comparison.                   |                 |
| family__logreg        | logreg        | success  | 0.786 ± 0.018   | 0.736 ± 0.036    | 0.693 ± 0.038 | 0.713 ± 0.026 |      0.015 | fe01__family__logreg. fe01 - Feature engineering 01 experiment - family/alone feature        |
| family__knn           | knn           | success  | 0.809 ± 0.021   | 0.775 ± 0.044    | 0.713 ± 0.038 | 0.742 ± 0.026 |      0.016 | fe01__family__knn. fe01 - Feature engineering 01 experiment - family/alone feature           |
| family__svc           | svc           | success  | 0.827 ± 0.015   | 0.813 ± 0.029    | 0.716 ± 0.045 | 0.76 ± 0.026  |      0.025 | fe01__family__svc. fe01 - Feature engineering 01 experiment - family/alone feature           |
| family__decision_tree | decision_tree | success  | 0.803 ± 0.023   | 0.828 ± 0.03     | 0.617 ± 0.093 | 0.702 ± 0.055 |      0.023 | fe01__family__decision_tree. fe01 - Feature engineering 01 experiment - family/alone feature |
| family__random_forest | random_forest | success  | 0.822 ± 0.02    | 0.824 ± 0.02     | 0.681 ± 0.062 | 0.744 ± 0.041 |      0.363 | fe01__family__random_forest. fe01 - Feature engineering 01 experiment - family/alone feature |
| family__extra_trees   | extra_trees   | success  | 0.804 ± 0.012   | 0.791 ± 0.009    | 0.664 ± 0.041 | 0.721 ± 0.025 |      0.284 | fe01__family__extra_trees. fe01 - Feature engineering 01 experiment - family/alone feature   |
| family__xgb           | xgb           | success  | 0.826 ± 0.025   | 0.811 ± 0.033    | 0.713 ± 0.06  | 0.758 ± 0.041 |      0.098 | fe01__family__xgb. fe01 - Feature engineering 01 experiment - family/alone feature           |

## EDA observations

<details>
<summary>Observations</summary>

### Dataframe

891 rows, 12 columns, no duplicate, 708 rows with at least 1 missing column (79.46% of the rwos), total 866 missing values.

Most columns actually have no missing values. Age have about 20% missing values, will need careful imputation for best results. Cabing has 687 (77%) missing values, far too many to impute, far too few for reliable use; not nescessarely useless, but very limited use; could the missingness itself be a signal?

### PassengerId

Id, no real use. drop.

### Survived
Target. 61.6% dead, 38.4% alive. somewhat unbalanced

### Pclass

int, but category like. Options are First, second and third class. Acording to correlation, Pclass has the highest correlation with survived among numerical columns. Maybe easier access to life boat?

### Name

Id like, unique per passenger, little use by itself. No missing values
Family name might have use for groupping families?
Mr, Miss, Mrs, Dr. in the middle. Titles?

### Sex

65% Male from which 81% died, 19% survived
35% female from which 25% died, 75% survived

Female clearly had a higher survival rate, Women likely had priority.

### Age

Mean: 29.7,
25% : 20,
50%: 28,
75%: 38,
Max: 80.
20% missing, 11 outliers, low skew

Despite the presence of babies and elders, the vast majority of age is around 20-40 and there aren't many outliers. Might be worth binning them into groups (kids, teens, adults, elders).

Relative high number of missing's worrying. Will need careful imputation for best results. Name had titles and none of them are missing, might be a good starting point for inputation.

### SibSp and Parch

Number of siblings + spouse and parent + children. Showing whether they were traveling alone or with a family. their correlation with survival is small on it's own, but families are more likely to try to reunite and go for rescue together. Perhaps creating a family feature would give a clearer result?

### Ticket

No missing, and 681 unique. A number of passengers share the same ticket, maybe family? But the tickets themselves varies wildly with no apparent pattern. Don't expect much use out of this.

### Fare

Price paid for the ticket:
No missing value
mean: 32
min: 0
25%: 8
50%: 14.5
75%: 31
max: 512
116 outliers, high skew, heavy tail

Fare prices varied wildly with many outliers, likely going to need proper preprocessing for efficient use, but has a good correlation with survived, certainly worth the effort.

### Cabin

77% missing, too many to impute safely, too few values for reliable use. drop?

They all start with a letter. A section of the ship perhaps? might be of use. Some have multiple cabin... family? might be worth exploring together with the letter. Still, expect limited use, likely save for later or last. Could the missingness itself be a signal?

### Embarked

2 missing values, probably stoways and the passengers with fare 0.
4 uniques: 
Missing: 100% survived
S: 72%, 34% survived
C: 19%, 55% Survived
Q: 8.7%, 39% Survived

C has a considerably higher survival rate, could this be linked to class or position on the ship?

Will need more exploring
</details>

## Feature engineering

<details>
<summary>Modified features</summary>

### Title

<details>
<summary>Reasoning:</summary>

Another way to describe a passenger's socio/economical condition, possibly age too.

Originally composed of:
['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady','Sir', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Mlle', 'Ms', 'Mme']
 
 Mr - > Adult men
 Miss - > unmarred women (varying ages)
 Mrs - > Married women
 Master -> Young man
 Mlle/Ms - > Miss in different languange
 Mrs - > Mrs in different language

 Remaining titles were rare were grouped into a Rare category. (7 Dr, 6 Rev, all others were present only once or twice)

 This groupping was made in a function apart from title to allow the use of these titles for imputation, if nescessary.

 </details>

### Family_size and Alone

<details>
<summary>Reasoning:</summary>

Passengers are more likely to reunite with family before attempting escape, while lone passengers are more likely to head straight to the boats. This is an attempt to see how the presence/number of families affect one's survival rate.

Family size = SibSp + Parch + 1

Alone = 1 if Family size == 1, 0 otherwise

</details>

### Deck and Has_cabin

<details>
<summary>Reasoning:</summary>

First letter from cabin seems to simbolize the deck that cabin is. This was extracted and used, as it may show how easy or hard it was for this passenger to reach the lifeboats. High number of missing values means the impact's likely going to be low.

Decided to create a Has_cabin feature to show whether this passenger's cabing (and deck) was known or not, this missingness might be a signal of itself.

They were made into different functions to allow individual exploration and test.

 </details>

</details>

## Experiments

<details>
<summary>Experiments information</summary>

### Baseline

Baseline for comparison

#### Result

| experiment              | model_name    | status   | test_accuracy   | test_precision   | test_recall   | test_f1       |   fit_time | notes                                                |
|:------------------------|:--------------|:---------|:----------------|:-----------------|:--------------|:--------------|-----------:|:-----------------------------------------------------|
| baseline__logreg        | logreg        | success  | 0.786 ± 0.018   | 0.736 ± 0.036    | 0.693 ± 0.038 | 0.713 ± 0.026 |      0.018 | Base logistic regression. Baseline for comparison.   |
| baseline__knn           | knn           | success  | 0.809 ± 0.021   | 0.775 ± 0.044    | 0.713 ± 0.038 | 0.742 ± 0.026 |      0.017 | Base kneighbors classifier. Baseline for comparison. |
| baseline__svc           | svc           | success  | 0.827 ± 0.015   | 0.813 ± 0.029    | 0.716 ± 0.045 | 0.76 ± 0.026  |      0.028 | Base SVC. Baseline for comparison.                   |
| baseline__decision_tree | decision_tree | success  | 0.803 ± 0.023   | 0.828 ± 0.03     | 0.617 ± 0.093 | 0.702 ± 0.055 |      0.013 | Base decision tree. Baseline for comparison.         |
| baseline__random_forest | random_forest | success  | 0.822 ± 0.02    | 0.824 ± 0.02     | 0.681 ± 0.062 | 0.744 ± 0.041 |      0.339 | Base random forest. Baseline for comparison.         |
| baseline__extra_trees   | extra_trees   | success  | 0.804 ± 0.012   | 0.791 ± 0.009    | 0.664 ± 0.041 | 0.721 ± 0.025 |      0.297 | Base extra trees. Baseline for comparison.           |
| baseline__xgb           | xgb           | success  | 0.826 ± 0.025   | 0.811 ± 0.033    | 0.713 ± 0.06  | 0.758 ± 0.041 |      0.311 | Base xgb. Baseline for comparison.                   |

<details>
<summary>Model details</summary>

#### logreg

Experiment - baseline__logreg:
| Field | Value |
|---|---|
|Train accuracy| 0.803 ± 0.005 |
|Train precision| 0.762 ± 0.011 |
|Train recall| 0.708 ± 0.015 |
|Train f1| 0.734 ± 0.008 |
|Test accuracy| 0.786 ± 0.018 |
|Test precision| 0.736 ± 0.036 |
|Test recall| 0.693 ± 0.038 |
|Test f1| 0.713 ± 0.026 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__logreg',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'logreg',
 'model_params': {'max_iter': 1000, 'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base logistic regression. Baseline for comparison.'}
```

</details>

#### knn

Experiment - baseline__knn:
| Field | Value |
|---|---|
|Train accuracy| 0.861 ± 0.01 |
|Train precision| 0.843 ± 0.013 |
|Train recall| 0.783 ± 0.019 |
|Train f1| 0.812 ± 0.015 |
|Test accuracy| 0.809 ± 0.021 |
|Test precision| 0.775 ± 0.044 |
|Test recall| 0.713 ± 0.038 |
|Test f1| 0.742 ± 0.026 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__knn',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'knn',
 'model_params': {'n_neighbors': 5},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base kneighbors classifier. Baseline for comparison.'}
```

</details>

#### svc

Experiment - baseline__svc:
| Field | Value |
|---|---|
|Train accuracy| 0.834 ± 0.006 |
|Train precision| 0.819 ± 0.01 |
|Train recall| 0.729 ± 0.015 |
|Train f1| 0.771 ± 0.009 |
|Test accuracy| 0.827 ± 0.015 |
|Test precision| 0.813 ± 0.029 |
|Test recall| 0.716 ± 0.045 |
|Test f1| 0.76 ± 0.026 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__svc',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'svc',
 'model_params': {'C': 1.0,
                  'kernel': 'rbf',
                  'gamma': 'scale',
                  'probability': False,
                  'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base SVC. Baseline for comparison.'}
```

</details>

#### decision_tree

Experiment - baseline__decision_tree:
| Field | Value |
|---|---|
|Train accuracy| 0.832 ± 0.005 |
|Train precision| 0.877 ± 0.046 |
|Train recall| 0.659 ± 0.05 |
|Train f1| 0.749 ± 0.017 |
|Test accuracy| 0.803 ± 0.023 |
|Test precision| 0.828 ± 0.03 |
|Test recall| 0.617 ± 0.093 |
|Test f1| 0.702 ± 0.055 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__decision_tree',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'decision_tree',
 'model_params': {'max_depth': 4, 'min_samples_leaf': 5, 'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base decision tree. Baseline for comparison.'}
```

</details>

#### random_forest

Experiment - baseline__random_forest:
| Field | Value |
|---|---|
|Train accuracy| 0.85 ± 0.007 |
|Train precision| 0.873 ± 0.016 |
|Train recall| 0.714 ± 0.015 |
|Train f1| 0.785 ± 0.011 |
|Test accuracy| 0.822 ± 0.02 |
|Test precision| 0.824 ± 0.02 |
|Test recall| 0.681 ± 0.062 |
|Test f1| 0.744 ± 0.041 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__random_forest',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'random_forest',
 'model_params': {'n_estimators': 200,
                  'max_depth': 5,
                  'min_samples_leaf': 3,
                  'random_state': 42,
                  'n_jobs': -1},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base random forest. Baseline for comparison.'}
```

</details>

#### extra_trees

Experiment - baseline__extra_trees:
| Field | Value |
|---|---|
|Train accuracy| 0.813 ± 0.004 |
|Train precision| 0.806 ± 0.012 |
|Train recall| 0.675 ± 0.013 |
|Train f1| 0.735 ± 0.006 |
|Test accuracy| 0.804 ± 0.012 |
|Test precision| 0.791 ± 0.009 |
|Test recall| 0.664 ± 0.041 |
|Test f1| 0.721 ± 0.025 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__extra_trees',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'extra_trees',
 'model_params': {'n_estimators': 200,
                  'max_depth': 5,
                  'min_samples_leaf': 3,
                  'random_state': 42,
                  'n_jobs': -1},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base extra trees. Baseline for comparison.'}
```

</details>

#### xgb

Experiment - baseline__xgb:
| Field | Value |
|---|---|
|Train accuracy| 0.885 ± 0.004 |
|Train precision| 0.895 ± 0.01 |
|Train recall| 0.793 ± 0.006 |
|Train f1| 0.841 ± 0.005 |
|Test accuracy| 0.826 ± 0.025 |
|Test precision| 0.811 ± 0.033 |
|Test recall| 0.713 ± 0.06 |
|Test f1| 0.758 ± 0.041 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'baseline__xgb',
 'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
 'feature_engineering': [],
 'preprocessing': {'numeric_features': ['Age', 'SibSp', 'Parch', 'Fare'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'xgb',
 'model_params': {'n_estimators': 200,
                  'max_depth': 3,
                  'learning_rate': 0.05,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'random_state': 42,
                  'eval_metric': 'logloss'},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Base xgb. Baseline for comparison.'}
```

</details>

#### Conclusion

Baseline working as expected, ready for experiments and comparison

</details>

### fe01

Experiment exploring the effects of replacing SibSp/Parch with FamilySize and IsAlone.

#### Result

| experiment                  | model_name    | status   | test_accuracy   | test_precision   | test_recall   | test_f1       |   fit_time | notes                                                                     |
|:----------------------------|:--------------|:---------|:----------------|:-----------------|:--------------|:--------------|-----------:|:--------------------------------------------------------------------------|
| fe01__family__logreg        | logreg        | success  | 0.795 ± 0.016   | 0.755 ± 0.038    | 0.693 ± 0.038 | 0.721 ± 0.021 |      0.02  | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__knn           | knn           | success  | 0.805 ± 0.027   | 0.767 ± 0.052    | 0.711 ± 0.037 | 0.737 ± 0.032 |      0.016 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__svc           | svc           | success  | 0.826 ± 0.026   | 0.815 ± 0.036    | 0.707 ± 0.058 | 0.756 ± 0.042 |      0.021 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__decision_tree | decision_tree | success  | 0.806 ± 0.021   | 0.82 ± 0.032     | 0.64 ± 0.1    | 0.712 ± 0.056 |      0.014 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__random_forest | random_forest | success  | 0.816 ± 0.015   | 0.817 ± 0.015    | 0.672 ± 0.065 | 0.735 ± 0.038 |      0.345 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__extra_trees   | extra_trees   | success  | 0.806 ± 0.004   | 0.793 ± 0.025    | 0.672 ± 0.045 | 0.726 ± 0.018 |      0.275 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |
| fe01__family__xgb           | xgb           | success  | 0.826 ± 0.026   | 0.817 ± 0.042    | 0.707 ± 0.065 | 0.756 ± 0.044 |      0.368 | Feature engineering 01: replaces SibSp/Parch with FamilySize and IsAlone. |

<details>
<summary>Model details</summary>

#### logreg

Experiment - fe01__family__logreg:
| Field | Value |
|---|---|
|Train accuracy| 0.806 ± 0.005 |
|Train precision| 0.766 ± 0.007 |
|Train recall| 0.711 ± 0.012 |
|Train f1| 0.738 ± 0.008 |
|Test accuracy| 0.795 ± 0.016 |
|Test precision| 0.755 ± 0.038 |
|Test recall| 0.693 ± 0.038 |
|Test f1| 0.721 ± 0.021 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__logreg',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'logreg',
 'model_params': {'max_iter': 1000, 'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### knn

Experiment - fe01__family__knn:
| Field | Value |
|---|---|
|Train accuracy| 0.862 ± 0.008 |
|Train precision| 0.842 ± 0.014 |
|Train recall| 0.789 ± 0.011 |
|Train f1| 0.815 ± 0.01 |
|Test accuracy| 0.805 ± 0.027 |
|Test precision| 0.767 ± 0.052 |
|Test recall| 0.711 ± 0.037 |
|Test f1| 0.737 ± 0.032 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__knn',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'knn',
 'model_params': {'n_neighbors': 5},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### svc

Experiment - fe01__family__svc:
| Field | Value |
|---|---|
|Train accuracy| 0.834 ± 0.008 |
|Train precision| 0.827 ± 0.012 |
|Train recall| 0.716 ± 0.018 |
|Train f1| 0.768 ± 0.012 |
|Test accuracy| 0.826 ± 0.026 |
|Test precision| 0.815 ± 0.036 |
|Test recall| 0.707 ± 0.058 |
|Test f1| 0.756 ± 0.042 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__svc',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'svc',
 'model_params': {'C': 1.0,
                  'kernel': 'rbf',
                  'gamma': 'scale',
                  'probability': False,
                  'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### decision_tree

Experiment - fe01__family__decision_tree:
| Field | Value |
|---|---|
|Train accuracy| 0.832 ± 0.004 |
|Train precision| 0.859 ± 0.046 |
|Train recall| 0.681 ± 0.043 |
|Train f1| 0.757 ± 0.011 |
|Test accuracy| 0.806 ± 0.021 |
|Test precision| 0.82 ± 0.032 |
|Test recall| 0.64 ± 0.1 |
|Test f1| 0.712 ± 0.056 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__decision_tree',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'decision_tree',
 'model_params': {'max_depth': 4, 'min_samples_leaf': 5, 'random_state': 42},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### random_forest

Experiment - fe01__family__random_forest:
| Field | Value |
|---|---|
|Train accuracy| 0.852 ± 0.006 |
|Train precision| 0.871 ± 0.015 |
|Train recall| 0.72 ± 0.007 |
|Train f1| 0.788 ± 0.008 |
|Test accuracy| 0.816 ± 0.015 |
|Test precision| 0.817 ± 0.015 |
|Test recall| 0.672 ± 0.065 |
|Test f1| 0.735 ± 0.038 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__random_forest',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'random_forest',
 'model_params': {'n_estimators': 200,
                  'max_depth': 5,
                  'min_samples_leaf': 3,
                  'random_state': 42,
                  'n_jobs': -1},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### extra_trees

Experiment - fe01__family__extra_trees:
| Field | Value |
|---|---|
|Train accuracy| 0.813 ± 0.004 |
|Train precision| 0.803 ± 0.008 |
|Train recall| 0.681 ± 0.013 |
|Train f1| 0.737 ± 0.008 |
|Test accuracy| 0.806 ± 0.004 |
|Test precision| 0.793 ± 0.025 |
|Test recall| 0.672 ± 0.045 |
|Test f1| 0.726 ± 0.018 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__extra_trees',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'extra_trees',
 'model_params': {'n_estimators': 200,
                  'max_depth': 5,
                  'min_samples_leaf': 3,
                  'random_state': 42,
                  'n_jobs': -1},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### xgb

Experiment - fe01__family__xgb:
| Field | Value |
|---|---|
|Train accuracy| 0.885 ± 0.007 |
|Train precision| 0.895 ± 0.008 |
|Train recall| 0.795 ± 0.012 |
|Train f1| 0.842 ± 0.009 |
|Test accuracy| 0.826 ± 0.026 |
|Test precision| 0.817 ± 0.042 |
|Test recall| 0.707 ± 0.065 |
|Test f1| 0.756 ± 0.044 |

<details>
<summary>Full configuration</summary>

```python
{'name': 'fe01__family__xgb',
 'features': ['Pclass',
              'Sex',
              'Age',
              'Fare',
              'Embarked',
              'FamilySize',
              'IsAlone'],
 'feature_engineering': ['add_family_features'],
 'preprocessing': {'numeric_features': ['Age', 'Fare', 'FamilySize', 'IsAlone'],
                   'onehot_features': ['Sex', 'Embarked'],
                   'ordinal_features': ['Pclass'],
                   'numeric_imputer': 'median',
                   'categorical_imputer': 'most_frequent',
                   'scaler': 'standard'},
 'model_name': 'xgb',
 'model_params': {'n_estimators': 200,
                  'max_depth': 3,
                  'learning_rate': 0.05,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'random_state': 42,
                  'eval_metric': 'logloss'},
 'evaluation': {'method': 'cross_validation',
                'cv': 5,
                'scoring': ['accuracy', 'precision', 'recall', 'f1'],
                'return_train_score': True,
                'n_jobs': -1},
 'notes': 'Feature engineering 01: replaces SibSp/Parch with FamilySize and '
          'IsAlone.'}
```

</details>

#### Conclusion

Replacing SibSp and Parch with FamilySize and IsAlone produced only minor changes across all models.

FamilySize and IsAlone appear to preserve most of the information contained in SibSp and Parch, but do not add substantial new predictive power by themselves.

The largest improvement was observed in Logistic Regression (+0.009 accuracy), suggesting that the engineered features provide a slightly more useful representation for linear models.

</details>

</details>

## Model comparison notes

## Problems encountered

## Future ideas

### Feature engineering ideas
Age bin

Sex + Pclass column?
