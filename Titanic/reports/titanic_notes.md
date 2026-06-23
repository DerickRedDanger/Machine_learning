# Titanic Notes

## Current leaderboard

| experiment                             | model_name    |   test_accuracy_mean |   test_f1_mean |
|:---------------------------------------|:--------------|---------------------:|---------------:|
| fe05__title__xgb                       | xgb           |                0.836 |          0.772 |
| fe05__title__svc                       | svc           |                0.834 |          0.771 |
| fe04__cabin_features__xgb              | xgb           |                0.832 |          0.767 |
| fe05__title__random_forest             | random_forest |                0.832 |          0.768 |
| fe07__age_imputation_title_pclass__svc | svc           |                0.829 |          0.764 |
| fe06__age_imputation_title__svc        | svc           |                0.829 |          0.764 |
| fe07__age_imputation_title_pclass__xgb | xgb           |                0.828 |          0.761 |
| baseline__raw__svc                     | svc           |                0.827 |          0.76  |
| fe06__age_imputation_title__xgb        | xgb           |                0.826 |          0.756 |
| baseline__raw__xgb                     | xgb           |                0.826 |          0.758 |

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

Id like, unique per passenger, little use by itself. No missing values.

Family name might have use for groupping families?

Mr, Miss, Mrs, Dr. in the middle. Titles?

### Sex

65% Male from which 81% died, 19% survived.
35% female from which 25% died, 75% survived.

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
 
 Mme - > Mrs in different language

 Remaining titles were rare, and thus, were grouped into a Rare category. (7 Dr, 6 Rev, all others were present only once or twice)

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

### Baseline__raw

Baseline with raw features for comparison.

<details>
<summary>Details</summary>

#### Result

| model_name    | accuracy      | f1            |
|:--------------|:--------------|:--------------|
| logreg        | 0.786 ± 0.018 | 0.713 ± 0.026 |
| knn           | 0.809 ± 0.021 | 0.742 ± 0.026 |
| svc           | 0.827 ± 0.015 | 0.76 ± 0.026  |
| decision_tree | 0.803 ± 0.023 | 0.702 ± 0.055 |
| random_forest | 0.822 ± 0.02  | 0.744 ± 0.041 |
| extra_trees   | 0.804 ± 0.012 | 0.721 ± 0.025 |
| xgb           | 0.826 ± 0.025 | 0.758 ± 0.041 |

#### Conclusion

SVC and XGB are the strongest initial baselines. Random Forest is close. Logistic Regression is weaker but useful as a simple reference model.

</details>

### fe01__family

Tests whether FamilySize and IsAlone replace SibSp/Parch effectively.

<details>
<summary>Details</summary>

<details>
<summary>Comparison details</summary>

#### Comparison vs baseline

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe01__family    | logreg        |                          0.786 |                        0.795 |                      0.009 |                    0.713 |                  0.721 |                0.008 |
| baseline__raw     | fe01__family    | knn           |                          0.809 |                        0.805 |                     -0.004 |                    0.742 |                  0.737 |               -0.005 |
| baseline__raw     | fe01__family    | svc           |                          0.827 |                        0.826 |                     -0.001 |                    0.76  |                  0.756 |               -0.004 |
| baseline__raw     | fe01__family    | decision_tree |                          0.803 |                        0.806 |                      0.003 |                    0.702 |                  0.712 |                0.01  |
| baseline__raw     | fe01__family    | random_forest |                          0.822 |                        0.816 |                     -0.006 |                    0.744 |                  0.735 |               -0.009 |
| baseline__raw     | fe01__family    | extra_trees   |                          0.804 |                        0.806 |                      0.002 |                    0.721 |                  0.726 |                0.005 |
| baseline__raw     | fe01__family    | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.756 |               -0.002 |

#### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe01__family    |                     0.000428571 |                         -0.006 |                          0.009 |               0.000428571 |                   -0.009 |                     0.01 |

</details>

#### Conclusion

Small/negligible impact overall. LogReg improved slightly.

</details>

### fe02__has_cabin

Tests if the missingness of the cabins is a signal in itself.

<details>
<summary>Details</summary>

<details>
<summary>Comparison details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe02__has_cabin | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.723 |                0.01  |
| baseline__raw     | fe02__has_cabin | knn           |                          0.809 |                        0.809 |                      0     |                    0.742 |                  0.743 |                0.001 |
| baseline__raw     | fe02__has_cabin | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.756 |               -0.004 |
| baseline__raw     | fe02__has_cabin | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe02__has_cabin | random_forest |                          0.822 |                        0.818 |                     -0.004 |                    0.744 |                  0.739 |               -0.005 |
| baseline__raw     | fe02__has_cabin | extra_trees   |                          0.804 |                        0.807 |                      0.003 |                    0.721 |                  0.729 |                0.008 |
| baseline__raw     | fe02__has_cabin | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.752 |               -0.006 |

#### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe02__has_cabin |                    -0.000714286 |                         -0.004 |                          0.005 |               0.000428571 |                   -0.006 |                     0.01 |

</details>

#### Conclusion

Negligible changes on it's on, likely on the level of noise. Surprisingly LogReg had a small improvement. Going to explore the impact of Deck and Deck + has_cabin to find out whether they complement each other or if Deck's enough on it's own.

</details>

### fe03__deck

Testing the impact of the information from deck in the models. Expecting a higher impact then has_cabin, but not by a large margen, given that only 23% of the decks are know.

<details>
<summary>Details</summary>

<details>
<summary>Comparison details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group   | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe03__deck      | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.722 |                0.009 |
| baseline__raw     | fe03__deck      | knn           |                          0.809 |                        0.818 |                      0.009 |                    0.742 |                  0.752 |                0.01  |
| baseline__raw     | fe03__deck      | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.757 |               -0.003 |
| baseline__raw     | fe03__deck      | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe03__deck      | random_forest |                          0.822 |                        0.813 |                     -0.009 |                    0.744 |                  0.734 |               -0.01  |
| baseline__raw     | fe03__deck      | extra_trees   |                          0.804 |                        0.799 |                     -0.005 |                    0.721 |                  0.719 |               -0.002 |
| baseline__raw     | fe03__deck      | xgb           |                          0.826 |                        0.822 |                     -0.004 |                    0.758 |                  0.751 |               -0.007 |

#### Summary

| compare_group   |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe03__deck      |                     -0.00128571 |                         -0.009 |                          0.009 |              -0.000571429 |                    -0.01 |                     0.01 |

</details>

#### Conclusion

Cabin-derived features showed only minor impact. However, approximately 77% of cabin values are missing, leaving usable cabin information for only about 23% of passengers. This severely limits the feature's potential contribution. The weak results may therefore reflect limited coverage rather than lack of predictive signal. Cabin-based features remain an interesting indicator, but their usefulness is constrained by the large amount of missing data.

</details>

### fe04__cabin_features

Testing the impact of using has_cabin and deck together, to see if this union generates better information or if they are redundant.

<details>
<summary>Details</summary>


<details>
<summary>Comparison details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group        | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe04__cabin_features | logreg        |                          0.786 |                        0.791 |                      0.005 |                    0.713 |                  0.724 |                0.011 |
| baseline__raw     | fe04__cabin_features | knn           |                          0.809 |                        0.817 |                      0.008 |                    0.742 |                  0.752 |                0.01  |
| baseline__raw     | fe04__cabin_features | svc           |                          0.827 |                        0.825 |                     -0.002 |                    0.76  |                  0.757 |               -0.003 |
| baseline__raw     | fe04__cabin_features | decision_tree |                          0.803 |                        0.8   |                     -0.003 |                    0.702 |                  0.701 |               -0.001 |
| baseline__raw     | fe04__cabin_features | random_forest |                          0.822 |                        0.808 |                     -0.014 |                    0.744 |                  0.726 |               -0.018 |
| baseline__raw     | fe04__cabin_features | extra_trees   |                          0.804 |                        0.805 |                      0.001 |                    0.721 |                  0.728 |                0.007 |
| baseline__raw     | fe04__cabin_features | xgb           |                          0.826 |                        0.832 |                      0.006 |                    0.758 |                  0.767 |                0.009 |

#### Summary

| compare_group        |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe04__cabin_features |                     0.000142857 |                         -0.014 |                          0.008 |                0.00214286 |                   -0.018 |                    0.011 |

</details>

#### Conclusion

Results were surprising. Meanwhile the mean delta is essentially 0, this combination had greater influence on the models then deck or has_cabin. Meaning that these features compliment each other, rathen then being redundant. 

Random forest was the model that suffered the most from it, believed to be because it's know for splitting important information across redundant variables.

Meanwhile XGB seemed to better exploit that union, to the point of raising to the top of the current leaderboard with a mean accuracy of 0.832 (+0.006 compared to raw), ahead of Raw SVC by 0.005.

This reinforces my belief that Cabin does contain useful information, but it held back by it's large number of missing values.

</details>

### fe05__title

Title's a feature that possibly bundles Sex + age + social status, this experiment test how much the addition of this feature impact in the model's prediction.

<details>
<summary>Details</summary>

<details>
<summary>Comparison details</summary>

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

#### Conclusion

Unlike the previous feature engineering experiments, Title produced consistent improvements across every tested model. The feature appears to encode information related to age, gender, and social status simultaneously, making it substantially more informative than FamilySize, Has_Cabin, or Deck. Given the magnitude and consistency of the improvements, Title is a strong candidate for inclusion in future feature sets.

It's inclusion led to a change to the leaderboard, leading title__Xgb and title__scv to jump to first/second place respectively with title_random__forest closely behind cabin_feature__xgb which's now third.

One interesting details is that logred has it's accuracy increased by 0.039, the highest increase in any model so far. and it was also the one feature that always increase in accuracy on every feature engineering. Likely meaning that each of them is exposing relationship that the linear model would struggle to discover on it's on.

</details>

### fe06__age_imputation_title

Age have 20% of it's values missing, imputing with median is decent, but not nescesarely the best. This how good imputing age with group by title is, and how this would affect the accuracy of the models.

<details>
<summary>Experiment details</summary>

<details>
<summary>Comparison details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group              | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:---------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe06__age_imputation_title | logreg        |                          0.786 |                        0.787 |                      0.001 |                    0.713 |                  0.712 |               -0.001 |
| baseline__raw     | fe06__age_imputation_title | knn           |                          0.809 |                        0.808 |                     -0.001 |                    0.742 |                  0.739 |               -0.003 |
| baseline__raw     | fe06__age_imputation_title | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe06__age_imputation_title | decision_tree |                          0.803 |                        0.808 |                      0.005 |                    0.702 |                  0.711 |                0.009 |
| baseline__raw     | fe06__age_imputation_title | random_forest |                          0.822 |                        0.825 |                      0.003 |                    0.744 |                  0.749 |                0.005 |
| baseline__raw     | fe06__age_imputation_title | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | fe06__age_imputation_title | xgb           |                          0.826 |                        0.826 |                      0     |                    0.758 |                  0.756 |               -0.002 |

#### Summary

| compare_group              |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:---------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe06__age_imputation_title |                      0.00142857 |                         -0.001 |                          0.005 |                0.00171429 |                   -0.003 |                    0.009 |

</details>

#### Conclusion

Overall, imputing by title lead to gains, even if only slightly.

Wondering if the reason is because only 20% of title is missing, so a better imputation might still not influence much. If age itself is a weak signal, compared to the other features. Or if imputation with median was already pretty effective.

Interesting how Xgb and Extra tree had no change in accuracy and barely any in f1. They didn't care about the changes in age? Or they already got the information they needed from the other features?

</details>


### fe07__age_imputation_title_pclass

Testing the effect of imputing using both Title and Pclass, expecting better results then imputing with title alone.

<details>
<summary>Experiment details</summary>

<details>
<summary>Comparison details</summary>

#### Comparison vs baseline__raw

| reference_group   | compare_group                     | model_name    |   test_accuracy_mean_reference |   test_accuracy_mean_compare |   test_accuracy_mean_delta |   test_f1_mean_reference |   test_f1_mean_compare |   test_f1_mean_delta |
|:------------------|:----------------------------------|:--------------|-------------------------------:|-----------------------------:|---------------------------:|-------------------------:|-----------------------:|---------------------:|
| baseline__raw     | fe07__age_imputation_title_pclass | logreg        |                          0.786 |                        0.799 |                      0.013 |                    0.713 |                  0.725 |                0.012 |
| baseline__raw     | fe07__age_imputation_title_pclass | knn           |                          0.809 |                        0.802 |                     -0.007 |                    0.742 |                  0.737 |               -0.005 |
| baseline__raw     | fe07__age_imputation_title_pclass | svc           |                          0.827 |                        0.829 |                      0.002 |                    0.76  |                  0.764 |                0.004 |
| baseline__raw     | fe07__age_imputation_title_pclass | decision_tree |                          0.803 |                        0.803 |                      0     |                    0.702 |                  0.699 |               -0.003 |
| baseline__raw     | fe07__age_imputation_title_pclass | random_forest |                          0.822 |                        0.826 |                      0.004 |                    0.744 |                  0.75  |                0.006 |
| baseline__raw     | fe07__age_imputation_title_pclass | extra_trees   |                          0.804 |                        0.804 |                      0     |                    0.721 |                  0.721 |                0     |
| baseline__raw     | fe07__age_imputation_title_pclass | xgb           |                          0.826 |                        0.828 |                      0.002 |                    0.758 |                  0.761 |                0.003 |

#### Summary

| compare_group                     |   test_accuracy_mean_delta_mean |   test_accuracy_mean_delta_min |   test_accuracy_mean_delta_max |   test_f1_mean_delta_mean |   test_f1_mean_delta_min |   test_f1_mean_delta_max |
|:----------------------------------|--------------------------------:|-------------------------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|
| fe07__age_imputation_title_pclass |                           0.002 |                         -0.007 |                          0.013 |                0.00242857 |                   -0.005 |                    0.012 |

</details>

#### Conclusion

Imputing Age using Title and Pclass produced slightly better results than imputing by Title alone. 

The strongest improvement was observed in Logistic Regression, suggesting that the more refined age estimates create relationships that are easier for linear models to exploit. 

The overall gains remain small, likely because Age is only missing for approximately 20% of passengers. Nevertheless, the experiment indicates that Pclass provides useful contextual information when estimating missing ages.

</details>

</details>

</details>

## Model comparison notes

## Problems encountered

## Future ideas

### Feature engineering ideas
Age bin

Sex + Pclass column?
