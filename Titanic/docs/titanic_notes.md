# Titanic Notes

## EDA observations

### Dataframe

891 rows, 12 columns, no duplicate, 708 ros with at least 1 missing column (79.46% of the rwos), total 866 missing values.

Most columns actually have no missing values. Age have about 20% missing values, will need careful imputation for best results. Cabing has 687 (77%) missing values, far too many to impute, far too few for reliable use; not nescessarely useless, but very limited use; could the missingness itself be a signal?

### PassengerId

Id, no real use. drop.

### Survived
Target. 61.6% dead, 38.4% alive. Slightly unbalanced

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

Despite the presence of babies and elders, the vast majority of age is around 20-40 and there aren't many outliers. Might be worth pinning them into groups (kids, teens, adults, elders).

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

C has a considerably higher survival rate, could this linked to class or position or the ship?

Will need more exploring

## Feature engineering ideas

## Experiments

## Model comparison notes

## Problems encountered

## Future ideas