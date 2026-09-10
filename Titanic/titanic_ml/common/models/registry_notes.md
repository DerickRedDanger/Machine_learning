## Model Name

### What it is
Short explanation.

### Defining features
- ...

### Strengths
- ...

### Weaknesses
- ...

### Good use cases
- ...

### Notes for Titanic
- ...

LogisticRegression
What it is

A linear classification model that estimates the probability of belonging to a class.

How it thinks
Survival Score =
a + b1*Age + b2*Fare + b3*Sex + ...

Then converts the score into a probability.

Strengths
Fast
Highly interpretable
Works well with small datasets
Good baseline model
Weaknesses
Assumes roughly linear relationships
Doesn't naturally capture feature interactions
Needs scaling for best results
Titanic Notes

Usually performs surprisingly well because:

Sex is very predictive
Passenger class is highly predictive
Relationships are relatively simple
KNeighborsClassifier (KNN)
What it is

Predicts based on the labels of nearby training examples.

How it thinks
New Passenger
      |
      V
Find 5 closest passengers
      |
      V
Majority vote
Strengths
Very intuitive
No training phase
Can learn complex boundaries
Weaknesses
Slow prediction
Sensitive to feature scaling
Struggles with many features
Titanic Notes

Always scale:

StandardScaler()

before KNN.

Great for learning how scaling affects models.

SVC (Support Vector Classifier)
What it is

A Support Vector Machine classifier.

How it thinks

Finds the boundary that maximizes separation between classes.

Dead  |-----Margin-----| Survived
Strengths
Powerful on small datasets
Handles nonlinear relationships with kernels
Often achieves high accuracy
Weaknesses
Slower on large datasets
Harder to interpret
Sensitive to scaling
Titanic Notes

Often among the strongest non-ensemble models.

Use:

StandardScaler()

before training.

DecisionTreeClassifier
What it is

A tree of if/then decisions.

How it thinks
Sex == Female?
├── Yes -> Survive
└── No
     └── Age < 10?
           ├── Yes -> Survive
           └── No -> Dead
Strengths
Very interpretable
Captures nonlinear relationships
No scaling required
Weaknesses
Overfits easily
Unstable
High variance
Titanic Notes

Excellent for understanding:

Feature importance
Splitting criteria
Overfitting
RandomForestClassifier
What it is

Many decision trees trained on random subsets of data.

How it thinks
Tree 1 -> Survive
Tree 2 -> Dead
Tree 3 -> Survive
...
Majority Vote -> Survive
Strengths
Strong performance
Robust
Handles nonlinear relationships
Little preprocessing
Weaknesses
Less interpretable
Larger memory footprint
Titanic Notes

Usually one of the best "plug-and-play" models.

Feature engineering often helps less than with linear models because forests discover interactions automatically.

ExtraTreesClassifier
What it is

A more randomized version of Random Forest.

How it thinks

Instead of finding the best split:

Age < 32.7 ?

it may choose a random threshold:

Age < 28.4 ?

and build many such trees.

Strengths
Fast
Often performs as well as Random Forest
Lower variance
Strong default model
Weaknesses
Slightly less interpretable
Occasionally underfits compared to RF
Titanic Notes

A fantastic benchmark model.

Many people find:

Extra Trees ≈ Random Forest

with slightly faster training.

XGBClassifier (XGBoost)
What it is

A gradient boosting model.

Rather than building independent trees:

Tree 1
Tree 2 fixes Tree 1 errors
Tree 3 fixes Tree 2 errors
Tree 4 fixes remaining errors
How it thinks

Sequentially improves mistakes.

Weak Tree
  ↓
Correct Errors
  ↓
Correct Remaining Errors
  ↓
Repeat
Strengths
State-of-the-art on tabular data
Captures complex interactions
Excellent predictive power
Weaknesses
More hyperparameters
Easier to overfit
Slower tuning process
Titanic Notes

Often achieves the highest Kaggle Titanic scores among these models.

Feature engineering still matters because boosting can exploit new features extremely well.

