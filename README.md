# Experiment 3: Logistic Regression for Classification

## Aim

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    \[ `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex} \]

    where

    \[ z = `\beta`{=tex}\_0 + `\beta`{=tex}\_1 x_1 + `\cdots `{=tex}+
    `\beta`{=tex}\_p x_p \]

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**




## Program
## Input 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
Y = iris.target
names = iris.target_names

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)

# Fit LDA
lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train, Y_train)
X_test_lda = lda.transform(X_test)

# Predict on test data
Y_pred = lda.predict(X_test)

# Evaluate accuracy
acc=accuracy_score(Y_test, Y_pred)
print("LDA Test Accuracy:", acc)

# LDA projection for visualization
X_lda = lda.fit_transform(X, Y)

# Plot
plt.figure(figsize=(10, 5))

for label, marker, color in zip(
    [0, 1, 2],
    ('o', '^', 's'),
    ('red', 'green', 'blue')):

  plt.scatter(
      X_lda[Y == label, 0],
      X_lda[Y == label, 1],
      marker=marker,
      color=color,
      label=names[label]
  )

plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA Projection (Iris Dataset)')
plt.legend()
plt.grid(True)
plt.show()
```
## Output 
```
LDA Test Accuracy: 0.9777777777777777
```

<img width="844" height="470" alt="image" src="https://github.com/user-attachments/assets/a438213c-585a-402d-a280-35479a807b37" />


## Result
Thus, the Linear Discriminant Analysis (LDA) classification model was successfully implemented using the Iris dataset, and an accuracy of 97.78% was achieved.

