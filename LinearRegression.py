

# TERMS TO KNOW:
# - feature:
# -- a variable that is used to predict a value
# -- referred to as: "x", independent variable, predictor
# -- example:
# --- house: area, location, number of rooms, etc.
# - label:
# -- the variable that is predicted using our feature variables
# -- referred to as: "y", dependent variable
# -- example
# - training algorithm:
# -- the mathematical algorithm used to let our model learn the patterns and relationships between
# our
# - covariance and variance:
# -- "co" is between two features, ie how strongly are two features correlated,
# "variance" refers to the spread of one feature


# INTRODUCTION:
# - LinearRegression is the "hello, world" for getting into the world of machine learning models
# definitely nowhere near as trivial, but it is definitely the shallow end of the pool. If you can
# remember back to simpleton algebra days when we learned about 'y = mx + b', then you already have
# dipped your feet into prediction ! The roots of LinearRegression date back all the way
# to the 17th century.

# DESCRIPTION:
# - type: supervised learning model
# - goal: calculate the line of "best fit" that can be used to predict values for "y"
# - equation:
# -- y = b1x1 + b2x2 + ... + b0 + e
# -- e = y - b1x1 + b2x2 + ... + b0
# -- bi = model coefficients that are used to weight the importance of each
# variable.
# -- "e" represents some error or random noise that represents the model's lack of perfection
# -- how can we minimize the error, or improve our model's predictive capabilities?
# -- training algorithms:
# --- least-squares: minimizes the sum of the squared errors (residuals)
# or in other words, minimizing the difference between our model's predicted values
# and the actual values represented in as our "label" variable.


# MATH:
# - dot product (matrix multiplication):
# -- takes 2 vectors of equal-length and multiplies each element at an
# index 'i' with other element at index 'i' in the other vector.
# -- does this for every element and returns a single number that represents
# a vector that describes the relationship between the two vectors
# CLOSED-FORM SOLUTION FOR LINEAR REGRESSION:
# - least-squares: B = (X^T @ X)^-1 @ X^T @ y
# -- X: is a feature matrix where columns are your features
# and each row is an individual observation
# -- X^T: is the transpose of the feature matrix preparing us to do
# a '@' in order to find the correlation/interaction between our features
# -- X @ X^T: outputs a matrix that represents how correlated our features are
# to each other. The diagonal elements are how correlated features are with themselves
# and the "off-diagonal" elements are how correlated two features are with each other.
# --- Singular Matrix:
# ----- Where two features are linearly dependent (multi-collinearity), one is
# a combination of the other. Determinant is zero, cannot compute an inverse, so
# the algorithm will fail. Long-story short make sure your features are independent of
# each other. Don't do feature 1: x and feature 2: x + 2
# --- Non-Singular Matrix:
# ----- Opposite of Singular Matrix
# --- if the features are too correlated than regularization can be used to reduce
# penalize large coefficients.
# --- Determinant:
# ----- A value that represents the scalar product of the matrix values. How much information can you gain
# from each of them separately. Marks linear independence, and reliability of coefficients. If the determinant
# is close to zero (near singular) then the coefficients will become exponentially large and unstable when
# we take the inverse of X@X^T

# LEAST-SQUARES CODE:
# - must use np.array([...]) or a container class that implements
# the __matul__ dunder to use the '@' operator between two object types.

import numpy as np

size = (100, 5)

X = np.random.randint(1, 10, size=size)
Y = np.random.randint(2, 4, size=(size[0],)) * X[:, 0] + np.random.randint(5, 7, size=(size[0],)) * X[:, 1]

# IMPORTANT: scale your features because you need the feature interaction vectors be accurate representations
# of the relationships between them. Shouldn't be biased by different units or scales. Will lead to more
# accurate determinants when computing the inverse of the gram matrix, and by consequence better "B"
# model coefficients.
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# the design matrix representing how features relate to each other
# too dependent (multi-collinear), or independent and well-spread?
# goal is non-singularity
# each outputted value is a scalar product representing the similarity between rows
# -- cosine similarity
# when data is scaled by Z-Score then if two feature values in different rows are similar then
# we can either get a negative * negative = positive OR positive * positive = positive, so values that are
# larger (more positive) have low correlation (linearly independent), otherwise if a value has a negative
# Z-Score (lower than the mean) * positive Z-Score then there will be a small value outputted (not similar)
# ...so negative values are good!
gram_mx = X.T @ X

# next take the inverse to flatten the values back into its original form, so we can recover the
# unique B model coefficients. Scales back each feature according to its variance, reflecting the individual
# contributions of each feature.
# -- the diagonal elements of the inverse will represent the adjustment to the variance of a feature
# with regard to the correlations to the other features.
# -- scales by the determinant, which can be found by computing cofactors for each element in A, which
# represents the contribution of each feature to the rest of the matrix minus its row and column
# 1 / det(A) * adj(A) the adjugate matrix which contains the cofactors (determinant * alternating negative pattern)

# -- rank: number of linearly independent rows or columns in a matrix. If a matrix has full rank then
# its rank is equal to its smallest dimension.

inv_gram_mx = np.linalg.inv(gram_mx)


# then once we have a representation of how much unique information each feature contributes then
# we can take the product of that with the interaction between the feature columns and labels
# This is a computationally expensive closed-form solution where the gram matrix is O(p^3)
# must be "well-conditioned" (means non-singular, little multi-collinearity) and with a "p" (feature set)
# < 1000.
result = inv_gram_mx @ (X.T @ Y)

# otherwise if feature set is greater than 1000 and < 10,000 then you can use a "Cholesky" decomposition
#