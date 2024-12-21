

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


# CODE:
# - must use np.array([...]) or a container class that implements
# the __matul__ dunder to use the '@' operator between two object types.

import numpy as np

print(np.array([1, 2, 3]) @ np.array([4, 5, 6]))
