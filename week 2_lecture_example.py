"""
Week 2
Combined Parts 1-6 Exercise
File: code_p1_p6.py

PURPOSE
-------
This file combines the concepts demonstrated in Sentdex Parts 1-6 into
one working exercise using current Python syntax.

Students do NOT need to recreate separate code_p1.py through code_p6.py
files for this portion of the assignment.

STUDENT TASK
------------
1. Download this file.
2. Open it in your Python coding environment.
3. Run the entire file.
4. Confirm that it runs without errors.
5. Submit:
   - this Python file,
   - the console output showing successful execution,
   - the output artifacts created by the program:
       marketing_sales_model.pkl
       sales_forecast.png

The goal is to demonstrate the same sequence introduced in Parts 1-6:
data -> features/labels -> training/testing -> prediction -> saving a model.
"""

# ============================================================
# PART 1 - INTRODUCTION / SETUP
# ============================================================
# Sentdex Part 1 introduces the basic machine-learning workflow and
# the Python libraries used throughout the series.
#
# For this exercise we use:
#   pandas       -> organize and inspect tabular data
#   scikit-learn -> preprocessing, train/test split, and linear regression
#   matplotlib   -> visualize actual and forecasted values
#   pickle       -> save and reload a trained Python model
#
# The important idea from Part 1 is that machine learning is not just
# "calling an algorithm." We first need usable data, then we prepare it,
# train a model, test it, and finally use it to make predictions.

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


print("=" * 70)
print("AI 600 WEEK 2 - SENTDEX PARTS 1-6 COMBINED EXERCISE")
print("=" * 70)


# ============================================================
# PART 2 - REGRESSION: INTRODUCTION AND DATA
# ============================================================
# In the original Sentdex tutorial, Part 2 loads historical stock-market
# data from Quandl. That original online workflow is no longer reliable
# for this course exercise.
#
# We are keeping the SAME learning objective:
#   start with a real-looking table of numeric data and prepare it for
#   regression.
#
# Here the example is monthly marketing/sales activity.
#
# Each row represents one month.
# The columns describe activity that may help explain or predict Sales.

data = {
    "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "AdSpend": [10, 12, 13, 15, 18, 20, 22, 24, 26, 29, 31, 34],
    "WebsiteVisits": [220, 240, 260, 275, 310, 330, 360, 390, 410, 450, 470, 505],
    "Leads": [18, 20, 21, 24, 28, 31, 34, 37, 39, 44, 46, 50],
    "Sales": [52, 56, 59, 63, 70, 74, 79, 84, 88, 94, 99, 106]
}

df = pd.DataFrame(data)

print("\nPART 2 - ORIGINAL DATA")
print(df)

# Basic inspection is an important first step before modeling.
print("\nFirst rows:")
print(df.head())

print("\nRows and columns:", df.shape)

print("\nSummary statistics:")
print(df.describe())


# ============================================================
# PART 3 - FEATURES AND LABELS
# ============================================================
# Sentdex Part 3 introduces one of the most important ideas in
# supervised machine learning:
#
#   FEATURES = the information given to the model
#   LABEL    = the value we want the model to learn/predict
#
# We first create two additional features from the existing columns.
#
# ConversionRate:
#   What percentage of website visits became leads?
#
# SalesPerLead:
#   How many sales are associated with each lead?
#
# Creating useful variables from existing data is often called
# feature engineering.

df["ConversionRate"] = (df["Leads"] / df["WebsiteVisits"]) * 100
df["SalesPerLead"] = df["Sales"] / df["Leads"]

print("\nPART 3 - ENGINEERED FEATURES")
print(df[["Month", "AdSpend", "ConversionRate", "SalesPerLead", "Sales"]])

# We will forecast two periods ahead.
#
# shift(-2) moves future Sales values upward by two rows.
# That makes the current month's features line up with Sales two
# months later.
#
# This is the same basic idea used in the Sentdex forecasting example:
# use known information to learn a relationship with a future value.

forecast_out = 2
df["FutureSales"] = df["Sales"].shift(-forecast_out)

print("\nCurrent Sales aligned with the future label:")
print(df[["Month", "Sales", "FutureSales"]])

# These are the FEATURES supplied to the model.
feature_cols = [
    "AdSpend",
    "WebsiteVisits",
    "Leads",
    "ConversionRate",
    "SalesPerLead"
]

X_all = df[feature_cols].values

# The final two rows do not yet have FutureSales labels.
# Those are exactly the rows we want the model to forecast later.
X_future = X_all[-forecast_out:]

# Drop rows whose future label is unknown.
known_df = df.dropna()

X_known = known_df[feature_cols].values

# y is the LABEL / target the model is learning to predict.
y_known = known_df["FutureSales"].values

print("\nRows reserved for future forecasting:")
print(X_future)

print("\nNumber of rows with known labels:", len(X_known))


# ============================================================
# PART 4 - TRAINING AND TESTING
# ============================================================
# Before fitting the regression model, we scale the features.
#
# StandardScaler transforms variables so they are on comparable scales.
# The scaler is FIT only on the known/training-style data and then used
# to TRANSFORM both the known rows and future rows.
#
# This preserves the same preprocessing idea shown in the Sentdex series.

scaler = StandardScaler()

X_known_scaled = scaler.fit_transform(X_known)
X_future_scaled = scaler.transform(X_future)

print("\nPART 4 - SCALED DATA")
print("First scaled known row:")
print(X_known_scaled[0])

print("\nScaled future rows:")
print(X_future_scaled)

# We split known observations into:
#
#   TRAINING DATA -> used to fit the model
#   TESTING DATA  -> held back so we can evaluate the fitted model
#
# random_state=42 makes this split reproducible so student output is
# consistent each time the file is run.

X_train, X_test, y_train, y_test = train_test_split(
    X_known_scaled,
    y_known,
    test_size=0.25,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# PART 5 - FORECASTING AND PREDICTING
# ============================================================
# LinearRegression finds the linear relationship that best connects
# our features (X) to the label (y).
#
# fit() = learn from the training data
# score() = evaluate the model on data it did not train on
# predict() = generate estimated outcomes

model = LinearRegression()

model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)

print("\nPART 5 - LINEAR REGRESSION MODEL")
print("Test R-squared:", round(r2, 4))
print("Intercept:", round(model.intercept_, 4))
print("Coefficients:", model.coef_)

# Compare predictions against the known test values.
test_predictions = model.predict(X_test)

print("\nTest-set predictions:")
for actual, predicted in zip(y_test, test_predictions):
    print("Actual:", actual, "Predicted:", round(predicted, 2))

# Now use the model on the two rows that were held aside earlier.
future_predictions = model.predict(X_future_scaled)

print("\nFuture sales predictions:")
for i, prediction in enumerate(future_predictions, start=1):
    print(f"Forecast period {i}: {prediction:.2f}")

# Create a visual artifact showing the original Sales series and
# the two forecast values.
#
# The forecast points are placed after Month 12 so the chart makes
# the distinction between historical data and predicted data clear.

future_months = [13, 14]

plt.figure(figsize=(8, 5))
plt.plot(df["Month"], df["Sales"], marker="o", label="Actual Sales")
plt.plot(future_months, future_predictions, marker="o", linestyle="--",
         label="Forecast Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("AI 600 Week 2 - Linear Regression Forecast")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png", dpi=150)
plt.close()

print("\nCreated artifact: sales_forecast.png")


# ============================================================
# PART 6 - PICKLING AND SAVING THE MODEL
# ============================================================
# Training a model can take time. Once a model has been trained,
# Python's pickle module can save the model as a file.
#
# This is still a valid Python technique and preserves the purpose
# of Sentdex Part 6.
#
# "wb" = write binary
# pickle.dump() writes the trained model to disk.

with open("marketing_sales_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Created artifact: marketing_sales_model.pkl")

# We now reload the saved model.
#
# "rb" = read binary
# pickle.load() reconstructs the saved Python object.

with open("marketing_sales_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# If the reloaded model generates the same forecasts, we have evidence
# that the save/reload process worked correctly.

loaded_predictions = loaded_model.predict(X_future_scaled)

print("\nPART 6 - PREDICTIONS FROM RELOADED MODEL")
for i, prediction in enumerate(loaded_predictions, start=1):
    print(f"Reloaded forecast period {i}: {prediction:.2f}")


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================
# Students should see this message at the bottom of the console.
# If it appears and the two artifact files exist, the exercise ran
# successfully.

print("\n" + "=" * 70)
print("SUCCESS: Parts 1-6 completed without errors.")
print("Submit your console output and the generated artifacts.")
print("=" * 70)



