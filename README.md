# Week 2 Lecture Example

This repository contains the Week 2 lecture example for **Machine Learning Basics**.

The exercise modernizes and combines the concepts introduced in **Sentdex Machine Learning with Python Parts 1–6** into one working Python program using current Python and scikit-learn syntax.

## Week 2 Focus

This example introduces the basic machine-learning workflow through **Linear Regression**.

Students will work through the following sequence:

1. Import the required Python libraries.
2. Create and inspect a dataset.
3. Identify features and labels.
4. Perform basic feature engineering.
5. Prepare data for forecasting.
6. Scale the feature variables.
7. Split the data into training and testing sets.
8. Train a Linear Regression model.
9. Evaluate the model using R-squared.
10. Generate future predictions.
11. Create a forecast visualization.
12. Save and reload a trained model using Python's `pickle` module.

The overall workflow is:

**Data → Features/Labels → Training/Testing → Model → Prediction → Saved Model**

## Why This Version Is Different

The original Sentdex tutorial uses historical stock-market data obtained through Quandl and older versions of Python and scikit-learn.

Some portions of that original workflow no longer run reliably with modern Python environments.

This lecture example preserves the important machine-learning concepts from Parts 1–6 while using:

* Current Python syntax
* Current scikit-learn methods
* A self-contained marketing and sales dataset
* No external API or dataset dependency

This allows us to focus on the machine-learning concepts rather than troubleshooting outdated libraries or unavailable data services.

## Required Python Packages

The example uses:

* `pandas`
* `scikit-learn`
* `matplotlib`

Python's `pickle` module is included with Python and does not need to be installed separately.

If needed, install the required packages from the VS Code terminal:

```bash
pip install pandas scikit-learn matplotlib
```

## Running the Lecture Example

Download or clone this repository and open:

```text
week_2_lecture_example.py
```

in Visual Studio Code.

Run the entire Python file.

The program will display each stage of the machine-learning workflow in the console.

At the end, you should see:

```text
SUCCESS: Parts 1-6 completed without errors.
```

## Files Created by the Program

Running the program creates two additional files:

```text
marketing_sales_model.pkl
sales_forecast.png
```

### `sales_forecast.png`

A visualization showing the original monthly sales observations and the two periods forecast by the Linear Regression model.

### `marketing_sales_model.pkl`

A saved copy of the trained Linear Regression model created using Python's `pickle` module.

The program reloads the model and generates predictions again to demonstrate that a trained machine-learning model can be saved and reused.

## Connection to Sentdex Parts 1–6

### Part 1 – Introduction and Setup

Introduces the Python libraries and basic machine-learning workflow.

### Part 2 – Regression Data

Creates and examines a structured dataset that can be used for regression.

### Part 3 – Features and Labels

Introduces supervised-learning terminology, feature engineering, and the distinction between predictor variables and the value being predicted.

### Part 4 – Training and Testing

Scales the feature variables and divides the known observations into training and testing datasets.

### Part 5 – Forecasting and Prediction

Fits a Linear Regression model, evaluates it using R-squared, makes predictions, and generates a forecast visualization.

### Part 6 – Saving the Model

Uses Python's `pickle` module to save the trained model, reload it, and verify that it can still generate predictions.

## Important

This exercise is intended to demonstrate the **concepts** taught in Sentdex Parts 1–6 using a modern Python environment.

Students do **not** need to recreate separate `code_p1.py` through `code_p6.py` files for this portion of the assignment.

---

**Machine Learning Basics**
**Week 2: Linear Regression**

