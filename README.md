# Home Expense Predictor

A Python machine learning project that forecasts household spending and classifies expense transactions using supervised learning and SQL integration.

## Overview

The project generates 5 years of synthetic household expense data and applies two supervised machine learning models:

* **Multiple Linear Regression** — predicts the following month's total household spending using the current month's total spending, transaction count, and average transaction amount.
* **Logistic Regression** — classifies individual transactions into seven categories based on transaction amount, day of the week, and month.

The generated data is also stored in a **SQLite database**, where SQL queries are used for expense aggregation and filtering.

## Features

* Generates a reproducible 5-year synthetic household expense dataset.
* Uses Multiple Linear Regression for next-month spending prediction.
* Uses multi-class Logistic Regression to classify transactions into Groceries, Rent, Transport, Entertainment, Utilities, Dining, and Medical.
* Stores expense data in SQLite and performs SQL queries using `GROUP BY`, `SUM`, `WHERE`, and `ORDER BY`.
* Uses Pandas for data cleaning, aggregation, and feature preparation.
* Generates an actual-vs-predicted spending visualization using Matplotlib.

## Technologies

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `SQLite` · `Matplotlib`

## How to Run

pip3 install pandas numpy matplotlib scikit-learn
python3 home_expense_predictor.py

The script generates:

* `home_expenses.csv` — synthetic expense dataset
* `home_expenses.db` — SQLite database
* `regression_chart.png` — actual vs. predicted spending visualization

## Sample Output

MODEL 1: Predicting Next Month's Household Spending
Predicted total household spending for next month: Rs.55,342

MODEL 2: Predicting Expense Category
Category prediction accuracy on test data: 42.6%

## Model Performance

The classification model achieves approximately **42.6% test accuracy** using transaction amount, day of the week, and month as features. The limited accuracy is expected because expense categories can overlap significantly based on these features alone. Additional information such as merchant details or transaction descriptions could improve classification performance.

## Project Structure


home_expense_predictor.py   # Main Python script
home_expenses.csv           # Generated dataset
home_expenses.db            # SQLite database
regression_chart.png        # Actual vs. predicted visualization
README.md                    # Project documentation

