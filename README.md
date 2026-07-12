# Home Expense Predictor

A Python machine learning pipeline that forecasts household spending and classifies expense transactions by category using synthetic data, supervised learning, and SQL integration.

## Overview

This project simulates **5 years of daily household expense data** and applies two supervised machine learning models:

1. **Multiple Linear Regression** — predicts the following month's total household spending based on the current month's spending patterns.
2. **Logistic Regression** — classifies individual transactions into one of seven expense categories based on transaction amount, day of the week, and month.

The project also integrates **SQLite** for data storage and demonstrates SQL-based aggregation and filtering queries.

## Features

* Generates a reproducible synthetic household expense dataset spanning 5 years, with realistic amount ranges and transaction frequencies across seven expense categories.
* Stores transaction data in a SQLite database and executes SQL queries using `GROUP BY`, `SUM`, `WHERE`, and `ORDER BY`.
* Uses **Multiple Linear Regression** with three current-month aggregate features:

  * Total spending
  * Number of transactions
  * Average transaction amount
* Predicts the following month's total household spending based on current-month spending patterns.
* Uses **multi-class Logistic Regression** to classify transactions into:

  * Groceries
  * Rent
  * Transport
  * Entertainment
  * Utilities
  * Dining
  * Medical
* Performs data preprocessing, cleaning, date handling, and monthly aggregation using Pandas.
* Generates an actual-vs-predicted spending visualization using Matplotlib.
* Evaluates the classification model using test-set accuracy and displays an example prediction.
* Uses fixed random seeds to ensure reproducible results across runs.

## Tools & Libraries

| Tool               | Purpose                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| Python 3           | Core programming language                                                                      |
| Pandas             | Data manipulation, aggregation, and cleaning                                                   |
| NumPy              | Synthetic data generation and reproducibility                                                  |
| scikit-learn       | Multiple Linear Regression, Logistic Regression, train/test splitting, and accuracy evaluation |
| SQLite (`sqlite3`) | Data storage and SQL querying                                                                  |
| Matplotlib         | Data visualization                                                                             |

## Machine Learning Models

### 1. Multiple Linear Regression

The regression model predicts the **following month's total household spending** using three features derived from the current month's transactions:

* Current month's total spending
* Current month's number of transactions
* Current month's average transaction amount

The target variable is created by shifting monthly total spending by one month, allowing the model to learn the relationship between the current month's spending behavior and the following month's total expenditure.

### 2. Logistic Regression

The classification model predicts the expense category of an individual transaction using:

* Transaction amount
* Day of the week
* Month number

The possible output categories are:

`Groceries`, `Rent`, `Transport`, `Entertainment`, `Utilities`, `Dining`, and `Medical`.

## SQL Integration

The generated expense data is stored in a SQLite database named:

```text
home_expenses.db
```

The project performs SQL queries to:

* Calculate total spending for each expense category.
* Count the number of transactions in each category.
* Identify the five largest transactions above ₹3,000.
* Sort query results based on transaction amounts and total spending.

## How to Run

### 1. Install the required libraries

```bash
pip3 install pandas numpy matplotlib scikit-learn
```

### 2. Run the project

```bash
python3 home_expense_predictor.py
```

The script will:

1. Generate a synthetic household expense dataset and save it as `home_expenses.csv`.
2. Store the generated data in `home_expenses.db`.
3. Execute SQL aggregation and filtering queries.
4. Train a Multiple Linear Regression model and compare actual versus predicted next-month spending.
5. Predict the following month's total household spending.
6. Generate and save `regression_chart.png`.
7. Train a multi-class Logistic Regression classifier.
8. Report classification accuracy and display an example transaction prediction.

## Sample Output

```text
MODEL 1: Predicting Next Month's Household Spending
Predicted total household spending for next month: Rs.55,342

MODEL 2: Predicting Expense Category
Category prediction accuracy on test data: 42.6%
```

The output is reproducible across runs because fixed random seeds are used for synthetic data generation and model splitting.

## Model Performance

The classification model achieves approximately **42.6% test accuracy** using only three input features: transaction amount, day of the week, and month.

This level of accuracy is expected because these features alone cannot always clearly distinguish between overlapping expense categories. For example, a ₹700 transaction could plausibly represent Groceries, Dining, Entertainment, Transport, or another category.

Classification performance could potentially be improved by incorporating additional features such as:

* Merchant information
* Transaction descriptions
* Payment methods
* Historical category-specific spending patterns

The project intentionally uses a compact feature set to remain self-contained and reproducible without relying on external datasets.

## Project Structure

```text
home_expense_predictor.py   # Main Python script
home_expenses.csv           # Generated synthetic dataset
home_expenses.db            # SQLite database
regression_chart.png        # Actual vs. predicted spending visualization
README.md                   # Project documentation
```

## Key Concepts Demonstrated

* Supervised machine learning
* Multiple Linear Regression
* Multi-class Logistic Regression
* Feature engineering
* Train/test splitting
* Classification accuracy evaluation
* Data preprocessing and aggregation
* SQL database integration
* Data visualization
* Reproducible machine learning experiments
