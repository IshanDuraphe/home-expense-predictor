"""
Home Expense Predictor
------------------------
1. Generates a synthetic dataset of daily household expenses.
2. Uses Multiple Linear Regression to predict NEXT MONTH'S total household spending.
3. Uses Logistic Regression to predict the EXPENSE CATEGORY of a transaction
   (Groceries, Rent, Transport, Entertainment, Utilities, Dining, Medical)
   from its amount, day of week, and month.
4. Stores expense data in an SQLite database and performs SQL queries.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ==================== STEP 1: GENERATE SYNTHETIC HOME EXPENSE DATA ====================

CATEGORIES = [
    'Groceries',
    'Rent',
    'Transport',
    'Entertainment',
    'Utilities',
    'Dining',
    'Medical'
]

# Approximate realistic amount ranges per category (in Rs.)
CATEGORY_RANGES = {
    'Groceries': (200, 2000),
    'Rent': (8000, 15000),
    'Transport': (50, 800),
    'Entertainment': (100, 1500),
    'Utilities': (500, 3000),
    'Dining': (150, 1200),
    'Medical': (100, 5000),
}


def generate_home_expenses(n_days=1825, start_date='2021-01-01'):

    dates = pd.date_range(
        start=start_date,
        periods=n_days,
        freq='D'
    )

    records = []

    for date in dates:

        if date.day == 1:

            records.append({
                'Date': date,
                'Category': 'Rent',
                'Amount': np.random.uniform(
                    *CATEGORY_RANGES['Rent']
                )
            })

        n_transactions = np.random.poisson(1.5)

        for _ in range(n_transactions):

            category = np.random.choice(
                [
                    'Groceries',
                    'Transport',
                    'Entertainment',
                    'Utilities',
                    'Dining',
                    'Medical'
                ],
                p=[0.3, 0.2, 0.15, 0.1, 0.2, 0.05]
            )

            low, high = CATEGORY_RANGES[category]

            amount = np.random.uniform(low, high)

            records.append({
                'Date': date,
                'Category': category,
                'Amount': round(amount, 2)
            })

    return pd.DataFrame(records)


expenses = generate_home_expenses()

expenses.to_csv(
    'home_expenses.csv',
    index=False
)

print(
    f"Generated synthetic dataset: "
    f"{len(expenses)} transactions "
    f"-> saved as home_expenses.csv\n"
)

print("Sample of the dataset:")
print(expenses.head(), "\n")


# ==================== STEP 1B: STORE DATA IN SQL DATABASE ====================

conn = sqlite3.connect('home_expenses.db')

expenses.to_sql(
    'expenses',
    conn,
    if_exists='replace',
    index=False
)

print("=" * 50)
print("SQL DEMO: Querying data from home_expenses.db")
print("=" * 50)


# -------------------- SQL QUERY 1 --------------------
# Calculate total spending and transaction count for every category

query1 = """
    SELECT
        Category,
        ROUND(SUM(Amount), 2) AS Total_Spent,
        COUNT(*) AS Num_Transactions
    FROM expenses
    GROUP BY Category
    ORDER BY Total_Spent DESC
"""

category_totals_sql = pd.read_sql_query(
    query1,
    conn
)

print("\nTotal spending by category (via SQL):")
print(category_totals_sql)


# -------------------- SQL QUERY 2 --------------------
# Find the five largest transactions above Rs. 3000

query2 = """
    SELECT
        Date,
        Category,
        Amount
    FROM expenses
    WHERE Amount > 3000
    ORDER BY Amount DESC
    LIMIT 5
"""

large_expenses_sql = pd.read_sql_query(
    query2,
    conn
)

print("\nTop 5 largest transactions above Rs.3000 (via SQL):")
print(large_expenses_sql, "\n")

conn.close()


# ==================== STEP 2: CLEAN DATA ====================

# Handle missing amounts
expenses['Amount'] = expenses['Amount'].fillna(
    expenses['Amount'].mean()
)

expenses['Category'] = (
    expenses['Category']
    .fillna('Uncategorized')
    .astype(str)
    .str.strip()
    .str.title()
)

expenses['Date'] = pd.to_datetime(
    expenses['Date']
)


# ==================== MODEL 1: MULTIPLE LINEAR REGRESSION ====================

"""
Goal:
Predict NEXT MONTH'S total household spending using information
from the CURRENT MONTH.

Input features:
1. Current month's total spending
2. Current month's number of transactions
3. Current month's average transaction amount

Target:
Next month's total household spending
"""

print("=" * 50)
print("MODEL 1: Predicting Next Month's Household Spending")
print("=" * 50)


monthly_data = expenses.groupby(
    expenses['Date'].dt.to_period('M')
).agg(
    Total_Spending=('Amount', 'sum'),
    Num_Transactions=('Amount', 'count'),
    Avg_Transaction=('Amount', 'mean')
).reset_index()


monthly_data['Next_Month_Spending'] = (
    monthly_data['Total_Spending'].shift(-1)
)


model_data = monthly_data.dropna().copy()


X = model_data[[
    'Total_Spending',
    'Num_Transactions',
    'Avg_Transaction'
]]


y = model_data['Next_Month_Spending']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



reg_model = LinearRegression()

reg_model.fit(
    X_train,
    y_train
)

y_pred_reg = reg_model.predict(X_test)


# Display actual vs predicted spending
print("\nActual vs Predicted Next-Month Spending:")

for actual, predicted in zip(
    y_test.values,
    y_pred_reg
):

    print(
        f"Actual: Rs.{actual:,.0f} | "
        f"Predicted: Rs.{predicted:,.0f}"
    )


latest_features = monthly_data[[
    'Total_Spending',
    'Num_Transactions',
    'Avg_Transaction'
]].iloc[[-1]]


predicted_amount = reg_model.predict(
    latest_features
)[0]


print(
    f"\nPredicted total household spending "
    f"for next month: Rs.{predicted_amount:,.0f}\n"
)


# -------------------- REGRESSION VISUALIZATION --------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred_reg
)

plt.xlabel(
    "Actual Next-Month Spending (Rs.)"
)

plt.ylabel(
    "Predicted Next-Month Spending (Rs.)"
)

plt.title(
    "Actual vs Predicted Household Spending"
)

plt.tight_layout()

plt.savefig(
    'regression_chart.png'
)

plt.close()

print(
    "Saved chart: regression_chart.png\n"
)


# ==================== MODEL 2: LOGISTIC REGRESSION ====================

"""
Goal:
Predict the expense category of a transaction.

Input features:
1. Transaction amount
2. Day of the week
3. Month number

Possible output categories:
Groceries, Rent, Transport, Entertainment,
Utilities, Dining, Medical
"""

print("=" * 50)
print("MODEL 2: Predicting Expense Category")
print("=" * 50)

expenses['DayOfWeek'] = (
    expenses['Date'].dt.dayofweek
)

expenses['MonthNum'] = (
    expenses['Date'].dt.month
)


# Input features
X_cls = expenses[[
    'Amount',
    'DayOfWeek',
    'MonthNum'
]]


# Target category
y_cls = expenses['Category']


# Split into training and testing sets
X_train_c, X_test_c, y_train_c, y_test_c = (
    train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=42
    )
)

clf = LogisticRegression(
    max_iter=1000,
    random_state=42
)


clf.fit(
    X_train_c,
    y_train_c
)


y_pred_cls = clf.predict(
    X_test_c
)

accuracy = accuracy_score(
    y_test_c,
    y_pred_cls
)


print(
    f"Category prediction accuracy "
    f"on test data: {accuracy:.1%}\n"
)


# -------------------- EXAMPLE CLASSIFICATION --------------------

sample = X_test_c.iloc[[0]]

predicted_category = clf.predict(
    sample
)[0]

actual_category = y_test_c.iloc[0]


print(
    f"Example input: "
    f"{sample.to_dict('records')[0]}"
)

print(
    f"Predicted category: {predicted_category} | "
    f"Actual category: {actual_category}\n"
)


print("Done.")
