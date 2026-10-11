
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Superstore dataset
url = "https://raw.githubusercontent.com/Technology-Hatchery/Rcode/master/Data/Sample%20-%20Superstore%20Sales%20%28Excel%29.csv"

df = pd.read_csv(url, encoding="latin1")

# 2. Display basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumn Names:")
print(df.columns.tolist())

# 3. Clean column names
df.columns = df.columns.str.strip()

# 4. Convert Order Date into datetime format
if "Order Date" not in df.columns:
    raise ValueError(
        "Order Date column not found. Check the column names printed above."
    )

df["Order Date"] = pd.to_datetime(
    df["Order Date"], errors="coerce"
)

# Convert Sales and Profit to numeric values
for col in ["Sales", "Profit"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing dates
df = df.dropna(subset=["Order Date"])

# 5. Monthly Sales Trend
if "Sales" in df.columns:
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

    monthly_sales = df.groupby("Month")["Sales"].sum()

    print("\nMonthly Sales:")
    print(monthly_sales)

    monthly_sales.plot(kind="line", marker="o", figsize=(10, 5))
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6. Top 10 Customers
if "Customer Name" in df.columns and "Sales" in df.columns:
    top_customers = (
        df.groupby("Customer Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 Customers by Sales:")
    print(top_customers)

    top_customers.sort_values().plot(
        kind="barh", figsize=(9, 5)
    )
    plt.title("Top 10 Customers by Sales")
    plt.xlabel("Total Sales")
    plt.tight_layout()
    plt.show()

# 7. Sales by Category
if "Category" in df.columns and "Sales" in df.columns:
    category_sales = df.groupby("Category")["Sales"].sum()

    print("\nSales by Category:")
    print(category_sales)

    category_sales.plot(kind="bar", figsize=(7, 4))
    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

# 8. Sales by Region
if "Region" in df.columns and "Sales" in df.columns:
    region_sales = df.groupby("Region")["Sales"].sum()

    print("\nSales by Region:")
    print(region_sales)

    sns.barplot(x=region_sales.index, y=region_sales.values)
    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

# 9. Customer Inactivity Analysis
# This estimates inactivity; it does not confirm customer churn.
if "Customer Name" in df.columns:
    reference_date = df["Order Date"].max()

    customer_activity = df.groupby("Customer Name").agg(
        Last_Order=("Order Date", "max")
    )

    customer_activity["Days_Inactive"] = (
        reference_date - customer_activity["Last_Order"]
    ).dt.days

    inactive_customers = customer_activity[
        customer_activity["Days_Inactive"] > 90
    ]

    print("\nCustomers Inactive for More Than 90 Days:")
    print(inactive_customers)

    print(
        "\nNumber of inactive customers:",
        len(inactive_customers)
    )

print("\nSales Trend Analysis Completed!")
