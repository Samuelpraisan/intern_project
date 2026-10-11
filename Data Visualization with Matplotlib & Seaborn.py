
# Data Visualization using Matplotlib and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Set plot style and color palette
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Handle missing values for selected columns
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# 1. Histogram - Age distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Age", bins=20, kde=True, color="skyblue")
plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Bar chart - Survival count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived", hue="Survived",
              palette=["tomato", "seagreen"], legend=False)
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# 3. Scatter plot - Age vs Fare
plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="Age", y="Fare",
                hue="Survived", palette="viridis", alpha=0.7)
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

# 4. Box plot - Fare by passenger class
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass",
            palette="pastel", legend=False)
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

# 5. Correlation matrix
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

print("Correlation Matrix:")
print(corr.round(2))

# 6. Heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Titanic Dataset")
plt.tight_layout()
plt.show()
