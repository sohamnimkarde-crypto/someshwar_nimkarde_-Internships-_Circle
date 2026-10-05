# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA) ON TITANIC DATASET
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 2. Load Dataset
# ============================================================

# If Titanic-Dataset.csv is inside data folder
df = pd.read_csv("data/Titanic-Dataset.csv")

print("=" * 60)
print("TITANIC DATASET")
print("=" * 60)


# ============================================================
# 3. Display First 5 Rows
# ============================================================

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 4. Dataset Shape
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ============================================================
# 5. Column Names
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 6. Dataset Information
# ============================================================

print("\nDataset Information:")
df.info()


# ============================================================
# 7. Statistical Summary
# ============================================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 8. Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


print("\nMissing Value Percentage:")

missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage.sort_values(ascending=False))


# ============================================================
# 9. Duplicate Rows
# ============================================================

print("\nDuplicate Rows:")

duplicate_count = df.duplicated().sum()

print(duplicate_count)


# ============================================================
# 10. Data Cleaning
# ============================================================

# Fill missing Age values with median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())


# Fill missing Embarked values with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )


# Drop Cabin column if it exists
if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])


print("\nMissing Values After Cleaning:")

print(df.isnull().sum())


# ============================================================
# 11. Survival Count
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 12. Survival by Gender
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 13. Survival by Passenger Class
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 14. Age Distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 15. Age vs Survival
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Survived",
    y="Age"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Age")

plt.tight_layout()
plt.show()


# ============================================================
# 16. Fare Distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 17. Family Size
# ============================================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nFamily Size:")
print(
    df[
        ["SibSp", "Parch", "FamilySize"]
    ].head()
)


# ============================================================
# 18. Survival by Family Size
# ============================================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="FamilySize",
    hue="Survived"
)

plt.title("Survival by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 19. Survival Rate by Gender
# ============================================================

survival_gender = (
    df.groupby("Sex")["Survived"].mean() * 100
)

print("\nSurvival Rate by Gender:")

print(
    survival_gender.round(2)
)


plt.figure(figsize=(7, 5))

survival_gender.plot(
    kind="bar"
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 20. Survival Rate by Passenger Class
# ============================================================

survival_class = (
    df.groupby("Pclass")["Survived"].mean() * 100
)

print("\nSurvival Rate by Passenger Class:")

print(
    survival_class.round(2)
)


plt.figure(figsize=(7, 5))

survival_class.plot(
    kind="bar"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 21. Survival by Embarked Port
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Embarked",
    hue="Survived"
)

plt.title("Survival by Embarked Port")
plt.xlabel("Embarked Port")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 22. Correlation Heatmap
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 23. Overall Survival Rate
# ============================================================

overall_survival_rate = (
    df["Survived"].mean() * 100
)


print("\n" + "=" * 60)
print("FINAL PROJECT RESULTS")
print("=" * 60)


print("\nTotal Passengers:")

print(len(df))


print("\nTotal Survivors:")

print(df["Survived"].sum())


print("\nOverall Survival Rate:")

print(
    round(overall_survival_rate, 2),
    "%"
)


# ============================================================
# 24. Final Insights
# ============================================================

print("\n" + "=" * 60)
print("FINAL INSIGHTS")
print("=" * 60)


print("\n1. Survival Rate by Gender:")

print(
    df.groupby("Sex")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)


print("\n2. Survival Rate by Passenger Class:")

print(
    df.groupby("Pclass")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)


print("\n3. Survival Rate by Embarked Port:")

print(
    df.groupby("Embarked")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)

print("EDA PROJECT COMPLETED SUCCESSFULLY!")

print("=" * 60)