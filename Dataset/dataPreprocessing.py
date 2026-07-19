# ==========================================
# Beijing Housing Price Prediction Project
# Phase 2 - Data Preprocessing
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# Load Dataset
# ==========================================


df = pd.read_csv(
    "new 3.csv",
    encoding="latin1",
    low_memory=False
)


print("="*60)
print("Dataset Loaded Successfully")
print("="*60)
print("Shape :", df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum()/len(df))*100
})

print(missing.sort_values(
    "Missing Percentage",
    ascending=False
))

columns_to_remove = [
    "url",
    "id",
    "price"
]

df.drop(columns=columns_to_remove, inplace=True)

print(df.shape)


before = len(df)

df = df[df["livingRoom"] != "#NAME?"]

after = len(df)

print("Rows Removed:", before-after)
print("Remaining:", after)

df["constructionTime"] = df["constructionTime"].replace(
    ["Î´Öª","0","1"],
    np.nan
)

print(df["constructionTime"].isnull().sum())

room_columns = [
    "livingRoom",
    "drawingRoom",
    "bathRoom"
]

for col in room_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print(df[room_columns].dtypes)


df["constructionTime"] = pd.to_numeric(
    df["constructionTime"],
    errors="coerce"
)
print(df["constructionTime"].dtype)


mode_columns = [
    "buildingType",
    "elevator",
    "fiveYearsProperty",
    "subway"
]

for col in mode_columns:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

df["communityAverage"] = df["communityAverage"].fillna(
    df["communityAverage"].median()
)

missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage":
    (df.isnull().sum()/len(df))*100
})

print(
    missing.sort_values(
        "Missing Percentage",
        ascending=False
    )
)

df["constructionTime"] = df["constructionTime"].fillna(
    df["constructionTime"].median()
)
print(df["constructionTime"].isnull().sum())


df["tradeTime"] = pd.to_datetime(
    df["tradeTime"],
    format="%Y-%m-%d",
    errors="coerce"
)
print(df["tradeTime"].dtype)
print(df["tradeTime"].head())
print(df["tradeTime"].isnull().sum())
# Trade Year
df["tradeYear"] = df["tradeTime"].dt.year

# Trade Month
df["tradeMonth"] = df["tradeTime"].dt.month

# Trade Quarter
df["tradeQuarter"] = df["tradeTime"].dt.quarter

# House Age
df["houseAge"] = df["tradeYear"] - df["constructionTime"]

print(df[[
    "tradeTime",
    "tradeYear",
    "tradeMonth",
    "tradeQuarter",
    "constructionTime",
    "houseAge"
]].head())

# Extract floor number
df["floorNumber"] = df["floor"].str.extract(r"(\d+)").astype(float)

print(df[["floor", "floorNumber"]].head())

df.drop(columns=["floor"], inplace=True)
print(df["DOM"].describe())
print(df["DOM"].value_counts().head(20))
print(df["DOM"].isnull().sum())

df.drop(columns=["DOM"], inplace=True)

print(df.shape)
print(df.dtypes)

df.drop(columns=["tradeTime"], inplace=True)

print(df.shape)

missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum()/len(df))*100
})

print(missing[missing["Missing Count"]>0])


import matplotlib.pyplot as plt

# Total Price
plt.figure(figsize=(6,4))
plt.boxplot(df["totalPrice"])
plt.title("Boxplot of Total Price")
plt.show()

# Square
plt.figure(figsize=(6,4))
plt.boxplot(df["square"])
plt.title("Boxplot of Square")
plt.show()

# Community Average
plt.figure(figsize=(6,4))
plt.boxplot(df["communityAverage"])
plt.title("Boxplot of Community Average")
plt.show()


def count_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"\nColumn: {column}")
    print(f"Q1 = {Q1}")
    print(f"Q3 = {Q3}")
    print(f"IQR = {IQR}")
    print(f"Lower Bound = {lower}")
    print(f"Upper Bound = {upper}")
    print(f"Outliers = {len(outliers)}")


count_outliers("totalPrice")
count_outliers("square")
count_outliers("communityAverage")


# df.to_csv("beijing_housing_cleaned.csv", index=False)

# print("Cleaned dataset saved successfully!")

print(df.shape)
print(df.info())

print(df.columns)