# ===========================
# Step 1: Import Libraries
# ===========================

import pandas as pd
import numpy as np

# ===========================
# Step 2: Load Dataset
# ===========================

df = pd.read_csv("new 3.csv", encoding="latin1")

# ===========================
# Step 3: Display Dataset
# # ===========================

# print("First 5 Rows")
# print(df.head())

# print("\nLast 5 Rows")
# print(df.tail())

# # ===========================
# # Step 4: Dataset Shape
# # ===========================

# print("\nDataset Shape:")
# print(df.shape)

# # ===========================
# # Step 5: Column Names
# # ===========================

# print("\nColumn Names:")
# print(df.columns.tolist())
# ===========================
# Dataset Information
# ===========================

print("\nDataset Information:")
df.info()
print(df[['livingRoom','drawingRoom','bathRoom']].nunique())
print(df[['livingRoom','drawingRoom','bathRoom']].head(20))
print(df['floor'].unique()[:20])
print(df['constructionTime'].unique()[:20])

# Missing values count
missing = df.isnull().sum()

# Missing percentage

print("MISSING VALUES")
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percentage
})

print(missing_df.sort_values("Missing Percentage", ascending=False))

print("\n  sorting by missing percentage")


print(df["livingRoom"].unique())
print(df["drawingRoom"].unique())
print(df["bathRoom"].unique())
print(df["constructionTime"].unique())

print(df[df["livingRoom"].apply(lambda x: isinstance(x, str))]["livingRoom"].unique())

print(df[df["drawingRoom"].apply(lambda x: isinstance(x, str))]["drawingRoom"].unique())

print(df[df["bathRoom"].apply(lambda x: isinstance(x, str))]["bathRoom"].unique())

print(df[df["constructionTime"].apply(lambda x: isinstance(x, str))]["constructionTime"].unique())

df[df["drawingRoom"].str.contains("Ö|¸|µ|¶", na=False)]

print(df[df["drawingRoom"].str.contains("Ö|¸|µ|¶", na=False)]["drawingRoom"].unique())
problem_rows = df[df["drawingRoom"].astype(str).str.contains("Ö|¸|µ|¶", na=False)]

print("Number of corrupted rows:", len(problem_rows))

problem_rows[['livingRoom',
              'drawingRoom',
              'kitchen',
              'bathRoom',
              'floor',
              'constructionTime']].head(10)
print(problem_rows[['livingRoom',
                    'drawingRoom',
                    'kitchen',
                    'bathRoom',
                    'floor',
                    'constructionTime']].head(10))
