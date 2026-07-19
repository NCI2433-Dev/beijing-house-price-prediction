import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("beijing_housing_cleaned.csv")

print(df.shape)

X = df.drop("totalPrice", axis=1)
y = df["totalPrice"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

print("Remaining columns:", df.columns.tolist())

print("totalPrice" in X.columns)

df.to_csv("beijing_housing_cleaned.csv", index=False)

print("Final cleaned dataset saved successfully!")