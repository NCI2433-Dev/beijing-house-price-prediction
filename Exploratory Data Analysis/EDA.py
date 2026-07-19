import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("beijing_housing_cleaned.csv")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

# ============================================
# 1. Target Variable Distribution
# ============================================

plt.figure(figsize=(8,5))

plt.hist(df["totalPrice"], bins=100)

plt.xlim(0, 3000)

plt.title("Distribution of House Prices")
plt.xlabel("Total Price")
plt.ylabel("Frequency")

plt.show()


# ============================================
# 2. House Size Distribution
# ============================================

plt.figure(figsize=(8,5))

plt.hist(df["square"], bins=100)      # <-- Correct column

plt.xlim(0, 300)

plt.title("Distribution of House Size")
plt.xlabel("Square Area (m²)")
plt.ylabel("Frequency")

plt.show()


# ============================================
# 3. House Size vs Total Price
# ============================================

plt.figure(figsize=(8,6))

plt.scatter(df["square"], df["totalPrice"], alpha=0.3)

plt.xlabel("Square Area (m²)")
plt.ylabel("Total Price")

plt.title("House Size vs Total Price")

plt.show()



# Calculate correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Create heatmap
plt.figure(figsize=(18, 14))

plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns,
           rotation=90)

plt.yticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# Correlation with target variable
price_corr = df.corr(numeric_only=True)["totalPrice"]

# Sort correlations
price_corr = price_corr.sort_values(ascending=False)

print(price_corr)


plt.figure(figsize=(10,8))

price_corr.drop("totalPrice").plot(kind="bar")

plt.title("Feature Correlation with Total House Price")

plt.xlabel("Features")
plt.ylabel("Correlation")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.show()


price_corr = df.corr(numeric_only=True)["totalPrice"].sort_values(ascending=False)
print("price correlation with total price:")
print(price_corr)


# # removing the CID column as it is not useful for analysis because it is just an identifier and does not provide any meaningful information for the analysis.
df.drop(columns=["Cid"], inplace=True)
print("Remaining columns:", df.columns.tolist())

df.to_csv("beijing_housing_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")

