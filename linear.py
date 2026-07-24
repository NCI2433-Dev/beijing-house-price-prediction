import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([6,8,10,13,15,18,20,22,25,27])

# Fit linear regression line
m, c = np.polyfit(X, Y, 1)
Y_pred = m * X + c

# Plot
plt.figure(figsize=(6,4))

plt.scatter(X, Y, label='Data Points')
plt.plot(X, Y_pred, label='Regression Line')

plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Target")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("linear_regression.png", dpi=300)
plt.show()