import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv", encoding="latin1")


# Extract NO2 column (lowercase!)
x = df["no2"].dropna()

print("Total NO2 values:", len(x))
r = 102303605
a_r = 0.05 * (r % 7)
b_r = 0.3 * ((r % 5) + 1)

print("a_r =", a_r)
print("b_r =", b_r)



z = x + a_r * np.sin(b_r * x)



mu = np.mean(z)
variance = np.var(z)

lambda_val = 1 / (2 * variance)

sigma = np.sqrt(variance)
c = 1 / (np.sqrt(2 * np.pi) * sigma)

print("\nEstimated Parameters:")
print("mu =", mu)
print("lambda =", lambda_val)
print("c =", c)

z_sorted = np.sort(z)
pdf = c * np.exp(-lambda_val * (z_sorted - mu)**2)

plt.hist(z, bins=50, density=True)
plt.plot(z_sorted, pdf)
plt.title("Histogram and Learned PDF")
plt.xlabel("z values")
plt.ylabel("Density")
plt.show()


with open("results.txt", "w") as f:
    f.write(f"mu = {mu}\n")
    f.write(f"lambda = {lambda_val}\n")
    f.write(f"c = {c}\n")

print("Results saved successfully in results.txt")

