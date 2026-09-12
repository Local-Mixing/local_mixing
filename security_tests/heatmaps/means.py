import pandas as pd
import matplotlib.pyplot as plt

# load data
means = pd.read_csv("means.txt", header=None, names=["mean"])

# basic stats
print(means.describe())
print("Outliers:")
q1 = means["mean"].quantile(0.25)
q3 = means["mean"].quantile(0.75)
iqr = q3 - q1
outliers = means[(means["mean"] < q1 - 1.5*iqr) | (means["mean"] > q3 + 1.5*iqr)]
print(outliers)

# plots
plt.figure()
plt.boxplot(means["mean"], vert=False)
plt.title("Means – Boxplot")

plt.figure()
plt.hist(means["mean"], bins=50)
plt.title("Means – Histogram")

plt.show()
