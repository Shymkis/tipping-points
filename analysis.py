import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('memory.csv', header=None)
grouped = df.groupby(0)
for name, group in grouped:
    vals = group.iloc[:, 2:].values
    x = group[1].values
    y = np.mean(vals, axis=1)
    y_err = np.std(vals, axis=1, ddof=1) / np.sqrt(vals.shape[1])
    plt.errorbar(x, y, y_err, label=name)
    plt.fill_between(x, y - y_err, y + y_err, alpha=0.25)

plt.xlabel("Memory")
plt.ylabel("Tipping Point")
plt.legend()
plt.tight_layout()
plt.show()
