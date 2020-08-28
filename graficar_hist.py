import matplotlib.pyplot as plt
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
df.plot.bar(x='Word',y='Count')
plt.show()