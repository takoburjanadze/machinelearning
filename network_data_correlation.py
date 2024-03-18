import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generating network packet data
np.random.seed(0)
data = np.random.rand(200, 4)  # Generating random data with 4 features and 200 samples

# Writing generated data to CSV file
df = pd.DataFrame(data)
df.to_csv('network_data.csv', index=False, header=False)

# Reading generated data and calculating correlation matrix
df = pd.read_csv('network_data.csv', header=None)  # Read data from CSV file
correlation_matrix = df.corr(method='pearson')  # Calculate correlation matrix using Pearson formula

# Displaying the names of features in the correlation matrix
feature_names = [f'Feature_{i+1}' for i in range(4)]
correlation_matrix.columns = feature_names
correlation_matrix.index = feature_names

# Saving correlation matrix to PDF file
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.pdf')

# Finding the two features with the highest correlation
correlation_values = correlation_matrix.unstack()
sorted_correlation_values = correlation_values.sort_values(ascending=False)
highest_correlation = sorted_correlation_values[sorted_correlation_values < 1].head(2)

# Saving the names of two features with the highest correlation to PDF file
try:
    with open('highest_correlation.pdf', 'w') as f:
        f.write(f'Two features with the highest correlation: {highest_correlation.index[0][0]} and {highest_correlation.index[1][0]}')
    print("Successfully saved 'highest_correlation.pdf'.")
except Exception as e:
    print("Error saving 'highest_correlation.pdf':", e)

# Displaying the correlation matrix
plt.show()

# I couldn’t open highest_correlation.pdf for some reason, so then i opened it using python again and then converted it into a PDF

>>> # Read and print the content of the "highest_correlation.pdf" file
>>> with open('highest_correlation.pdf', 'r') as f:
...     print(f.read())
...
Two features with the highest correlation: Feature_2 and Feature_4
