from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Print number of rows and columns
num_rows, num_columns = df.shape
print(f"Number of Rows: {num_rows}, Number of Columns: {num_columns}")

# Get min and max of the first column (Sepal Length)
min_value = df.iloc[:, 0].min()
max_value = df.iloc[:, 0].max()
print(f"Sepal Length Range: {min_value} cm to {max_value} cm")

# Get unique class values (target labels)
class_values = df['target'].unique()
print(f"Unique Class Values: {class_values}")

# Split dataset into training (80%) and testing (20%) sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Print the number of rows in each set
print(f"Training Set Rows: {train_set.shape[0]}")
print(f"Testing Set Rows: {test_set.shape[0]}")

# Print full training and testing sets
print("\nTraining Set:")
print(train_set.to_string())  # Print entire training set

print("\nTesting Set:")
print(test_set.to_string())  # Print entire testing set
