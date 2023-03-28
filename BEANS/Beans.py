from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('beans.csv')

print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Check for outliers
print(df.describe())

# Define predictor and outcome variables
X = df[['']]
y = df['']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit multiple regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict outcomes for testing data
y_pred = reg.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Fit PCA model
pca = PCA(n_components=2)
pca.fit(X)

# Transform data to principal components
X_pca = pca.transform(X)

# Split PCA data into training and testing sets
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Fit regression model with principal components
reg_pca = LinearRegression().fit(X_pca_train, y_train)

# Predict outcomes for testing data

y_pred_pca = reg_pca.predict(X_pca_test)
mse_pca = mean_squared_error(y_test, y_pred_pca)
print('Mean squared error (without PCA):', mse)
print('Mean squared error (with PCA):', mse_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')
plt.show()
