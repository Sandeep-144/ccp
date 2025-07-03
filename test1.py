import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("Iranian_Churn_Dataset.csv")

# Fix column name issues
df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Outlier detection using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Removing extreme outliers (1.5 * IQR rule)
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Engineering
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
feature_cols = ['Subscription Length', 'Charge Amount', 'Seconds of Use', 'Frequency of use', 'Customer Value']
X_poly = poly.fit_transform(df[feature_cols])
df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_cols))
df = pd.concat([df, df_poly], axis=1)

# Log transformation to reduce skewness
df['Seconds of Use'] = np.log1p(df['Seconds of Use'])

def plot_distribution(col):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Plot distributions
for col in ['Subscription Length', 'Charge Amount', 'Seconds of Use', 'Customer Value']:
    plot_distribution(col)

# Convert categorical features to numeric
df['Tariff Plan'] = df['Tariff Plan'].map({1: 0, 2: 1})
df['Status'] = df['Status'].map({1: 0, 2: 1})

# Define features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ANN model
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()
