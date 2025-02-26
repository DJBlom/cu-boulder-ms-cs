# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


plt.close('all')

# Load the dataset
df = pd.read_csv('plant_growth_data.csv')

# 🔹 Select Only Core Features
plt.figure(figsize=(8, 6))
sns.countplot(x='Growth_Milestone', data=df, hue='Growth_Milestone', palette='viridis', legend=False)
plt.title("Distribution of Plant Growth Stages")
plt.xlabel("Growth Stage (0 = Early, 1 = Mature)")
plt.ylabel("Number of Samples")
plt.savefig('growth_stage_distribution.png')
plt.show()


# 🔹 Select Core Features
feature_columns = ['Sunlight_Hours', 'Temperature', 'Humidity']
df['Growth_Milestone'] = df['Growth_Milestone'].astype('category').cat.codes

# 🔹 Train-test split
X = df[feature_columns]
y = df['Growth_Milestone']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 EDA: Feature Distributions
plt.figure(figsize=(18, 6))
for i, feature in enumerate(feature_columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[feature], kde=True, bins=20, color='blue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# 🔹 EDA: Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[feature_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.xlabel("Environmental Factors")
plt.ylabel("Environmental Factors")
plt.savefig('feature_correlation_heatmap.png')
plt.show()

# 🔹 EDA: Pairplot of Features
pairplot_fig = sns.pairplot(df, vars=feature_columns, hue='Growth_Milestone', palette='viridis', diag_kind="kde")
plt.suptitle("Pairwise Relationships Between Environmental Factors", y=1.02)
pairplot_fig.savefig('feature_pairplot.png')

# 🔹 Train SVM Model with RBF Kernel (All Features)
svm_model = SVC(C=10.0, kernel='rbf', degree=20, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
svm_model.fit(X_train_scaled, y_train)

# 🔹 Predict with SVM
y_pred_svm = svm_model.predict(X_test_scaled)

# 🔹 Evaluate SVM Performance
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"\nSVM (RBF Kernel) Accuracy: {svm_accuracy:.4f}")
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm))

# 🔹 Confusion Matrix (SVM)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, cmap='viridis', fmt='d')
plt.title('Confusion Matrix\n(SVM with RBF Kernel)')
plt.xlabel('Predicted Growth Stage')
plt.ylabel('Actual Growth Stage')
plt.savefig('svm_confusion_matrix.png')
plt.show()

# 🔹 Train a 2D SVM Model for Visualization (Using Only Two Features)
X_train_vis = X_train_scaled[:, :2]  # Use first two features (Sunlight_Hours & Temperature)
X_test_vis = X_test_scaled[:, :2]
svm_model_vis = SVC(C=10.0, kernel='rbf', degree=20, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
svm_model_vis.fit(X_train_vis, y_train)

# 🔹 Decision Boundary Plot (For Visualization)
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test_vis, y_test.to_numpy(), clf=svm_model_vis, legend=2)
plt.title("SVM Decision Boundary (Using Sunlight & Temperature)")
plt.xlabel(feature_columns[0])
plt.ylabel(feature_columns[1])
plt.savefig('svm_decision_boundary.png')
plt.show()
