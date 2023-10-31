import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the training and test data
train_data = pd.read_csv("train_final.csv")
test_data = pd.read_csv("test_final.csv")

# Define features (X) and target (y)
X = train_data.drop("income>50K", axis=1)
y = train_data["income>50K"]

# Encode categorical variables and scale continuous features
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define the classifier 
classifier = RandomForestClassifier(random_state=42)

# Create a pipeline that includes preprocessing and modeling
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Train a decision tree classifier
model.fit(X, y)

# Make predictions on the test data
test_X = test_data.drop("ID", axis=1)  # Remove the ID column

# predictions = model.predict(test_X)
probabilities = model.predict_proba(test_X)[:, 1]

# Create a DataFrame for the test predictions
test_predictions = pd.DataFrame({
    "ID": test_data["ID"],
    "Prediction": probabilities
})

# Save the predictions to a CSV file
test_predictions.to_csv("test_predictions.csv", index=False)

print(test_predictions)
