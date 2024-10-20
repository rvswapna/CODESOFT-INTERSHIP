import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('IMDb Movies India.csv')

# Perform data preprocessing and feature engineering steps

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define preprocessing steps (e.g., one-hot encoding for categorical variables)
preprocessor = ColumnTransformer(
    transformers=[
        ('genre_encoder', OneHotEncoder(), ['genre']),
        ('director_encoder', OneHotEncoder(), ['director']),
        ('actors_encoder', OneHotEncoder(), ['actor1', 'actor2', 'actor3'])
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
