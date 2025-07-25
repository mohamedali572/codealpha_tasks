import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_save_model(X_train, y_train, X_test, y_test):
    # Initialize models
    lin_model = LinearRegression()
    rf_model = RandomForestRegressor(random_state=42)
    
    # Train models
    lin_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # Evaluate models
    models = {'Linear Regression': lin_model, 'Random Forest': rf_model}
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name} → R²: {score:.2f}, RMSE: {rmse:.2f}")

        if score > best_score:
            best_score = score
            best_model = model

    # Ensure the models folder exists
    os.makedirs("models", exist_ok=True)

    # Save best model
    joblib.dump(best_model, "models/sales_model.pkl")
    print("Best model saved as sales_model.pkl")

    return best_model

