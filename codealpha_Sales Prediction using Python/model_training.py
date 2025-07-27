from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_and_save_model(X_train, y_train, X_test, y_test, save=False):
    """
    Train multiple models (Linear Regression, Random Forest) and return them with metrics.
    Returns a dictionary: {model_name: {"model": model_object, "r2": value, "rmse": value}}
    """

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        trained_models[name] = {
            "model": model,
            "r2": r2,
            "rmse": rmse
        }

    # لا نحفظ الموديل هنا، GUI هو اللي بيقرر الحفظ
    return trained_models






