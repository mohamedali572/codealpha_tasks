from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # ===================== #
    # 1) Scatter Plot
    # ===================== #
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Selling Price")
    plt.grid(True)
    plt.show()

    # ===================== #
    # 2) Residual Plot
    # ===================== #
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green', edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Selling Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()

    # ===================== #
    # 3) Histogram of Errors
    # ===================== #
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True)
    plt.show()

