from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ===================== #
    # 1) Scatter Plot
    # ===================== #
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='k')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual Selling Price")
    ax1.set_ylabel("Predicted Selling Price")
    ax1.set_title("Actual vs Predicted")

    # ===================== #
    # 2) Residual Plot
    # ===================== #
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.scatter(y_pred, residuals, alpha=0.7, color='green', edgecolors='k')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel("Predicted Selling Price")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")

    # ===================== #
    # 3) Histogram of Errors
    # ===================== #
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Error Distribution")

    # Return metrics + figures
    return (mae, rmse, r2), [fig1, fig2, fig3]


