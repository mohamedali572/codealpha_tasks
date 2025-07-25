from data_preprocessing import load_and_preprocess_data
from model_training import train_and_save_model
from prediction import predict_sales
from visualization import plot_correlation, plot_feature_importance

import pandas as pd

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(r"C:\Users\mS\Documents\CodeAlpha\Data Science\TASK 4 Sales Prediction using Python\Advertising.csv")

# Step 2: Train models and save the best
model = train_and_save_model(X_train, y_train, X_test, y_test)

# Step 3: Example prediction (TV, Radio, Newspaper spends)
new_data = [230, 37, 69]  # Example ad spending
prediction = predict_sales(new_data, scaler)
print(f"Expected Sales: {prediction:.2f}")
# Plot correlation matrix using the raw dataset
raw_df = pd.read_csv(r"C:\Users\mS\Documents\CodeAlpha\Data Science\TASK 4 Sales Prediction using Python\Advertising.csv")
plot_correlation(raw_df)

# Plot feature importance if model supports it (e.g., Random Forest)
plot_feature_importance(model, ['TV', 'Radio', 'Newspaper'])

