import joblib
import pandas as pd

def predict_sales(new_data, scaler):
    model = joblib.load("models/sales_model.pkl")
    
    # Create dataframe with same feature names
    feature_names = ['TV', 'Radio', 'Newspaper']
    new_df = pd.DataFrame([new_data], columns=feature_names)
    
    # Scale using the scaler
    scaled_data = scaler.transform(new_df)
    
    # Predict sales
    prediction = model.predict(scaled_data)
    return prediction[0]

