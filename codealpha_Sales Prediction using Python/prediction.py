import numpy as np

def predict_sales(new_data, scaler, model):
    """
    Make prediction using the provided trained model and scaler.
    """
    # Scale the new data
    new_data_scaled = scaler.transform(np.array(new_data).reshape(1, -1))

    # Predict using the selected model
    prediction = model.predict(new_data_scaled)

    return prediction[0]








