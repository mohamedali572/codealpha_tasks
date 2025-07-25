import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(Iris):
    data = pd.read_csv(Iris)
    data = data.drop('Id', axis=1)
    X = data.drop('Species', axis=1)
    y = data['Species']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return X, y_encoded, encoder, data

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)