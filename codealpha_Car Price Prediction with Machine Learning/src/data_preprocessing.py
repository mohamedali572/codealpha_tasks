import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(cardata):
    # Load data
    df = pd.read_csv(cardata)


    # Handle missing values
    df = df.dropna()

    # Encoding categorical columns
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_enc.fit_transform(df[col])

    # Features and target
    X = df.drop('Selling_Price', axis=1)  # assuming 'Price' is the target column
    y = df['Selling_Price']

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)
