import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(cardata):
    # 1. Load data
    df = pd.read_csv(cardata)

    # 2. Handle missing values (fill instead of drop)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # 3. Encode categorical features
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_enc.fit_transform(df[col])

    # 4. Feature / Target split
    if 'Selling_Price' not in df.columns:
        raise ValueError("Target column 'Selling_Price' not found in dataset")

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    # 5. Feature scaling (optional but recommended)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 6. Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

