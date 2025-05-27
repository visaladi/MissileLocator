import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(
    filepath=r"C:\Users\visal Adikari\OneDrive\Desktop\researches\research missile\data\missile_attacks_daily.csv"
):
    """
    Load and preprocess the missile-attacks dataset.
    - Parses time_start into datetime
    - Extracts hour & month
    - Fills missing numeric flags/counts with 0
    """
    df = pd.read_csv(filepath)

    # parse datetime
    df['datetime'] = pd.to_datetime(df['time_start'], errors='coerce')
    df['hour']     = df['datetime'].dt.hour
    df['month']    = df['datetime'].dt.month

    # numeric columns that may have NaNs
    num_cols = [
        'launched','destroyed','not_reach_goal',
        'cross_border_belarus','back_russia','still_attacking',
        'num_hit_location','num_fall_fragment_location'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    return df

def encode_features(df):
    """
    - One-hot encode: model, target, carrier
    - Stack with numeric time & count features
    """
    cat_cols = ['model', 'target', 'carrier']
    enc_feat = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = enc_feat.fit_transform(df[cat_cols])

    num_cols = [
        'hour','month',
        'launched','destroyed','not_reach_goal',
        'cross_border_belarus','back_russia','still_attacking',
        'num_hit_location','num_fall_fragment_location'
    ]
    X_num = df[num_cols].values

    X = np.hstack([X_cat, X_num])
    return X, enc_feat

def encode_labels(df):
    """
    - Label-encode the launch_place string
    """
    le = LabelEncoder()
    y = le.fit_transform(df['launch_place'])
    return y, le

if __name__ == "__main__":
    # 1. Load & preprocess
    df = load_data()

    # 2. Encode features & labels
    X, feat_encoder = encode_features(df)
    y, label_encoder = encode_labels(df)

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Quick sanity-check
    print("Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")

    # Now you can plug X_train/y_train into any sklearn estimator, e.g.:
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier().fit(X_train, y_train)
    # print("Test accuracy:", clf.score(X_test, y_test))
