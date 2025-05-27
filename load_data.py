import pandas as pd


def load_data(
        filepath=r"C:\Users\visal Adikari\OneDrive\Desktop\researches\research missile\data\missile_attacks_daily.csv"):
    """
    Load and preprocess the missile-attacks dataset.
    - Parses time_start into datetime
    - Extracts hour & month
    - Fills missing numeric flags/counts with 0
    """
    df = pd.read_csv(filepath)

    # parse datetime
    df['datetime'] = pd.to_datetime(df['time_start'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month

    # numeric columns that may have NaNs
    num_cols = [
        'launched', 'destroyed', 'not_reach_goal',
        'cross_border_belarus', 'back_russia', 'still_attacking',
        'num_hit_location', 'num_fall_fragment_location'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    return df


if __name__ == "__main__":
    # Load using the default path:
    df = load_data()
    print(df.head())

    # Or, explicitly:
    # custom_path = r"C:\Users\visal Adikari\OneDrive\Desktop\researches\research missile\data\missile_attacks_daily.csv"
    # df = load_data(custom_path)
