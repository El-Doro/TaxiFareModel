import pandas as pd
from sklearn.model_selection import train_test_split

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def get_Xy(df):
    # set X and y
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    return X,y


def hold_out(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X,y = get_Xy(df)
    # hold out
    X_train, X_val, y_train, y_val = hold_out(X,y)
