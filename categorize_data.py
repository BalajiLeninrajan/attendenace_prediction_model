import pickle
import pandas as pd
import tensorflow as tf


def preprocess_data(raw_data, d_type_label_encoder, scaler):
    df = pd.DataFrame(raw_data)
    df = df.drop(columns=["EMPLOYEE_NAME"])

    df["ACCESS_DATE"] = df["ACCESS_DATE"].fillna("2000-01-01")
    df["ACCESS_TIME"] = df["ACCESS_TIME"].fillna("12:00")

    df["ACCESS_DATETIME"] = pd.to_datetime(
        df["ACCESS_DATE"] + " " + df["ACCESS_TIME"]
    )
    df["HOUR"] = df["ACCESS_DATETIME"].dt.hour
    df["WEEK"] = df["ACCESS_DATETIME"].dt.dayofweek
    df["DAY"] = df["ACCESS_DATETIME"].dt.day

    df = df.drop(columns=["ACCESS_DATE", "ACCESS_TIME", "ACCESS_DATETIME"])

    df["D_TYPE"] = d_type_label_encoder.fit_transform(df["D_TYPE"])

    data = scaler.fit_transform(df)

    return data


def main():
    model = tf.keras.models.load_model("yty_attendance_model.keras")
    with open("additional_data.pkl", "rb") as f:
        utils = pickle.load(f)

    new_raw_data = [
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": "2024-06-19",
            "ACCESS_TIME": "19:05",
            "D_TYPE": "OUT"
        },
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": "2024-06-19",
            "ACCESS_TIME": "06:20",
            "D_TYPE": "IN"
        },
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": "2024-06-20",
            "ACCESS_TIME": "06:15",
            "D_TYPE": "OUT"
        },
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": "2024-06-20",
            "ACCESS_TIME": "12:00",
            "D_TYPE": "IN"
        },
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": None,
            "ACCESS_TIME": None,
            "D_TYPE": None
        },
        {
            "EMPLOYEE_ID": "701092",
            "EMPLOYEE_NAME": "SWETHA SRI A/P MOGANA",
            "ACCESS_DATE": "2024-06-21",
            "ACCESS_TIME": "19:05",
            "D_TYPE": "IN"
        }
    ]

    new_data = preprocess_data(
        new_raw_data, utils["d_type_label_encoder"], utils["scaler"])

    predicted_encoded_labels = model.predict(new_data).argmax(axis=1)

    predicted_labels = utils["shift_label_encoder"].inverse_transform(
        predicted_encoded_labels)

    for i, prediction in enumerate(predicted_labels):
        print(f"{i}: {prediction}")


if __name__ == "__main__":
    main()
