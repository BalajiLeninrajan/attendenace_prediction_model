import pickle
import requests
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

EPOCHS = 200
TEST_SIZE = 0.2
TEST_SEED = 75
DROPOUT_RATE = 0.1


def fetch_data(from_date, to_date, emp_id):
    print(f"Fetching {emp_id}'s records")
    return requests.post(
        "https://ytygroup.app/GA/api/getRawAttendance.php",
        json={
            "from_date": from_date,
            "to_date": to_date,
            "emp_id": emp_id
        },
        verify=False
    ).json()


def fetch_all_data():
    valid_ids = [
        "300008",
        "302462",
        "303046",
        "303525",
        "701092",
        "701055",
        "701135",
        "701092",
        "700792",
        "303520"
    ]

    data = []

    for valid_id in valid_ids:
        data.extend(fetch_data(
            "2022-01-01",
            "2024-05-31",
            valid_id
        ))

    return data


def preprocess_data(raw_data):
    df = pd.DataFrame(raw_data)
    df = df[df["SHIFT_DESC"].notnull()]
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

    d_type_label_encoder = LabelEncoder()
    df["D_TYPE"] = d_type_label_encoder.fit_transform(df["D_TYPE"])

    shift_label_encoder = LabelEncoder()
    df["SHIFT_DESC"] = shift_label_encoder.fit_transform(df["SHIFT_DESC"])

    scaler = StandardScaler()
    data = scaler.fit_transform(df.drop(columns=["SHIFT_DESC"]))
    labels = df["SHIFT_DESC"]

    labels = to_categorical(labels)

    return data, labels, d_type_label_encoder, shift_label_encoder, scaler


def build_model(data_shape, label_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            1024, activation="relu", input_shape=(data_shape,)
        ),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(label_shape, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    data, labels, d_type_label_encoder, shift_label_encoder, scaler = preprocess_data(
        fetch_all_data()
    )

    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=TEST_SEED
    )

    data_train = data_train.astype("float32")
    data_test = data_test.astype("float32")

    model = build_model(data_train.shape[1], label_train.shape[1])
    model.fit(data_train, label_train, batch_size=32, epochs=EPOCHS)

    loss, accuracy = model.evaluate(data_test, label_test)
    print(
        f"Test accuracy: {accuracy * 100:.2f}% | Model loss: {loss * 100:.2f}%"
    )

    model.save("yty_attendance_model.keras")
    with open("additional_data.pkl", "wb") as f:
        pickle.dump({
            "d_type_label_encoder": d_type_label_encoder,
            "shift_label_encoder": shift_label_encoder,
            "scaler": scaler
        }, f)


if __name__ == "__main__":
    main()
