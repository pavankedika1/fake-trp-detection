import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import bcrypt
import random
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Advanced Fake TRP Platform",
    layout="wide"
)

# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect(
    "advanced_trp.db",
    check_same_thread=False
)

cur = conn.cursor()

# USERS TABLE
cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password BLOB
)
""")

# FRAUD LOG TABLE
cur.execute("""
CREATE TABLE IF NOT EXISTS fraud_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT,
    trp_value REAL,
    fraud_probability REAL,
    result TEXT
)
""")

conn.commit()

# ============================================================
# SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Login"

# ============================================================
# USER FUNCTIONS
# ============================================================

def register_user(username, password):

    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    )

    cur.execute(
        "INSERT INTO users(username,password) VALUES (?,?)",
        (username, hashed)
    )

    conn.commit()


def login_user(username, password):

    cur.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )

    result = cur.fetchone()

    if result:

        stored_password = result[0]

        if bcrypt.checkpw(
            password.encode(),
            stored_password
        ):
            return True

    return False

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    # IMPORTANT:
    # Put advanced_trp_dataset.csv
    # in SAME folder as this Python file

    data = pd.read_csv("advanced_trp_dataset.csv")

    return data

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = load_dataset()

    # TARGET
    data = data.rename(
        columns={"is_fake": "Fake_TRP"}
    )

    # LABEL ENCODERS
    encoders = {}

    categorical_cols = [
        "channel",
        "device_id",
        "ip_address",
        "region",
        "time_of_day"
    ]

    for col in categorical_cols:

        encoder = LabelEncoder()

        data[col] = encoder.fit_transform(
            data[col]
        )

        encoders[col] = encoder

    # REMOVE DATE
    if "date" in data.columns:
        data = data.drop("date", axis=1)

    # FEATURES
    X = data.drop("Fake_TRP", axis=1)

    y = data["Fake_TRP"]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # MODEL
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return (
        model,
        encoders,
        accuracy,
        X_test
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Advanced Fake TRP Platform")

# REGISTER
if st.sidebar.button("Register"):
    st.session_state.page = "Register"

# LOGIN
if st.sidebar.button("Login"):
    st.session_state.page = "Login"

# DASHBOARD
if st.sidebar.button("Dashboard"):
    st.session_state.page = "Dashboard"

# PREDICTION
if st.sidebar.button("Prediction"):
    st.session_state.page = "Prediction"

# ANALYTICS
if st.sidebar.button("Analytics"):
    st.session_state.page = "Analytics"

# LOGOUT
if st.sidebar.button("Logout"):

    st.session_state.logged_in = False

    st.session_state.page = "Login"

    st.rerun()

menu = st.session_state.page

# ============================================================
# REGISTER
# ============================================================

if menu == "Register":

    st.title("Register")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Register User"):

        register_user(username, password)

        st.success("Registration Successful")

        time.sleep(1)

        st.session_state.page = "Login"

        st.rerun()

# ============================================================
# LOGIN
# ============================================================

elif menu == "Login":

    st.title("Secure Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login User"):

        if login_user(username, password):

            st.session_state.logged_in = True

            st.success("Login Successful")

            time.sleep(1)

            st.session_state.page = "Dashboard"

            st.rerun()

        else:

            st.error("Invalid Credentials")

# ============================================================
# DASHBOARD
# ============================================================

elif menu == "Dashboard":

    if st.session_state.logged_in:

        st.title("Advanced AI Fraud Dashboard")

        data = load_dataset()

        (
            model,
            encoders,
            accuracy,
            X_test
        ) = train_model()

        # METRICS
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Dataset Rows",
            len(data)
        )

        col2.metric(
            "Model Accuracy",
            f"{accuracy*100:.2f}%"
        )

        col3.metric(
            "Fraud Cases",
            int(data['is_fake'].sum())
        )

        # DATASET
        st.subheader("Dataset Preview")

        st.dataframe(data.head(5))

        # FEATURE IMPORTANCE
        st.subheader("Feature Importance")

        importance = model.feature_importances_

        features = X_test.columns

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.bar(features, importance)

        plt.xticks(rotation=90)

        st.pyplot(fig)

        # LIVE ALERTS
        st.subheader("Live Fraud Alerts")

        alerts = [
            "High TRP spike detected",
            "Suspicious viewing pattern found",
            "Repeated device activity detected",
            "Possible bot traffic identified"
        ]

        for alert in alerts:
            st.warning(alert)

    else:

        st.warning("Please Login First")

# ============================================================
# PREDICTION
# ============================================================

elif menu == "Prediction":

    if st.session_state.logged_in:

        st.title("Advanced Fraud Prediction")

        (
            model,
            encoders,
            accuracy,
            X_test
        ) = train_model()

        channel = st.text_input("Channel")

        viewing_minutes = st.number_input("Viewing Minutes")

        devices_used = st.number_input("Devices Used")

        location_count = st.number_input("Location Count")

        trp_value = st.number_input("TRP Value")

        device_id = st.text_input("Device ID")

        ip_address = st.text_input("IP Address")

        session_duration = st.number_input("Session Duration")

        channel_switches = st.number_input("Channel Switches")

        watch_pattern_score = st.number_input("Watch Pattern Score")

        device_behavior_score = st.number_input("Device Behavior Score")

        region = st.text_input("Region")

        time_of_day = st.selectbox(
            "Time Of Day",
            ["Morning", "Afternoon", "Evening", "Night", "Midnight"]
        )

        if st.button("Detect Fraud"):

            try:
                channel_encoded = encoders['channel'].transform([channel])[0]
            except:
                channel_encoded = 0

            try:
                device_encoded = encoders['device_id'].transform([device_id])[0]
            except:
                device_encoded = 0

            try:
                ip_encoded = encoders['ip_address'].transform([ip_address])[0]
            except:
                ip_encoded = 0

            try:
                region_encoded = encoders['region'].transform([region])[0]
            except:
                region_encoded = 0

            try:
                time_encoded = encoders['time_of_day'].transform([time_of_day])[0]
            except:
                time_encoded = 0

            input_data = [[
                channel_encoded,
                viewing_minutes,
                devices_used,
                location_count,
                trp_value,
                device_encoded,
                ip_encoded,
                session_duration,
                channel_switches,
                watch_pattern_score,
                device_behavior_score,
                region_encoded,
                time_encoded
            ]]

            prediction = model.predict(input_data)

            probability = model.predict_proba(input_data)

            fraud_probability = probability[0][1] * 100

            if prediction[0] == 1:

                result = "Fake TRP Detected"

                st.error(result)

            else:

                result = "Genuine TRP"

                st.success(result)

            st.info(
                f"Fraud Probability: {fraud_probability:.2f}%"
            )

            # SAVE LOG
            cur.execute(
                """
                INSERT INTO fraud_logs(
                    channel,
                    trp_value,
                    fraud_probability,
                    result
                )
                VALUES (?,?,?,?)
                """,
                (
                    channel,
                    trp_value,
                    fraud_probability,
                    result
                )
            )

            conn.commit()

            # AUTO REDIRECT
            time.sleep(2)

            st.session_state.page = "Analytics"

            st.rerun()

    else:

        st.warning("Please Login First")

# ============================================================
# ANALYTICS
# ============================================================

elif menu == "Analytics":

    if st.session_state.logged_in:

        st.title("Fraud Analytics")

        logs = pd.read_sql_query(
            "SELECT * FROM fraud_logs ORDER BY id DESC",
            conn
        )

        st.dataframe(logs)

        if len(logs) > 0:

            st.subheader("Fraud Probability Trend")

            fig2, ax2 = plt.subplots(figsize=(10, 5))

            ax2.plot(logs['fraud_probability'])

            ax2.set_xlabel("Log Index")

            ax2.set_ylabel("Fraud Probability")

            st.pyplot(fig2)

    else:

        st.warning("Please Login First")     #streamlit run enterprise_trp_platform.py      

