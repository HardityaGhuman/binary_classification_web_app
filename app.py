import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    # DATA 
    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data
    def split(df):
        y = df["type"]
        x = df.drop(columns=["type"])
        return train_test_split(x, y, test_size=0.3, random_state=0)

    # METRICS
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            st.pyplot(fig)

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    # LOAD
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # CLASSIFIER
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
    )

    metrics = st.sidebar.multiselect(
        "What metrics to plot?",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
    )

    # SVM
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"))

        if st.sidebar.button("Classify"):
            st.subheader("SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(metrics)

    # LOGISTIC REGRESSION
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization)", 0.01, 10.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider(
            "Maximum iterations", 100, 500, key="max_iter"
        )

        if st.sidebar.button("Classify", key="lr"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(metrics)

    # RANDOM FOREST
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")

        n_estimators = st.sidebar.number_input(
            "Number of Trees", 100, 5000, step=10
        )
        max_depth = st.sidebar.number_input(
            "Max Depth", 1, 20, step=1
        )
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees",
            (True, False)
        )

        if st.sidebar.button("Classify", key="rf"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap,
                random_state=0
            )
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = model.score(x_test, y_test)

            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))

            plot_metrics(metrics)

    # RAW DATA
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(df)


if __name__ == "__main__":
    main()