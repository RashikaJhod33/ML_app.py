import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Universal ML App",
    layout="wide"
)

st.title("🤖 Universal Machine Learning App")

st.markdown("""
Upload any CSV dataset and automatically perform:
- Classification
- Regression
- Data preprocessing
- Model training
""")

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Dataset Preview")
        st.dataframe(data.head())

    
        st.subheader("📊 Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", data.shape[0])

        with col2:
            st.metric("Columns", data.shape[1])

        
        target_column = st.selectbox(
            "🎯 Select Target Column",
            data.columns
        )

        
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        
        high_cardinality_cols = []

        for col in X.columns:
            if X[col].dtype == 'object':
                if X[col].nunique() > 50:
                    high_cardinality_cols.append(col)

        X = X.drop(columns=high_cardinality_cols)

        if high_cardinality_cols:
            st.warning(
                f"Dropped High Cardinality Columns: {high_cardinality_cols}"
            )

        X = pd.get_dummies(X, drop_first=True)

        X = X.fillna(0)

        X = X.astype('float32')

        scaler = StandardScaler()
        X = scaler.fit_transform(X)


        problem_type = "Classification"

        if y.dtype in ['float64', 'float32', 'int64', 'int32']:

            unique_values = y.nunique()

            if unique_values > 20:
                problem_type = "Regression"

        st.success(f"Detected Problem Type: {problem_type}")

        
        if problem_type == "Classification":

            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        st.sidebar.title("⚙️ Model Settings")

        if problem_type == "Classification":

            model_option = st.sidebar.selectbox(
                "Choose Model",
                [
                    "Logistic Regression",
                    "Random Forest Classifier",
                    "Both"
                ]
            )

        else:

            model_option = st.sidebar.selectbox(
                "Choose Model",
                [
                    "Linear Regression",
                    "Random Forest Regressor",
                    "Both"
                ]
            )

        if st.button("🚀 Train Model"):

            
            if problem_type == "Classification":

                
                if model_option in [
                    "Logistic Regression",
                    "Both"
                ]:

                    st.subheader(" Logistic Regression")

                    lr = LogisticRegression(
                        max_iter=200,
                        solver='saga'
                    )

                    lr.fit(X_train, y_train)

                    pred = lr.predict(X_test)

                    acc = accuracy_score(y_test, pred)

                    st.success(f"Accuracy: {acc:.4f}")

                    st.text("Classification Report")
                    st.text(
                        classification_report(y_test, pred)
                    )

                    st.text("Confusion Matrix")
                    st.write(
                        confusion_matrix(y_test, pred)
                    )

        
                if model_option in [
                    "Random Forest Classifier",
                    "Both"
                ]:

                    st.subheader(" Random Forest Classifier")

                    rf = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )

                    rf.fit(X_train, y_train)

                    pred = rf.predict(X_test)

                    acc = accuracy_score(y_test, pred)

                    st.success(f"Accuracy: {acc:.4f}")

                    st.text("Classification Report")
                    st.text(
                        classification_report(y_test, pred)
                    )

                    st.text("Confusion Matrix")
                    st.write(
                        confusion_matrix(y_test, pred)
                    )

            
            else:

                if model_option in [
                    "Linear Regression",
                    "Both"
                ]:

                    st.subheader("📈 Linear Regression")

                    lr = LinearRegression()

                    lr.fit(X_train, y_train)

                    pred = lr.predict(X_test)

                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)

                    st.success(f"R² Score: {r2:.4f}")

                    st.write(f"Mean Squared Error: {mse:.4f}")

            
                if model_option in [
                    "Random Forest Regressor",
                    "Both"
                ]:

                    st.subheader("🌲 Random Forest Regressor")

                    rf = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )

                    rf.fit(X_train, y_train)

                    pred = rf.predict(X_test)

                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)

                    st.success(f"R² Score: {r2:.4f}")

                    st.write(f"Mean Squared Error: {mse:.4f}")

        
        with st.expander("📁 View Full Dataset"):
            st.dataframe(data)

    except Exception as e:

        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV dataset.")