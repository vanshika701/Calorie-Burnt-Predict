import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

st.set_page_config(page_title="Calories Burnt Predictor")
st.title("🔥 Calories Burnt Prediction App")

# Upload CSV files
st.sidebar.header("Upload Your Data Files")
calories_file = st.sidebar.file_uploader("Upload calories.csv", type=['csv'])
exercise_file = st.sidebar.file_uploader("Upload exercise.csv", type=['csv'])

if calories_file and exercise_file:
    calories = pd.read_csv(calories_file)
    exercise_data = pd.read_csv(exercise_file)

    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

    st.subheader("📊 Data Overview")
    st.write(calories_data.head())

    # Data visualizations
    st.subheader("📈 Data Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='Gender', data=calories_data, ax=ax1)
    st.pyplot(fig1)

    for column in ['Age', 'Height', 'Weight']:
        fig, ax = plt.subplots()
        sns.histplot(calories_data[column], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    correlation = calories_data.select_dtypes(include='number').corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Encoding categorical values
    calories_data.replace({"Gender": {"male": 0, "female": 1}}, inplace=True)

    X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = calories_data['Calories']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Model training
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Evaluation
    predictions = model.predict(X_test)
    score = metrics.r2_score(Y_test, predictions)
    st.success(f"Model Trained! R^2 Score: {score:.2f}")

    # Prediction interface
    st.subheader("🧮 Predict Calories Burnt")
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.slider("Age", 10, 100, 25)
    height = st.slider("Height (cm)", 100, 250, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)
    duration = st.slider("Duration of Exercise (min)", 1, 120, 30)
    heart_rate = st.slider("Heart Rate", 60, 200, 100)
    body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

    if st.button("Predict"):
        input_data = np.array([[0 if gender == "male" else 1, age, height, weight, duration, heart_rate, body_temp]])
        calories_burnt = model.predict(input_data)
        st.success(f"Estimated Calories Burnt: {calories_burnt[0]:.2f} kcal")
else:
    st.info("Please upload both 'calories.csv' and 'exercise.csv' files to proceed.")
