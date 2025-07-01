# 🔥 Calories Burnt Prediction App

This Streamlit web application predicts the number of calories burnt during exercise based on user inputs like age, gender, height, weight, heart rate, exercise duration, and body temperature. It uses an XGBoost regression model trained on real-world exercise and calorie data.

---
👉 [Click here to try the app](https://calorie-burnt-predict-jwqshdbz2bb8pxraqx26pr.streamlit.app/)  
<!-- Replace the # with your deployed app link -->
## 📁 Files Required


Before running the app, make sure you have the following files:

- `exercise.csv` — contains user demographic and exercise details
- `calories.csv` — contains calories burnt corresponding to each user ID

---

## 🚀 How to Run

1. **Clone this repository** or place the app code (`app.py`) in your project folder.

2. **Install the dependencies** (preferably in a virtual environment):

   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Upload the required CSV files using the sidebar once the app launches in your browser.

📊 Features
Exploratory Data Analysis:

Data preview

Gender distribution chart

Histograms of Age, Height, and Weight

Correlation heatmap of numerical features

Model Training:

Uses XGBoost Regressor

Displays R² score on test data

Calories Prediction:

Enter values for gender, age, height, weight, duration, heart rate, and body temperature

Predicts estimated calories burnt

🧠 Model Details
Model Used: XGBRegressor from xgboost

Train-Test Split: 80-20

Target Variable: Calories

Features Used: All user and exercise-related columns except User_ID

💡 Sample Input Fields in UI
Gender: male / female

Age: 10–100

Height (cm): 100–250

Weight (kg): 30–200

Duration (minutes): 1–120

Heart Rate: 60–200

Body Temperature (°C): 35.0–42.0

📌 Note
Make sure to upload both CSV files together — the app will not run if either is missing.

Gender is automatically encoded as 0 (male) and 1 (female) during preprocessing.

🛡️ License
This project is for educational purposes only.

🙌 Acknowledgements
Streamlit

XGBoost

Sample datasets from open-source calorie tracking projects
