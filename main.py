import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def load_model():
    """
    Load the model from a pickle file.
    """
    with open("Student_lr_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess_data(data, scaler, le):
    """
    Preprocess the data using the provided scaler and label encoder.
    """
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict(data):
    model, scaler, le = load_model()
    preprocessed_data = preprocess_data(data, scaler, le)   
    prediction = model.predict(preprocessed_data)
    return prediction

def main():
    # Set page configuration
    st.set_page_config(page_title="Student Performance Prediction", page_icon="📚", layout="wide")

    # Add a header with a custom style
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-header">📚 Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict the performance of students based on various features.</div>', unsafe_allow_html=True)

    # Sidebar for input
    st.sidebar.header("Input Features")
    st.sidebar.write("Provide the following details:")

    hour_studied = st.sidebar.number_input("📖 Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.sidebar.number_input("📊 Previous Scores", min_value=40, max_value=100, value=75)
    extra = st.sidebar.selectbox("🎭 Extra Curricular Activities", ["Yes", "No"])
    sleeping_hours = st.sidebar.number_input("💤 Sleep Hours", min_value=6, max_value=10, value=8)
    number_of_paper_solved = st.sidebar.number_input("📝 Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    # Main content
    st.markdown("### Prediction Results")
    st.write("Click the button below to predict the student's performance:")

    if st.button("🎯 Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hours,
            "Sample Question Papers Practiced": number_of_paper_solved
        }

        prediction = predict(user_data)
        st.success(f"🎉 Predicted Score: {prediction[0]}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 14px;">
        Made with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()