import streamlit as st
import streamlit.components.v1 as stc
import pickle
from sklearn.preprocessing import RobustScaler
import numpy as np

# Load the pre-trained model from the correct path
with open('lr_noscld_tuned_model.pkl', 'rb') as file:
    lr_tuned = pickle.load(file)

# Define a function to render the header HTML
def render_header():
    html_temp = """
    <div style="background-color:#78bce4;padding:20px 40px;border-radius:15px;display:flex;align-items:center;justify-content:center;position:relative;width:100%;box-sizing:border-box;">
        <div style="text-align: center; flex: 1;">
            <h1 style="color:#fff;margin:0;text-shadow: 3px 3px 6px #000, 0 0 25px #000, 0 0 5px #000;">DIABETEST</h1>
            <h4 style="color:#fff;margin:0;text-shadow: 2px 2px 4px #000, 0 0 10px #000, 0 0 3px #000;">Made for: Staff Team</h4>
        </div>
        <div style="position:absolute;left:20px;bottom:10px;font-size:8vw;max-width:80px;">
            <span style="font-size:60px;">🏥</span>
        </div>
    </div>"""
    stc.html(html_temp, height=150)

# Header and description HTML
desc_temp = """
### Diabetes Prediction App
This app is used by the Hospital Team for predicting diabetes risk.
<br>
### Data Source
Kaggle: <a href="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data">Link</a>
<br>
### Importance of Health

**Why You Should Predict Your Diabetes?** <br>
It's crucial to check your potential risk for diabetes because this condition can develop gradually without clear symptoms in its early stages. By using this app to evaluate your health factors, such as Body Mass Index (BMI), blood pressure, and cholesterol levels, you gain an early insight into your risk for diabetes. Early detection allows you to take preventive measures or seek treatment sooner, which can help manage or even prevent the progression of diabetes. With a better understanding of your risk, you can make informed decisions about lifestyle and health habits, and consult with medical professionals for further steps to maintain your overall health.

## How to Use It?
1. **Go to the Sidebar**: Locate the sidebar on the left side of the screen.
2. **Open the Dropdown Menu**: Click on the dropdown menu (selectbox) in the sidebar.
3. **Select 'Prediction App'**: Choose the "Prediction App" option from the dropdown menu.
4. **Input Your Details**: Enter the required information in the fields provided in the Prediction App.
"""

def main():
    menu = ["Home", "Prediction App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        render_header()  # Render header HTML on the Home page
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Prediction App":
        run_ml_app()

def run_ml_app():
    # Add CSS styling to the app
    st.markdown("""
    <style>
    .prediction-app-container {
        background-color: #e0f7fa; /* Light blue background */
        border: 2px solid #007bb2; /* Dark blue outline */
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Add HTML structure with the custom class
    design = """
    <div class="prediction-app-container">
        <div style="padding:15px;text-align:center;">
            <h1 style="color:#000;">Diabetes Prediction</h1>
        </div>
    </div>"""
    st.markdown(design, unsafe_allow_html=True)
    
    # Use responsive columns
    left, right = st.columns(2)
    GenHlth = left.number_input('General Health (Scale 1-5) (Good to Bad)', min_value=1, max_value=5)
    BMI = right.number_input('Body Mass Index [Check here](https://www.calculator.net/bmi-calculator.html?cage=25&csex=m&cheightfeet=5&cheightinch=10&cpound=160&cheightmeter=180&ckg=65&printit=0&ctype=metric)', min_value=15, max_value=45)
    Age = left.number_input('Age', min_value=18, max_value=200)
    HighBP = right.selectbox('High Blood Pressure', ('Yes', 'No'))
    HighChol = left.selectbox('High Cholesterol', ('Yes', 'No'))
    HeartDiseaseorAttack = right.selectbox('Heart Attack History', ('Yes', 'No'))
    DiffWalk = left.selectbox('Difficulty of walking', ('Yes', 'No'))
    Stroke = right.selectbox('Stroke History', ('Yes', 'No'))
    PhysHlth = left.number_input('Physical Health: I have been feeling sick since __ days ago.(0 if no pain at all).', min_value=0, max_value=30)

    button = st.button("Predict")

    if button:
        result = predict(BMI, Age, HighChol, HighBP, GenHlth, PhysHlth, DiffWalk, Stroke, HeartDiseaseorAttack)

        if result == 'Dont':
            st.success(f'You {result} have a potential risk of diabetes!')
        else:
            st.warning(f'You {result} have a potential risk of diabetes.')
            # Embed YouTube video
            st.video("https://www.youtube.com/watch?v=1Q0ftaFPxlk")
            
            # Ask for phone number and send notification

def predict(BMI, Age, HighChol, HighBP, GenHlth, PhysHlth, DiffWalk, Stroke, HeartDiseaseorAttack):
    age_category = categorize_age(Age)
    highbp = 1 if HighBP == 'Yes' else 0
    highchol = 1 if HighChol == 'Yes' else 0
    diffWalk = 1 if DiffWalk == 'Yes' else 0
    stroke = 1 if Stroke == 'Yes' else 0
    hd = 1 if HeartDiseaseorAttack == 'Yes' else 0

    # Scale the features BMI and PhysHlth
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform([[BMI, PhysHlth]])
    BMI_scaled = scaled_features[0][0]
    PhysHlth_scaled = scaled_features[0][1]

    # Making prediction
    prediction = lr_tuned.predict([[GenHlth, BMI_scaled, age_category, highbp, highchol, diffWalk, stroke, hd, PhysHlth_scaled]])
    result = 'Dont' if prediction == 0 else 'Have'
    return result

def categorize_age(Age):
    if 18 <= Age <= 24:
        return 1
    elif 25 <= Age <= 29:
        return 2
    elif 30 <= Age <= 34:
        return 3
    elif 35 <= Age <= 39:
        return 4
    elif 40 <= Age <= 44:
        return 5
    elif 45 <= Age <= 49:
        return 6
    elif 50 <= Age <= 54:
        return 7
    elif 55 <= Age <= 59:
        return 8
    elif 60 <= Age <= 64:
        return 9
    elif 65 <= Age <= 69:
        return 10
    elif 70 <= Age <= 74:
        return 11
    elif 75 <= Age <= 79:
        return 12
    else:
        return 13


if __name__ == "__main__":
    main()
