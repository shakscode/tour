import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
# import re # Removed, as snake_case conversion is unnecessary

# =============================
# Set Streamlit Page Configuration (for a clean look and wide mode)
# =============================
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar for instructions/info
)

# =============================
# Streamlit Custom Styling (Dark Theme & Colors)
# =============================
# This CSS block is injected to customize the appearance,
# enhancing the dark theme with vibrant prediction colors.
st.markdown("""
<style>
    /* Main body background color is handled by Streamlit's dark theme */
    /* Enhance the main title */
    .stApp > header {
        background-color: transparent;
    }

    h1 {
        color: #FF6347; /* Tomato red for the main title */
        font-weight: 700;
        text-shadow: 2px 2px 4px #000000;
    }

    /* Subheader for prediction result */
    .stSuccess > div {
        background-color: #28a745 !important; /* Darker green for success background */
        color: white !important;
        border-radius: 10px;
        padding: 20px;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button {
        background-color: #007bff; /* Bright blue for the button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        border-color: #0056b3;
        box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Input fields styling for better dark theme contrast */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div, .stSlider>div>div>div {
        background-color: #1e1e1e; /* Slightly lighter dark background for contrast */
        border: 1px solid #444444;
        color: white;
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)


# =============================
# Load the trained model
# =============================
# Use a spinner while loading
with st.spinner('Loading model...'):
    try:
        # Replace with your repo id where model is uploaded
        model_path = hf_hub_download(
            repo_id="ShaksML/tourism",
            filename="top_tourism_model_v1.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        st.sidebar.success("Model loaded successfully! 🎉")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


# =============================
# Streamlit UI - Title and Description
# =============================
st.title("✈️ Tourism Package Prediction App")

# Add a concise description to the sidebar
st.sidebar.header("About the App")
st.sidebar.markdown("""
This application uses a pre-trained machine learning model to predict whether a customer is likely to **purchase a tourism package** based on their personal and behavioral data.

---
**Instructions:**
1. Enter the customer details using the input fields below.
2. Click the **'Predict Purchase'** button.
3. The result will appear at the bottom.
""")


st.markdown("### Enter Customer Details")
st.markdown("""
Please enter the customer's personal, financial, and interaction details across the two sections below to get a prediction.
""")

# -----------------------------
# User Inputs - Layout with st.columns
# -----------------------------

# Use st.container for a nice grouping of inputs
with st.container(border=True):
    col1, col2 = st.columns(2)

    # --- Column 1: Personal & General Details ---
    with col1:
        st.markdown("#### 👤 Personal & General")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, key='age')
        gender = st.selectbox("Gender", ["Male", "Female"], key='gender')
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key='marital_status')
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], key='occupation')
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], key='designation')
        passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='passport')
        own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key='own_car')

    # --- Column 2: Package & Financial Details ---
    with col2:
        st.markdown("#### 💰 Financial & Package")
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000, key='monthly_income')
        city_tier = st.selectbox("City Tier", [1, 2, 3], key='city_tier')
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], key='product_pitched')
        preferred_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5], key='preferred_star')
        duration_pitch = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=60, value=10, key='duration_pitch')
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"], key='typeof_contact')
        pitch_score = st.slider("Pitch Satisfaction Score (1=Low, 5=High)", 1, 5, 3, key='pitch_score')

    # --- Column for counts (Spans across full width) ---
    st.markdown("#### 👨‍👩‍👧‍👦 Travel Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, key='num_persons')
    with col4:
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0, key='num_children')
    with col5:
        num_trips = st.number_input("Number of Previous Trips", min_value=0, max_value=50, value=5, key='num_trips')

    st.markdown("#### 📞 Follow-up Details")
    num_followups = st.number_input("Number of Follow-ups Made", min_value=0, max_value=10, value=2, key='num_followups')


# Assemble input data into DataFrame (using PascalCase names as required by the expected features list)
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact, # Note: using type_of_contact from st.selectbox
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}])


# -----------------------------
# Prediction Button and Logic
# -----------------------------
# Center the button using columns
st.markdown("---")
pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])

with pred_col2:
    if st.button("Predict Purchase", use_container_width=True):
        try:
            # --- FIX: Apply One-Hot Encoding and Column Alignment (UNCHANGED FUNCTIONALITY) ---

            # 1. Apply One-Hot Encoding and ensure original categorical columns are kept.
            input_dummies = pd.get_dummies(input_data, drop_first=False)
            
            # Start with the dummified features (which include the numeric features)
            input_data_processed = input_dummies.copy()
            
            # Manually add the original categorical columns back from input_data
            categorical_cols_to_restore = [
                'Designation', 'ProductPitched', 'MaritalStatus', 
                'TypeofContact', 'Gender', 'Occupation'
            ]
            
            for col in categorical_cols_to_restore:
                # Add the original column to the processed data
                input_data_processed[col] = input_data[col]


            # 2. Define the full list of features the model was trained on
            expected_features = [
                # Numeric/Ordinal Features (12)
                'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
                'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',

                # Original Categorical Features (6, which the model is demanding)
                'Designation', 'ProductPitched', 'MaritalStatus', 'TypeofContact', 'Gender', 'Occupation',

                # Categorical Features (21 dummified columns)
                'TypeofContact_Company Invited', 'TypeofContact_Self Enquiry',
                'Occupation_Salaried', 'Occupation_Small Business', 'Occupation_Large Business', 'Occupation_Free Lancer',
                'Gender_Male', 'Gender_Female',
                'ProductPitched_Basic', 'ProductPitched_Standard', 'ProductPitched_Deluxe',
                'ProductPitched_Super Deluxe', 'ProductPitched_King',
                'MaritalStatus_Single', 'MaritalStatus_Married', 'MaritalStatus_Divorced',
                'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager',
                'Designation_AVP', 'Designation_VP'
            ]

            # 3. Add any missing dummified columns (set to 0) to ensure feature completeness
            for col in expected_features:
                if col not in input_data_processed.columns:
                    input_data_processed[col] = 0

            # 4. Reorder columns to match the training order exactly
            input_data_final = input_data_processed[expected_features]

            # Predict using the properly structured DataFrame
            prediction = model.predict(input_data_final)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.balloons()
                st.success(f"🎉 The model predicts: **Will Purchase Package** (Prediction: {prediction})")
            else:
                st.error(f"😔 The model predicts: **Will Not Purchase Package** (Prediction: {prediction})")


        except Exception as e:
            st.exception(f"An error occurred during prediction: {e}")
