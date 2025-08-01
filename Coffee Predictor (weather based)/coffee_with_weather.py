import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import speech_recognition as sr

# Page setup
st.set_page_config(page_title="Coffee Predictor", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #111111;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #333333;
        color: white;
    }
    .stSelectbox {
        border-radius: 10px;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Coffee Purchase Predictor")
st.write("Predict whether a person will buy coffee based on conditions.")

# Load dataset
df = pd.read_csv("coffee_data_with_days.csv")
df_model = df.drop("Day", axis=1)

# Encode categorical features
encoders = {}
for col in df_model.columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# Model training
X = df_model.drop("BuyCoffee", axis=1)
y = df_model["BuyCoffee"]
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Voice recognition function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized: {text}")
        return text.capitalize().strip()
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError:
        st.error("Speech Recognition service error.")
    return None

# Initialize session state
for field in ["Weather", "TimeOfDay", "SleepQuality", "Mood"]:
    if f"{field}_voice_value" not in st.session_state:
        st.session_state[f"{field}_voice_value"] = None

# Layout and inputs
st.subheader("🎛️ Choose or Confirm Values")
selected_inputs = {}

for field in ["Weather", "TimeOfDay", "SleepQuality", "Mood"]:
    col1, col2 = st.columns([3, 1])

    with col1:
        default_val = (
            st.session_state[f"{field}_voice_value"]
            if st.session_state[f"{field}_voice_value"]
            else encoders[field].classes_[0]
        )
        selected_val = st.selectbox(
            field,
            encoders[field].classes_,
            index=list(encoders[field].classes_).index(default_val),
            key=f"{field}_input"
        )
        selected_inputs[field] = selected_val

    with col2:
        if st.button(f"🎤 Speak for {field}", key=f"{field}_btn"):
            value = recognize_speech()
            if value in encoders[field].classes_:
                st.session_state[f"{field}_voice_value"] = value
                st.rerun()
            else:
                st.warning(f"'{value}' not recognized for {field}")

# Prediction
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([[selected_inputs["Weather"],
                               selected_inputs["TimeOfDay"],
                               selected_inputs["SleepQuality"],
                               selected_inputs["Mood"]]],
                            columns=["Weather", "TimeOfDay", "SleepQuality", "Mood"])
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    result = encoders["BuyCoffee"].inverse_transform([prediction])[0]

    if result == "Yes":
        st.success("✅ You're likely to buy coffee. Enjoy your cup! ☕")
        st.markdown("**\"A yawn is a silent scream for coffee.\"** 😴 → ☕")
        st.markdown("**\"Life begins after coffee.\"** 🌅")
    else:
        st.info("ℹ️ You're unlikely to buy coffee right now.")
        st.markdown("**\"No coffee now? Maybe you're already fueled by good vibes.✨\"**")
        st.markdown("**\"Skip the cup, sip the moment.💫\"**")

# Decision Tree Plot
st.subheader("📊 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=encoders["BuyCoffee"].classes_, filled=True)
st.pyplot(fig)
