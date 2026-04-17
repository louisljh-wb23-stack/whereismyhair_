import streamlit as st
import time
import pandas as pd
import joblib

# =========================
# LOAD MODEL (Random Forest only)
# =========================
model = joblib.load("rf2.joblib")

# =========================
# INIT STATE
# =========================
if "started" not in st.session_state:
    st.session_state.started = False

# =========================
# START PAGE CSS
# =========================
start_css = """
<style>

html, body {
    height: 100%;
    margin: 0;
    overflow: hidden;
}

.stApp {
    background-image: url("https://raw.githubusercontent.com/louisljh-wb23-stack/online_gamj/326d080c022ad5d8648e064b01dda026c446aba9/unicorn.png");
    background-size: 45%;
    background-repeat: no-repeat;
    background-position: center;
}

.center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}

.stButton > button {
    background: linear-gradient(45deg, #A4DE02, #8BC34A);
    color: black;
    padding: 20px 85px;
    font-size: 24px;
    border-radius: 50px;
    border: none;
}

</style>
"""

# =========================
# MAIN CSS
# =========================
main_css = """
<style>
.stApp {
    background: none !important;
}
</style>
"""

# =========================
# START SCREEN
# =========================
if not st.session_state.started:

    st.markdown(start_css, unsafe_allow_html=True)

    st.markdown('<div class="center">', unsafe_allow_html=True)

    st.title("👾 Press Start")

    if st.button("START"):
        time.sleep(0.2)
        st.session_state.started = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================
else:

    st.markdown(main_css, unsafe_allow_html=True)

    st.title("Tell me about you")

    # =========================
    # INPUT FEATURES (5-digit limit)
    # =========================
    age = st.number_input("Age", 0, 99999, 20, step=1)
    playtime = st.number_input("Play Time Hours", 0, 99999, 10, step=1)
    sessions = st.number_input("Sessions Per Week", 0, 99999, 5, step=1)
    achievements = st.number_input("Achievements Unlocked", 0, 99999, 10, step=1)
    level = st.number_input("Player Level", 0, 99999, 1, step=1)
    duration = st.number_input("Avg Session Duration", 0, 99999, 30, step=1)

    FEATURES = [
        "Age",
        "PlayTimeHours",
        "SessionsPerWeek",
        "AchievementsUnlocked",
        "PlayerLevel",
        "AvgSessionDurationMinutes"
    ]

    # =========================
    # PREDICT
    # =========================
    if st.button("Predict"):

        input_data = pd.DataFrame([[
            age,
            playtime,
            sessions,
            achievements,
            level,
            duration
        ]], columns=FEATURES)

        pred = model.predict(input_data)[0]

        label_map = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        label = label_map.get(pred, pred)

        st.success(f"🎯 Prediction: {label}")

        # =========================
        # PROBABILITY
        # =========================
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]

            st.write("Probability:")
            st.write({
                "Low": proba[0],
                "Medium": proba[1],
                "High": proba[2]
            })

    # =========================
    # BACK BUTTON
    # =========================
    if st.button("Back"):
        st.session_state.started = False
        st.rerun()
