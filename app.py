"""Interactive dashboard for the simulated smart-home sensor."""

import random
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "smart_home_model.joblib"
FEATURES = ["temperature_c", "humidity_percent"]

st.set_page_config(
    page_title="Smart Home IoT Monitor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def predict_reading(model, temperature, humidity):
    sample = pd.DataFrame(
        [[temperature, humidity]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(sample)[0]
    confidence_index = probabilities.argmax()
    return {
        "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "room_status": model.classes_[confidence_index],
        "confidence": float(probabilities[confidence_index]),
    }


def record_reading(reading, source):
    record = {**reading, "source": source}
    st.session_state.readings.append(record)
    st.session_state.latest_reading = record


if not MODEL_PATH.is_file():
    st.error(
        "The trained model was not found. Run `python train_model.py` "
        "from any directory, then refresh this page."
    )
    st.stop()

model = load_model(str(MODEL_PATH))
if list(getattr(model, "feature_names_in_", [])) != FEATURES:
    st.error(
        "The saved model does not match this dashboard's expected input "
        "features. Run `python train_model.py` to rebuild it."
    )
    st.stop()

if "readings" not in st.session_state:
    st.session_state.readings = []
if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = None

st.title("🏠 Smart Home IoT Monitor")
st.write(
    "Explore simulated temperature and humidity readings and classify "
    "the room with a locally trained decision tree."
)
st.caption(
    "Educational demonstration only. No physical sensor is connected, "
    "and predictions are not a certified comfort or safety standard."
)

input_column, result_column = st.columns([1, 1], gap="large")
with input_column:
    st.subheader("Room conditions")
    temperature = st.slider(
        "Temperature (°C)",
        min_value=10.0,
        max_value=42.0,
        value=24.0,
        step=0.5,
    )
    humidity = st.slider(
        "Humidity (%)",
        min_value=20,
        max_value=95,
        value=50,
        step=1,
    )
    predict_clicked = st.button(
        "Predict room status",
        type="primary",
        width="stretch",
    )
    st.subheader("Sensor simulation")
    if st.button("Generate simulated sensor reading", width="stretch"):
        simulated_reading = predict_reading(
            model,
            round(random.uniform(15, 36), 1),
            random.randint(25, 90),
        )
        record_reading(simulated_reading, "Simulated")

if predict_clicked:
    reading = predict_reading(model, temperature, humidity)
    record_reading(reading, "Manual")

with result_column:
    st.subheader("Latest prediction")
    latest = st.session_state.latest_reading
    if latest is None:
        st.info("Choose room conditions and predict, or generate a simulated reading.")
    else:
        metric_columns = st.columns(3)
        metric_columns[0].metric("Temperature", f"{latest['temperature_c']:.1f} °C")
        metric_columns[1].metric("Humidity", f"{latest['humidity_percent']}%")
        metric_columns[2].metric("Confidence", f"{latest['confidence']:.0%}")

        status = latest["room_status"]
        if status == "Comfortable":
            st.success(f"Room status: **{status}**")
        elif status == "Too Hot":
            st.warning(f"Room status: **{status}**")
        else:
            st.info(f"Room status: **{status}**")
        st.caption(f"{latest['source']} reading · {latest['time']}")

st.divider()
st.subheader("Reading history")
if not st.session_state.readings:
    st.write("Predictions and simulated readings from this browser session will appear here.")
else:
    history = pd.DataFrame(st.session_state.readings)
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.caption("Temperature over time")
        st.line_chart(history.set_index("time")[["temperature_c"]], height=220)
    with chart_columns[1]:
        st.caption("Humidity over time")
        st.line_chart(history.set_index("time")[["humidity_percent"]], height=220)

    st.dataframe(
        history.sort_index(ascending=False),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download reading history as CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="smart_home_reading_history.csv",
        mime="text/csv",
    )

with st.expander("About this model"):
    st.write("**Algorithm:** Decision Tree Classifier")
    st.write("**Inputs:** Temperature (°C) and relative humidity (%)")
    st.write("**Possible labels:** Comfortable, Too Hot, Too Cold")
    st.write(
        "The classifier was trained on synthetic data with illustrative "
        "labels. Its predictions may not represent real-world conditions."
    )
