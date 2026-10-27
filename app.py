
import streamlit as st
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_model.pkl")


try:
    artifact = load_model()

    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]
    model_name = artifact["model_name"]

except FileNotFoundError:
    st.error(
        "Trained model not found. "
        "Please run train_model.py first."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# 3. LOAD PUBLIC DATASET
# --------------------------------------------------

data = load_breast_cancer()


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("🎗️ Breast Cancer Prediction System")

st.markdown(
    """
    An educational machine learning application that uses
    numerical features from the Breast Cancer Wisconsin
    Diagnostic dataset to classify tumor samples.
    """
)

st.info(
    "Important: This application is for educational purposes "
    "only. It is not a medical diagnostic tool and must not "
    "be used to make medical decisions."
)

st.write(f"**Machine Learning Model:** {model_name}")

st.divider()


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.write("**Project:** Breast Cancer Prediction")

st.sidebar.write("**Algorithm:** " + model_name)

st.sidebar.write("**Dataset samples:** 569")

st.sidebar.write("**Numerical features:** 30")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built using Python, Scikit-learn, Pandas and Streamlit."
)


# --------------------------------------------------
# 6. SAMPLE SELECTION
# --------------------------------------------------

st.header("🧪 Test the Prediction Model")

st.write(
    "Choose a sample from the public dataset to test the "
    "application using actual numerical measurements."
)

sample_options = {
    "Sample 1": 0,
    "Sample 2": 19,
    "Sample 3": 100,
    "Sample 4": 200
}

selected_sample = st.selectbox(
    "Select a dataset sample",
    list(sample_options.keys())
)

sample_index = sample_options[selected_sample]

sample_values = data.data[sample_index]

actual_class = data.target_names[
    data.target[sample_index]
]


# --------------------------------------------------
# 7. DISPLAY INPUT MEASUREMENTS
# --------------------------------------------------

st.subheader("📋 Sample Measurements")

input_dataframe = pd.DataFrame({
    "Feature": feature_names,
    "Measurement": sample_values
})

st.dataframe(
    input_dataframe,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Measurements are taken from the selected public dataset sample."
)


# --------------------------------------------------
# 8. MAKE PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Selected Sample",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [sample_values],
        columns=feature_names
    )

    prediction = int(model.predict(input_data)[0])

    predicted_class = target_names[prediction]

    st.divider()

    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Class",
            value=predicted_class.title()
        )

    with col2:
        st.metric(
            label="Dataset Label",
            value=actual_class.title()
        )

    if predicted_class == actual_class:
        st.success(
            "The model prediction matches this dataset sample's label."
        )
    else:
        st.warning(
            "The model prediction differs from this dataset sample's label."
        )

    # ----------------------------------------------
    # 9. PREDICTION PROBABILITIES
    # ----------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        probability_dataframe = pd.DataFrame({
            "Class": target_names,
            "Probability (%)": probabilities * 100
        })

        st.subheader("Prediction Probability")

        st.bar_chart(
            probability_dataframe.set_index("Class")
        )

        st.dataframe(
            probability_dataframe.style.format(
                {"Probability (%)": "{:.2f}%"}
            ),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "These probabilities are model outputs, not verified "
        "clinical probabilities. Agreement with a dataset label "
        "does not establish medical reliability."
    )


# --------------------------------------------------
# 10. PROJECT FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Breast Cancer Prediction Using Machine Learning | "
    "Educational prototype"
)