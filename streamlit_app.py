import streamlit as st
import pandas as pd
import joblib

# Function to load a model from a pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model

# Streamlit UI elements
st.title('Model Prediction App')

# Load sample data from a CSV file
sample_data = pd.read_csv('Raw spectral 1.csv')

# Print out the raw spectral CSV where the first row contains wavelength and the second row contains spectral value
st.write('Raw spectral CSV:')
st.table(sample_data)

# Load the UMAP model from the joblib file
umap_model = load_model('umap_model_10.joblib').transform(sample_data)

# Button to trigger prediction for both models
if st.button('Predict'):
    # Load the Linear Regression model and make a prediction
    linear_reg_model = load_model('linear_reg_model_10.joblib')
    linear_reg_prediction = linear_reg_model.predict(sample_data)

    # Load the Decision Tree model and make a prediction
    decision_tree_model = load_model('decision_tree_model_10.joblib')
    decision_tree_prediction = decision_tree_model.predict(sample_data)

    # Load the Linear Regression model with UMAP and make prediction
    linear_reg_model_umap = load_model('linear_reg_model_umap_10.joblib')
    linear_reg_umap_pred = linear_reg_model_umap.predict(umap_model)

    # Load the Decision Tree model with UMAP and make prediction
    decision_tree_model_umap = load_model('decision_tree_model_umap_10.joblib')
    decision_tree_umap_pred = decision_tree_model_umap.predict(umap_model)

    # Display predictions from both models in a larger and bold format
    st.markdown('<font size="6"><b>Predictions:</b></font>', unsafe_allow_html=True)

    st.markdown('**Linear Regression Model:**')
    st.markdown(f'<font size="5"><b>{linear_reg_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)

    st.markdown('**Decision Tree Model:**')
    st.markdown(f'<font size="5"><b>{decision_tree_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)

    st.markdown('**Linear Regression Model with UMAP:**')
    st.markdown(f'<font size="5"><b>{linear_reg_umap_pred[0]:.1f} g/dL</b></font>', unsafe_allow_html=True)

    st.markdown('**Decision Tree Model with UMAP:**')
    st.markdown(f'<font size="5"><b>{decision_tree_umap_pred[0]} g/dL</b></font>', unsafe_allow_html=True)
