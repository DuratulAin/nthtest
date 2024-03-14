import streamlit as st
import pandas as pd
import joblib
import requests
import base64
import umap
from io import StringIO

# # Function to retrieve data from Xano and save it as a CSV file
# def retrieve_data():
#     xano_api_endpoint_bg = 'https://x8ki-letl-twmt.n7.xano.io/api:IAHLwzVr/bgdata'
#     payload_bg = {}
#     response_bg = requests.get(xano_api_endpoint_bg, params=payload_bg)

#     if response_bg.status_code == 200:
#         data_bg = response_bg.json()
#     else:
#         error_message = "Failed to retrieve data. Status code: " + str(response_bg.status_code)
#         st.error(error_message)
#         return None

#     xano_api_endpoint_spectral = 'https://x8ki-letl-twmt.n7.xano.io/api:IAHLwzVr/spectraldata'
#     payload_spectral = {}
#     response_spectral = requests.get(xano_api_endpoint_spectral, params=payload_spectral)

#     if response_spectral.status_code == 200:
#         data_spectral = response_spectral.json()
#     else:
#         error_message = "Failed to retrieve data. Status code: " + str(response_spectral.status_code)
#         st.error(error_message)
#         return None

    # Extract first line and convert to numeric
    df_bg = pd.DataFrame(data_bg).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df_spectral = pd.DataFrame(data_spectral).iloc[:1].apply(pd.to_numeric, errors='coerce')

    # Calculate absorbance
    absorbance = df_bg.div(df_spectral.values).pow(2)

    absorbance.to_csv('absorbanceData.csv', index=False)
    return absorbance

# Main Streamlit app
def main():
    # Retrieve data from Xano
    data_df = retrieve_data()


if __name__ == "__main__":
    main()

# Function to load a model from a pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model

# Streamlit UI elements
st.title('Model Prediction App')

# Load the CSV data from Xano
xano_data_df = pd.read_csv('absorbanceData.csv')

# Load the CSV data of original data
original_data = pd.read_csv('Raw data all w.csv')

# Combine both datas
combined_data = pd.concat([xano_data_df, original_data])

st.dataframe(xano_data_df)
st.dataframe(original_data)
st.dataframe(combined_data)

# Load the UMAP model from the joblib file
umap_model = load_model('umap_model_new_10.joblib').transform(combined_data)

# Load the Linear Regression model and make a prediction
linear_reg_model = load_model('linear_reg_model_new_10.joblib')
linear_reg_prediction = linear_reg_model.predict(combined_data)

# Load the Decision Tree model and make a prediction
decision_tree_model = load_model('decision_tree_model_new_10.joblib')
decision_tree_prediction = decision_tree_model.predict(combined_data)

# Load the Linear Regression model with UMAP and make prediction
linear_reg_model_umap = load_model('linear_reg_model_new_umap_10.joblib')
linear_reg_umap_pred = linear_reg_model_umap.predict(umap_model)

# Load the Decision Tree model with UMAP and make prediction
decision_tree_model_umap = load_model('decision_tree_model_new_umap_10.joblib')
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




# # Button to trigger prediction for both models
# if st.button('Predict'):
#     # Load the Linear Regression model and make a prediction
#     linear_reg_model = load_model('linear_reg_model_10.joblib')
#     linear_reg_prediction = linear_reg_model.predict(combined_data)

#     # Load the Decision Tree model and make a prediction
#     decision_tree_model = load_model('decision_tree_model_10.joblib')
#     decision_tree_prediction = decision_tree_model.predict(combined_data)

#     # Load the Linear Regression model with UMAP and make prediction
#     linear_reg_model_umap = load_model('linear_reg_model_umap_10.joblib')
#     linear_reg_umap_pred = linear_reg_model_umap.predict(umap_model)

#     # Load the Decision Tree model with UMAP and make prediction
#     decision_tree_model_umap = load_model('decision_tree_model_umap_10.joblib')
#     decision_tree_umap_pred = decision_tree_model_umap.predict(umap_model)

#     # Display predictions from both models in a larger and bold format
#     st.markdown('<font size="6"><b>Predictions:</b></font>', unsafe_allow_html=True)

#     st.markdown('**Linear Regression Model:**')
#     st.markdown(f'<font size="5"><b>{linear_reg_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)

#     st.markdown('**Decision Tree Model:**')
#     st.markdown(f'<font size="5"><b>{decision_tree_prediction[0]} g/dL</b></font>', unsafe_allow_html=True)

#     st.markdown('**Linear Regression Model with UMAP:**')
#     st.markdown(f'<font size="5"><b>{linear_reg_umap_pred[0]:.1f} g/dL</b></font>', unsafe_allow_html=True)

#     st.markdown('**Decision Tree Model with UMAP:**')
#     st.markdown(f'<font size="5"><b>{decision_tree_umap_pred[0]} g/dL</b></font>', unsafe_allow_html=True)
