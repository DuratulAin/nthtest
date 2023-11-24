import streamlit as st
import pandas as pd
import joblib
import json
from flask import request 
from streamlit.web.server.server import Server
import requests

# Function to retrieve data from Xano
def retrieve_data():
    xano_api_endpoint = 'https://x8ki-letl-twmt.n7.xano.io/api:U4wk_Gn6/data'

    response = requests.get(xano_api_endpoint)

    if response.status_code == 200:
        data = response.json()

        # Save the JSON data to a file
        with open('data.json', 'w') as f:
            json.dump(data, f)

        return data
    else:
        st.error("Failed to retrieve data. Status code:", response.status_code)
        return None

# Main Streamlit app
def main():
    st.title("Streamlit App")

    # Retrieve data from Xano
    data = retrieve_data()

    # Display the retrieved data
    if data:
        st.write("Retrieved Data:", data)

if __name__ == "__main__":
    main()

 
# # Enable CORS
# Server.enableCORS = True

# def enable_cors(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     return response

# @st.experimental_singleton  
# def receive_data(sendstreamlit):
#     data_flutter = request.get_json(sendstreamlit) 
#     # Do something with data
#     st.write(data)

# flutterflow_data = receive_data()
# print(flutterflow_datar)

# Function to load a model from a pickle file
def load_model(model_file):
  with open(model_file, 'rb') as f:
    model = joblib.load(f)
  return model

# Streamlit UI elements
st.title('Model Prediction App')

# Load the JSON data
with open('Raw data all.json') as jsonfile:
  data = json.load(jsonfile)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Print out the raw spectral CSV where the first row contains wavelength and the second row contains spectral value
#st.write('Raw spectral CSV:')
#st.table(sample_data)

# Load the UMAP model from the joblib file
umap_model = load_model('umap_model_10.joblib').transform(df)

# Button to trigger prediction for both models
if st.button('Predict'):
  # Load the Linear Regression model and make a prediction
  linear_reg_model = load_model('linear_reg_model_10.joblib')
  linear_reg_prediction = linear_reg_model.predict(df)

  # Load the Decision Tree model and make a prediction
  decision_tree_model = load_model('decision_tree_model_10.joblib')
  decision_tree_prediction = decision_tree_model.predict(df)

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
