import streamlit as st
import pandas as pd
import joblib
import requests
import base64
from io import StringIO

# Function to retrieve spectral data from Xano and save it as a CSV file
def spectral_data():
    xano_api_endpoint = 'https://x8ki-letl-twmt.n7.xano.io/api:U4wk_Gn6/spectral_data'
    payload = {}

    response = requests.get(xano_api_endpoint, params=payload)

    if response.status_code == 200:
        data = response.json()

        # Convert the Xano data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Save only the first row as a CSV file
        df.iloc[:1].to_csv('spectral_data.csv', index=False)

        return df.iloc[:1]  # Return only the first row
    else:
        st.error("Failed to retrieve data. Status code:", response.status_code)
        return None

# Function to retrieve background data from Xano and save it as a CSV file
def bg_data():
    xano_bg_api_endpoint = 'https://x8ki-letl-twmt.n7.xano.io/api:U4wk_Gn6/BackgroundReading'
    payload = {}

    response = requests.get(xano_bg_api_endpoint, params=payload)

    if response.status_code == 200:
        data_bg = response.json()

        # Convert the Xano data to a pandas DataFrame
        df_bg = pd.DataFrame(data_bg)

        # Save only the first row as a CSV file
        df_bg.iloc[:1].to_csv('background_data.csv', index=False)

        return df_bg.iloc[:1]  # Return only the first row
    else:
        st.error("Failed to retrieve data. Status code:", response.status_code)
        return None
        
# Main Streamlit app
def main():

    # Retrieve data from Xano
    data_df = spectral_data()

    # Retrieve data from Xano
    data_df_bg = bg_data()

# Function to load a model from a pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model

if __name__ == "__main__":
    main()

# Streamlit UI elements
st.title('Model Prediction App')

# Load the CSV data from Xano
xano_data_df = pd.read_csv('spectral_data.csv')

# Load the CSV data of original data
original_data = pd.read_csv('Raw data all w.csv')

# Combine both datas
combined_data = pd.concat([xano_data_df.iloc[:1], original_data])

st.dataframe(xano_data_df)
st.dataframe(background_data_df)
st.dataframe(original_data)
st.dataframe(combined_data)


# Load the UMAP model from the joblib file
umap_model = load_model('umap_model_10.joblib').transform(combined_data)

# Load the Linear Regression model and make a prediction
linear_reg_model = load_model('linear_reg_model_10.joblib')
linear_reg_prediction = linear_reg_model.predict(combined_data)

# Load the Decision Tree model and make a prediction
decision_tree_model = load_model('decision_tree_model_10.joblib')
decision_tree_prediction = decision_tree_model.predict(combined_data)

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
