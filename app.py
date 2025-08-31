import numpy as np
import pickle
import streamlit as st
import base64
from PIL import Image
from xgboost import XGBRegressor

# Page config
st.set_page_config(page_title="Big Mart Sales Prediction", layout="wide")

# Hide Streamlit header and menu
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
.main-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 10px;
    margin: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Load and encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("istockphoto-968898244-612x612.jpg")

# Apply background
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)

# Load model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title('🛒 Big Mart Sales Prediction')
st.write('Enter the product details to predict sales:')

# Create input fields in columns
col1, col2 = st.columns(2)

with col1:
    Item_Weight = st.number_input('Item Weight', min_value=0.0, value=9.30)
    Item_Fat_Content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
    Item_Visibility = st.number_input('Item Visibility', min_value=0.0, max_value=1.0, value=0.016)
    Item_Type = st.selectbox('Item Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'])
    Item_MRP = st.number_input('Item MRP', min_value=0.0, value=249.81)
    
with col2:
    Outlet_Establishment_Year = st.number_input('Outlet Establishment Year', min_value=1985, max_value=2009, value=1999)
    Outlet_Size = st.selectbox('Outlet Size', ['Medium', 'Small', 'High'])
    Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

# Prediction
if st.button('Predict Sales', type='primary'):
    # Encode categorical features
    fat_content_map = {'Low Fat': 0, 'Regular': 1}
    item_type_map = {'Dairy': 0, 'Soft Drinks': 1, 'Meat': 2, 'Fruits and Vegetables': 3, 'Household': 4, 'Baking Goods': 5, 'Snack Foods': 6, 'Frozen Foods': 7, 'Breakfast': 8, 'Health and Hygiene': 9, 'Hard Drinks': 10, 'Canned': 11, 'Breads': 12, 'Starchy Foods': 13, 'Others': 14, 'Seafood': 15}
    outlet_size_map = {'Medium': 0, 'Small': 1, 'High': 2}
    outlet_location_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
    outlet_type_map = {'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2, 'Grocery Store': 3}
    
    # Prepare input data with encoded values
    input_data = np.array([[Item_Weight, fat_content_map[Item_Fat_Content], Item_Visibility, item_type_map[Item_Type], Item_MRP, Outlet_Establishment_Year, outlet_size_map[Outlet_Size], outlet_location_map[Outlet_Location_Type], outlet_type_map[Outlet_Type]]])
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    
    st.success(f'💰 Predicted Sales: ${prediction[0]:.2f}')
    
st.markdown('</div>', unsafe_allow_html=True)