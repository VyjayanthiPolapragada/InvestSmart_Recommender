import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from frontend.dashboard import run_dashboard
from frontend.recommendation import run_recommender

# Load datasets
sales_data = pd.read_csv('data/sales_with_ROI.csv')  
lease_data = pd.read_csv('data/lease_cleaned_data.csv')  
sales_transit = pd.read_csv('data/sales_transit_data.csv')
lease_transit = pd.read_csv('data/lease_transit_data.csv')

# Page setup
st.set_page_config(page_title="InvestSmart - CRE Platform", layout="wide")
st.sidebar.title("InvestSmart App")

# Sidebar navigation (🏠 Home is now default)
app_mode = st.sidebar.radio("Choose a feature", ["🏠 Home", "📊 Dashboard", "🤖 Recommendation System"])

# Run the selected feature
if app_mode == "🏠 Home":
    st.title("Welcome to InvestSmart 💼")
    st.markdown("""
    InvestSmart is your go-to Commercial Real Estate (CRE) analytics and recommendation platform.
    
    🔍 Explore trends in property prices, lease rates, and transit accessibility  
    🤖 Get personalized investment recommendations  
    📈 Make data-driven decisions backed by intuitive visualizations
    
    Use the sidebar to get started!
    """)
elif app_mode == "📊 Dashboard":
    run_dashboard(sales_data, lease_data, sales_transit, lease_transit)
elif app_mode == "🤖 Recommendation System":
    run_recommender(sales_data, lease_data, sales_transit, lease_transit)

