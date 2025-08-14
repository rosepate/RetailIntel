import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from pydantic import ValidationError

# Initial setup
st.set_page_config(page_title="RETAILINTEL AI", page_icon="🧠", layout="wide")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load agent
from agent.agent import load_agent
agent = load_agent()

# Unified file upload for all features
uploaded_file = st.file_uploader("Upload your retail sales CSV file (used for Chatbot and Dashboard)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Sale Date" in df.columns:
        df.rename(columns={"Sale Date": "Date"}, inplace=True)
else:
    df = None

# Import views
from chat.streamlit_chats import chatbot_view
from dashboard.streamlit_dashboards import dashboard_view

# Streamlit UI Setup
st.title("RETAILINTEL AI")
st.markdown("Welcome to RETAILINTEL AI! Use the navigation menu to explore the chatbot or dashboard.")

# Navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to:", ["💬 Chatbot", "📊 Dashboard"])

# View rendering
if page == "💬 Chatbot":
    if df is not None:
        chatbot_view(agent)
    else:
        st.info("Please upload a CSV file to use the chatbot.")
elif page == "📊 Dashboard":
    if df is not None:
        dashboard_view(df)
    else:
        st.info("Please upload a CSV file to use the dashboard.")