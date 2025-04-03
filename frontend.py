import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://localhost:5000/"

st.title("Flask & Streamlit Connection")

if st.button("Fetch Message"):
    try:
        response = requests.get(BACKEND_URL)
        response.raise_for_status()  # Raise error for HTTP issues
        data = response.json()
        st.success(data.get("message", "No response received."))
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
    except ValueError:
        st.error("Invalid JSON response from the server.")
