import streamlit as st
import os
from src.predict import predict_review

st.title("🛡️ Fake Review Detection System")

review_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review_input.strip() == "":
        st.warning("Please enter a review")
    else:
        result = predict_review(review_input)
        st.success(result)