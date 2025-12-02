import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Store Item Demand Forecasting App")

st.markdown("""
Welcome!

### Use the left sidebar to navigate between:

- 📊 **EDA Dashboard**  
- 🔮 **Single-Day Prediction**  
- 📈 **Future Forecasting**
""")
