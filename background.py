# styling.py
import streamlit as st

def apply_css():
    

    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://www.google.com/imgres?q=black%20background%20images&imgurl=https%3A%2F%2Fimg.freepik.com%2Ffree-vector%2Fdark-black-background-design-with-stripes_1017-38064.jpg&imgrefurl=https%3A%2F%2Fwww.freepik.com%2Fvectors%2Fblack-background&docid=zQiHfULHXpZXJM&tbnid=xPPPoJGNZBz_wM&vet=12ahUKEwif-a71mduHAxXhzzgGHdWjMSIQM3oECGsQAA..i&w=626&h=358&hcb=2&ved=2ahUKEwif-a71mduHAxXhzzgGHdWjMSIQM3oECGsQAA");
        background-size: 200vw 100vh;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

    input_style = """
    <style>
    input[type="text"] {
        background-color: transparent;
        color: #a19eae;
    }
    div[data-baseweb="base-input"] {
        background-color: transparent !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(input_style, unsafe_allow_html=True)
