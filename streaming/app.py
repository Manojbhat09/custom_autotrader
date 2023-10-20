import streamlit as st
from pages import main_page, dropdown_ticker_analysis, multi_ticker_analysis

PAGES = {
    "Main Page": main_page,
    "Dropdown Ticker Analysis": dropdown_ticker_analysis,
    "Multi-Ticker Analysis": multi_ticker_analysis
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.app()