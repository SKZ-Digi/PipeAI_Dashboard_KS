import model_train
import dev_dash
import user_dash
import history
import information
import knowledge
import streamlit as st

PAGES_User = {
    "Associative Information": knowledge,
    "User Dashboard": user_dash,
    "History": history,
    "Information Algorithm": information
}
PAGES_Developer = {
    "Associative Information": knowledge,
    "Developer Dashboard": dev_dash,
    "Information Algorithm": information
}
st.sidebar.title('Navigation')

option = st.sidebar.selectbox( 'Select User', ('End-User','Developer'))

if option == 'End-User':
    selection = st.sidebar.radio("Go to", list(PAGES_User.keys()))
    page = PAGES_User[selection]
    page.app()

if option == 'Developer':
    selection = st.sidebar.radio("Go to", list(PAGES_Developer.keys()))
    page = PAGES_Developer[selection]
    page.app()
