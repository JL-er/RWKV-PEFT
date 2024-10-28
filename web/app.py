import streamlit as st

st.set_page_config(layout="wide")

# st.title("RWKV-PEFT-WEB-HOME")
with st.sidebar:
    st.sidebar.page_link('app.py', label='Home', icon='🏠')
    st.sidebar.page_link('pages/training.py', label='Training', icon='🎈')
    st.sidebar.page_link('pages/merge.py', label='Merge', icon='🔀')