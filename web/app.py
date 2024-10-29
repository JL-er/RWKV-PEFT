import streamlit as st
from common.utils import get_project_root
from PIL import Image
import os
from components.sidebar import Sidebar

st.set_page_config(layout="wide")

# 插入css
st.markdown("""
<style>
div[data-testid="stSidebarHeader"]{
    align-items:center !important;
}
img[data-testid="stLogo"] {
    height: 3.5rem;
    width: 3.5rem;
    background-color: #fff;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

st.logo(Image.open(os.path.join(get_project_root() + '/web', 'assets/peft-logo.png')))
Sidebar().show()
