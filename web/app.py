import streamlit as st
from common.utils import get_project_root
from PIL import Image
import os
from components.sidebar import Sidebar

st.set_page_config(layout="wide")

st.logo(Image.open(os.path.join(get_project_root() + '/web', 'assets/peft-logo.png')))
Sidebar().show()
