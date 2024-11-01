import streamlit as st
from common.utils import get_project_root

class Sidebar:
    def __init__(self):
        self.project_root = get_project_root()
        # 插入css
        st.markdown("""
        <style>
        div[data-testid="stSidebarNav"]{
            display: none !important;
        }
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
    
    def show(self, is_running=False):
        """显示侧边栏
        Args:
            is_running (bool): 是否正在运行训练，用于控制链接是否禁用
        """
        with st.sidebar:
            # Navigation links
            st.page_link('app.py', label='Home', disabled=is_running)
            st.page_link('pages/training.py', label='Training', disabled=is_running)
            st.page_link('pages/merge.py', label='Merge', disabled=is_running)
            st.page_link('pages/data.py', label='Data', disabled=is_running)
            