import streamlit as st
from PIL import Image
import os
from common.utils import get_project_root

class Sidebar:
    def __init__(self):
        self.project_root = get_project_root()
    
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