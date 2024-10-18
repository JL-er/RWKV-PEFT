import streamlit as st
from page.training_page import TrainingPage
from page.merge_page import MergePage

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Training", "Merge"])

    if page == "Training":
        training_page = TrainingPage()
        training_page.render()
    elif page == "Merge":
        merge_page = MergePage()
        merge_page.render()

if __name__ == "__main__":
    main()
