import gradio as gr
from training import TrainingInterface
from data import DataInterface
from merge import MergeInterface
from utils import get_project_root
import os

# 添加 logo
logo_path = os.path.join(get_project_root('app.py') + '/assets', 'logo.png')

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown(f"# RWKV-PEFT WEB")
        
        with gr.Tabs():
            with gr.Tab("Train"):
                training_interface = TrainingInterface()
                training_interface.interface
            with gr.Tab("Merge"):
                merge_interface = MergeInterface()
                merge_interface.interface
            with gr.Tab("Data"):
                data_interface = DataInterface()
                data_interface.interface
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
