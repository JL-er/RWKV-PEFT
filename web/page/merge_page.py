import streamlit as st
import os
import subprocess
import sys

class MergePage:
    def __init__(self):
        self.config = {}
        self.working_directory = "/home/ryan/code/RWKV-PEFT-WEB"  # 添加工作目录设置

    def render(self):
        self.setup_page()
        self.setup_config()
        self.show_merge_command()

    def setup_page(self):
        st.title("🔀 RWKV-PEFT Model Merge Interface")
        st.write("Welcome to the RWKV-PEFT model merge interface!")

    def setup_config(self):
        st.header("Merge Configuration")
        self.config["merge_type"] = st.selectbox("Select Merge Type", ("bone", "pissa", "lora"))

        # Common configuration options
        self.config["base_model"] = st.text_input("Base Model Path", "/home/ryan/code/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")
        self.config["lora_checkpoint"] = st.text_input("LoRA Checkpoint Path", "/home/ryan/code/out_model/meta{}-1.6b/rwkv-0.pth".format(self.config["merge_type"]))
        self.config["output"] = st.text_input("Output Path", "/home/ryan/code/model/meta{}-1.6b.pth".format(self.config["merge_type"]))

        if self.config["merge_type"] == "pissa":
            self.config["lora_init"] = st.text_input("LoRA Init Path", "/home/ryan/code/out_model/metapissa-1.6b/init_pissa.pth")

        if self.config["merge_type"] == "lora":
            self.config["lora_alpha"] = st.number_input("LoRA Alpha", value=32, min_value=1)

    def show_merge_command(self):
        st.header("Merge Command")
        command = self.generate_merge_command()
        st.code(command, language="bash")

        if st.button("Run Merge"):
            self.run_merge_command(command)

    def generate_merge_command(self):
        python_path = sys.executable  # 获取当前 Python 解释器的路径
        if self.config["merge_type"] == "bone":
            return f"""{python_path} {self.working_directory}/merge/merge_bone.py --base_model {self.config['base_model']} \
--lora_checkpoint {self.config['lora_checkpoint']} \
--output {self.config['output']}"""
        elif self.config["merge_type"] == "pissa":
            return f"""{python_path} {self.working_directory}/merge/merge.py --base_model {self.config['base_model']} \
--lora_init {self.config['lora_init']} \
--lora_checkpoint {self.config['lora_checkpoint']} \
--output {self.config['output']} \
--type pissa"""
        elif self.config["merge_type"] == "lora":
            return f"""{python_path} {self.working_directory}/merge/merge.py --base_model {self.config['base_model']} \
--lora_checkpoint {self.config['lora_checkpoint']} \
--output {self.config['output']} \
--type lora \
--lora_alpha {self.config['lora_alpha']}"""

    def run_merge_command(self, command):
        try:
            st.info(f"Executing command: {command}")
            st.info(f"Working directory: {self.working_directory}")
            
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=True,
                cwd=self.working_directory
            )
            
            st.success("Merge completed successfully!")
            st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred during the merge process: {e}")
            st.error(f"Command output: {e.output}")
            st.error(f"Command stderr: {e.stderr}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
