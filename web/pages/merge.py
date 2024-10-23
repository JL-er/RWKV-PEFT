import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT Merge")
import os
import subprocess
import sys

class Merge:
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
        col1, col2 = st.columns([1,2])
        with col1:
            with st.container(border=True):
                st.subheader("Basic Configuration")
                self.config["merge_type"] = st.selectbox("Select Merge Type", ("bone", "pissa", "lora", "state"))
                self.config["quant"] = st.selectbox("Quant", ["none", "int8", "nf4"], index=0)
                self.config["output"] = st.text_input("Output Path", f"/home/ryan/code/model/meta{self.config['merge_type']}-1.6b.pth")
        with col2:
            with st.container(border=True):
                st.subheader("Model Configuration")
                # Base Model Path
                base_model_directory = st.text_input("Base Model Directory", "/home/ryan/code/model")
                if 'base_model_files' not in st.session_state:
                    st.session_state.base_model_files = []
                if st.button("Check Base Model Directory"):
                    if os.path.exists(base_model_directory):
                        st.success("Base model directory exists!")
                        st.session_state.base_model_files = get_model_files(base_model_directory)
                    else:
                        st.error("Base model directory does not exist!")
                        st.session_state.base_model_files = []
                if st.session_state.base_model_files:
                    self.config["base_model"] = st.selectbox(
                        "Base Model Path",
                        options=st.session_state.base_model_files,
                        index=0,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning("No .pth files found in base model directory.")
                    self.config["base_model"] = st.text_input("Base Model Path", "")

                # Checkpoint Path (LoRA or State)
                checkpoint_label = "State Checkpoint" if self.config["merge_type"] == "state" else "LoRA Checkpoint"
                checkpoint_directory = st.text_input(f"{checkpoint_label} Directory", f"/home/ryan/code/out_model/meta{self.config['merge_type']}")
                if 'checkpoint_files' not in st.session_state:
                    st.session_state.checkpoint_files = []
                if st.button(f"Check {checkpoint_label} Directory"):
                    if os.path.exists(checkpoint_directory):
                        st.success(f"{checkpoint_label} directory exists!")
                        st.session_state.checkpoint_files = get_model_files(checkpoint_directory)
                    else:
                        st.error(f"{checkpoint_label} directory does not exist!")
                        st.session_state.checkpoint_files = []
                if st.session_state.checkpoint_files:
                    checkpoint_key = "state_checkpoint" if self.config["merge_type"] == "state" else "lora_checkpoint"
                    self.config[checkpoint_key] = st.selectbox(
                        f"{checkpoint_label} Path",
                        options=st.session_state.checkpoint_files,
                        index=0,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning(f"No .pth files found in {checkpoint_label.lower()} directory.")
                    checkpoint_key = "state_checkpoint" if self.config["merge_type"] == "state" else "lora_checkpoint"
                    self.config[checkpoint_key] = st.text_input(f"{checkpoint_label} Path", "")

                # PISSA specific configuration
                if self.config["merge_type"] == "pissa":
                    pissa_init_directory = st.text_input("PISSA Init Directory", "/home/ryan/code/out_model/metapissa")
                    if 'pissa_init_files' not in st.session_state:
                        st.session_state.pissa_init_files = []
                    if st.button("Check PISSA Init Directory"):
                        if os.path.exists(pissa_init_directory):
                            st.success("PISSA init directory exists!")
                            st.session_state.pissa_init_files = get_model_files(pissa_init_directory)
                        else:
                            st.error("PISSA init directory does not exist!")
                            st.session_state.pissa_init_files = []
                    if st.session_state.pissa_init_files:
                        self.config["lora_init"] = st.selectbox(
                            "PISSA Init Path",
                            options=st.session_state.pissa_init_files,
                            index=0,
                            format_func=lambda x: os.path.basename(x)
                        )
                    else:
                        st.warning("No .pth files found in PISSA init directory.")
                        self.config["lora_init"] = st.text_input("PISSA Init Path", "")

                # LoRA specific configuration
                if self.config["merge_type"] == "lora":
                    self.config["lora_alpha"] = st.number_input("LoRA Alpha", value=32, min_value=1)

    def show_merge_command(self):
        with st.container():
            st.subheader("Merge Command")
            command = self.generate_merge_command()
            st.code(command, language="bash")

        if st.button("Run Merge"):
            self.run_merge_command(command)

    def generate_merge_command(self):
        common_args = f"""--base_model {self.config['base_model']} \\
--output {self.config['output']} \\
--quant {self.config['quant']}"""

        if self.config["merge_type"] == "bone":
            specific_args = f"""--lora_checkpoint {self.config['lora_checkpoint']}"""
            script = "merge/merge_bone.py"
        elif self.config["merge_type"] in ["pissa", "lora", "state"]:
            script = "merge/merge.py"
            specific_args = f"""--type {self.config['merge_type']}"""
            
            if self.config["merge_type"] == "state":
                specific_args += f" \\\n--state_checkpoint {self.config['state_checkpoint']}"
            else:
                specific_args += f" \\\n--lora_checkpoint {self.config['lora_checkpoint']}"
            
            if self.config["merge_type"] == "pissa":
                specific_args += f" \\\n--lora_init {self.config['lora_init']}"
            elif self.config["merge_type"] == "lora":
                specific_args += f" \\\n--lora_alpha {self.config['lora_alpha']}"

        return f"""python {script} \\
{common_args} \\
{specific_args}"""

    def run_merge_command(self, command):
        try:
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            
            # 使用st.spinner显示正在运行的状态
            with st.spinner('Merging in progress... Please wait.'):
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    text=True,
                    capture_output=True,
                    cwd=self.working_directory
                )
            
            # 合并成功后显示成功消息
            st.balloons()
            st.success("Merge completed successfully!✨✨✨")
            
            # 可选：显示命令输出
            with st.expander("View command output"):
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred during the merge process: {e}")
            st.error(f"Command output: {e.output}")
            st.error(f"Command stderr: {e.stderr}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

def get_model_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]

if __name__ == "__main__":
    merge = Merge()
    merge.render()
