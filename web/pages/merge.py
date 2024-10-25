import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT Merge")
import os
import subprocess
import sys
import yaml

# Add sidebar
st.sidebar.page_link('home.py', label='Home', icon='🏠')
st.sidebar.page_link('pages/training.py', label='Training', icon='🎈')
st.sidebar.page_link('pages/merge.py', label='Merge', icon='🔀')

def read_cache(cache_file):
    try:
        with open(cache_file, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}

def write_cache(data, cache_file):
    # Read existing cache
    cache = read_cache(cache_file)
    # Update the merge section
    cache['merge'] = data
    # Write back to the cache file
    with open(cache_file, 'w') as file:
        yaml.safe_dump(cache, file)

class Merge:
    def __init__(self):
        self.config = {}
        self.project_root = self.get_project_root()
        self.cache_name = 'cache.yml'
        # Load the merge section from the cache
        self.cache = read_cache(os.path.join(self.project_root + '/web', self.cache_name)).get('merge', {})
        
    @staticmethod
    def get_project_root():
        # 获取当前文件的绝对路径
        current_path = os.path.abspath(__file__)
        # 向上遍历直到找到项目根目录（根目录包含 'train.py' 文件）
        while True:
            parent_path = os.path.dirname(current_path)
            if os.path.exists(os.path.join(parent_path, 'train.py')):
                return parent_path
            if parent_path == current_path:  # 已经到达文件系统的根目录
                raise Exception("Could not find project root directory")
            current_path = parent_path

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
                output_path = st.text_input(
                    "Output Path", 
                    self.cache.get('output_path', f"/home/ryan/code/model/meta{self.config['merge_type']}-1.6b.pth")
                )
                if st.button("Save Output Path"):
                    # Save to cache
                    self.cache['output_path'] = output_path
                    write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    st.success("Output path saved!")
                self.config["output"] = output_path
        with col2:
            with st.container(border=True):
                st.subheader("Model Configuration")
                # Base Model Path
                base_model_directory = st.text_input(
                    "Base Model Directory", 
                    self.cache.get('base_model_directory', "/home/ryan/code/model")
                )
                if 'base_model_files' not in st.session_state:
                    st.session_state.base_model_files = []
                if st.button("Check Base Model Directory"):
                    if os.path.exists(base_model_directory):
                        st.success("Base model directory exists!")
                        st.session_state.base_model_files = get_model_files(base_model_directory)
                        # Save to cache
                        self.cache['base_model_directory'] = base_model_directory
                        write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
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
                checkpoint_directory = st.text_input(
                    f"{checkpoint_label} Directory", 
                    self.cache.get('checkpoint_directory', "/home/ryan/code/out_model/metabone")
                )
                if 'checkpoint_files' not in st.session_state:
                    st.session_state.checkpoint_files = []
                if st.button(f"Check {checkpoint_label} Directory"):
                    if os.path.exists(checkpoint_directory):
                        st.success(f"{checkpoint_label} directory exists!")
                        st.session_state.checkpoint_files = get_model_files(checkpoint_directory)
                        # Save to cache
                        self.cache['checkpoint_directory'] = checkpoint_directory
                        write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
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
                    pissa_init_directory = st.text_input(
                        "PISSA Init Directory", 
                        self.cache.get('pissa_init_directory', "/home/ryan/code/out_model/metapissa")
                    )
                    if 'pissa_init_files' not in st.session_state:
                        st.session_state.pissa_init_files = []
                    if st.button("Check PISSA Init Directory"):
                        if os.path.exists(pissa_init_directory):
                            st.success("PISSA init directory exists!")
                            st.session_state.pissa_init_files = get_model_files(pissa_init_directory)
                            # Save to cache
                            self.cache['pissa_init_directory'] = pissa_init_directory
                            write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
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
        merge_types = {
            "bone": {
                "script": "merge/merge_bone.py",
                "specific_args": lambda config: f"--lora_checkpoint {config['lora_checkpoint']}"
            },
            "pissa": {
                "script": "merge/merge.py",
                "specific_args": lambda config: f"--type {config['merge_type']} \\\n--lora_checkpoint {config['lora_checkpoint']} \\\n--lora_init {config['lora_init']}"
            },
            "lora": {
                "script": "merge/merge.py",
                "specific_args": lambda config: f"--type {config['merge_type']} \\\n--lora_checkpoint {config['lora_checkpoint']} \\\n--lora_alpha {config['lora_alpha']}"
            },
            "state": {
                "script": "merge/merge_state.py",
                "specific_args": lambda config: f"--state_checkpoint {config['state_checkpoint']}"
            }
        }

        merge_type = self.config["merge_type"]
        if merge_type not in merge_types:
            raise ValueError(f"Unsupported merge type: {merge_type}")

        script = merge_types[merge_type]["script"]
        specific_args = merge_types[merge_type]["specific_args"](self.config)

        # 为非state类型的合并添加quant参数
        quant_arg = f"--quant {self.config['quant']}" if merge_type != "state" else ""

        common_args = f"""--base_model {self.config['base_model']} \\
--output {self.config['output']}"""

        # 如果有quant参数，添加到common_args中
        if quant_arg:
            common_args += f" \\\n{quant_arg}"

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
                    cwd=self.project_root
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
