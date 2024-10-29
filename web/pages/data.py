import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT Data")
import os
import subprocess
import yaml
from PIL import Image
from common.utils import get_project_root
from components.sidebar import Sidebar
# Language dictionary
language_dict = {
    "en": {
        "title": "Data",
        "basic_config": "Basic Configuration",
        "save_output_path": "Save Output Path",
        "output_saved": "Output path saved!",
        "model_config": "Model Configuration",
        "check_data_dir": "Check Data Directory",
        "data_dir_exists": "Data directory exists!",
        "data_dir_not_exists": "Data directory does not exist!",
        "no_checkpoint_files": "No .pth files found in {} directory.",
        "check_pissa_init_dir": "Check PISSA Init Directory",
        "pissa_init_exists": "PISSA init directory exists!",
        "pissa_init_not_exists": "PISSA init directory does not exist!",
        "no_pissa_files": "No .pth files found in PISSA init directory.",
        "no_jsonl_files": "No .jsonl files found in data directory.",
        "data_command": "Data Processing Command",
        "run_data": "Run Data Processing",
        "data": "Data in progress... Please wait.",
        "data_success": "Data processing completed successfully!✨✨✨",
        "command_output": "View command output",
        "error": "An error occurred during the data processing: {}",
        "command_output_error": "Command output: {}",
        "command_stderr_error": "Command stderr: {}",
        "unexpected_error": "An unexpected error occurred: {}"
    },
    "zh": {
        "title": "数据处理界面",
        "basic_config": "基本配置",
        "save_output_path": "保存输出路径",
        "output_saved": "输出路径已保存！",
        "model_config": "模型配置",
        "check_data_dir": "检查数据目录",
        "data_dir_exists": "数据目录存在！",
        "data_dir_not_exists": "数据目录不存在！",
        "no_checkpoint_files": "{} 目录中未找到 .pth 文件。",
        "check_pissa_init_dir": "检查 PISSA 初始化目录",
        "pissa_init_exists": "PISSA 初始化目录存在！",
        "pissa_init_not_exists": "PISSA 初始化目录不存在！",
        "no_pissa_files": "PISSA 初始化目录中未找到 .pth 文件。",
        "no_jsonl_files": "数据目录中未找到 .jsonl 文件。",
        "data_command": "数据处理命令",
        "run_data": "运行数据处理",
        "data": "数据处理进行中... 请稍候。",
        "data_success": "数据处理完成！✨✨✨",
        "command_output": "查看命令输出",
        "error": "数据处理过程中发生错误：{}",
        "command_output_error": "命令输出：{}",
        "command_stderr_error": "命令标准错误：{}",
        "unexpected_error": "发生意外错误：{}"
    }
}

def read_cache(cache_file):
    try:
        with open(cache_file, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}

def write_cache(data, cache_file, is_public=False):
    # Read existing cache
    cache = read_cache(cache_file)
    if is_public:
        cache['public'] = data
    else:
        # Ensure the 'training' section exists
        if 'data' not in cache:
            cache['data'] = {}
        # Update only the specified keys in the 'training' section
        cache['data'].update(data)
    # Write back to the cache file
    with open(cache_file, 'w') as file:
        yaml.safe_dump(cache, file)

class Data:
    def __init__(self):
        self.config = {}
        self.project_root = get_project_root()
        self.cache_name = 'cache.yml'
        Sidebar().show()
        self.lang_code = self.show_language_selection()
        # Load the data section from the cache
        self.cache = read_cache(os.path.join(self.project_root + '/web', self.cache_name)).get('data', {})
        self.config['vocab'] = os.path.join(self.project_root + '/json2binidx_tool', '20B_tokenizer.json')
    
    def show_language_selection(self):
        # Language selection in the sidebar
        language = st.sidebar.selectbox(
        "", 
        ["English", "中文"], 
        index=0 if read_cache(os.path.join(get_project_root() + '/web', 'cache.yml')).get("public", {}).get('language', 'en') == 'en' else 1,
        key='language',
        on_change=self.update_language_cache
        )
        self.lang_code = "en" if language == "English" else "zh"
        return self.lang_code

    def update_language_cache(self):
        self.lang_code = "en" if st.session_state.language == "English" else "zh"
        # Update only the 'language' key in the 'training' section
        write_cache({'language': self.lang_code}, os.path.join(get_project_root() + '/web', 'cache.yml'), is_public=True)

    def render(self):
        self.setup_page()
        self.setup_config()
        self.show_data_command()

    def setup_page(self):
        st.title(language_dict[self.lang_code]["title"])
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

        st.logo(Image.open(os.path.join(self.project_root + '/web', 'assets/peft-logo.png')))

    def setup_config(self):
        with st.container(border=True):
            st.subheader(language_dict[self.lang_code]["basic_config"])
            data_dir = st.text_input(
                "Data Directory", 
                self.cache.get('data_dir', "/home/rwkv/data")
            )
            if 'data_dir_files' not in st.session_state:
                st.session_state.data_dir_files = []
            if st.button(language_dict[self.lang_code]["check_data_dir"]):
                if os.path.exists(data_dir):
                    st.success(language_dict[self.lang_code]["data_dir_exists"])
                    st.session_state.data_dir_files = get_data_files(data_dir)
                    # Save to cache
                    self.cache['data_dir'] = data_dir
                    write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                else:
                    st.error(language_dict[self.lang_code]["data_dir_not_exists"])
                    st.session_state.data_dir_files = []
            if st.session_state.data_dir_files:
                self.config["input"] = st.selectbox(
                    "Data Path",
                    options=st.session_state.data_dir_files,
                    index=0,
                    format_func=lambda x: os.path.basename(x)
                )
            else:
                st.warning(language_dict[self.lang_code]["no_jsonl_files"])
                self.config["input"] = st.text_input("Data Path", "")
            
            # Output name 设置
            output_name = st.text_input(
                "Output Name", 
                self.cache.get('output_name', "sample")
            )
            
            if st.button(language_dict[self.lang_code]["save_output_path"]):
                self.cache['output_name'] = output_name
                write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                st.success(language_dict[self.lang_code]["output_saved"])
            
            # 根据 data_dir 验证状态构建完整的输出路径
            if self.cache.get('data_dir') and os.path.exists(self.cache.get('data_dir')):
                self.config["output"] = os.path.join(self.cache.get('data_dir'), output_name)
            else:
                self.config["output"] = output_name
            
            # 显示完整输出路径（可选）
            st.text(f"Full output path: {self.config['output']}")
            
            self.config["append_eod"] = st.toggle("Append EOD", value=self.cache.get('append_eod', True))

    def show_data_command(self):
        with st.container():
            st.subheader(language_dict[self.lang_code]["data_command"])
            command = self.generate_command()
            st.code(command, language="bash")

        if st.button(language_dict[self.lang_code]["run_data"]):
            self.run_data_command(command)

    def generate_command(self):
        script = "json2binidx_tool/tools/preprocess_data.py"
        common_args = f"--input {self.config['input']} \\\n--output-prefix {self.config['output']} \\\n--vocab {self.config['vocab']} \\\n--dataset-impl mmap \\\n--tokenizer-type HFTokenizer"
        if self.config["append_eod"]:
            common_args += " --append-eod"
        return f"""python {script} {common_args}"""

    def run_data_command(self, command):
        try:
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            
            # 使用st.spinner显示正在运行的状态
            with st.spinner(language_dict[self.lang_code]["data"]):
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
            st.success(language_dict[self.lang_code]["data_success"])
            
            # 可选：显示命令输出
            with st.expander(language_dict[self.lang_code]["command_output"]):
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(language_dict[self.lang_code]["error"].format(e))
            st.error(language_dict[self.lang_code]["command_output_error"].format(e.output))
            st.error(language_dict[self.lang_code]["command_stderr_error"].format(e.stderr))
        except Exception as e:
            st.error(language_dict[self.lang_code]["unexpected_error"].format(str(e)))

def get_data_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')]

if __name__ == "__main__":
    data = Data()
    data.render()
