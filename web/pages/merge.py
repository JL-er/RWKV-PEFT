import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT Merge")
import os
import subprocess
import yaml

# Language dictionary
language_dict = {
    "en": {
        "title": "🔀 RWKV-PEFT Model Merge Interface",
        "welcome": "Welcome to the RWKV-PEFT model merge interface!",
        "basic_config": "Basic Configuration",
        "save_output_path": "Save Output Path",
        "output_saved": "Output path saved!",
        "model_config": "Model Configuration",
        "check_base_model_dir": "Check Base Model Directory",
        "base_model_exists": "Base model directory exists!",
        "base_model_not_exists": "Base model directory does not exist!",
        "no_pth_files": "No .pth files found in base model directory.",
        "check_checkpoint_dir": "Check {} Directory",
        "checkpoint_exists": "{} directory exists!",
        "checkpoint_not_exists": "{} directory does not exist!",
        "no_checkpoint_files": "No .pth files found in {} directory.",
        "check_pissa_init_dir": "Check PISSA Init Directory",
        "pissa_init_exists": "PISSA init directory exists!",
        "pissa_init_not_exists": "PISSA init directory does not exist!",
        "no_pissa_files": "No .pth files found in PISSA init directory.",
        "merge_command": "Merge Command",
        "run_merge": "Run Merge",
        "merging": "Merging in progress... Please wait.",
        "merge_success": "Merge completed successfully!✨✨✨",
        "command_output": "View command output",
        "error": "An error occurred during the merge process: {}",
        "command_output_error": "Command output: {}",
        "command_stderr_error": "Command stderr: {}",
        "unexpected_error": "An unexpected error occurred: {}"
    },
    "zh": {
        "title": "🔀 RWKV-PEFT 模型合并界面",
        "welcome": "欢迎使用 RWKV-PEFT 模型合并界面！",
        "basic_config": "基本配置",
        "save_output_path": "保存输出路径",
        "output_saved": "输出路径已保存！",
        "model_config": "模型配置",
        "check_base_model_dir": "检查基底模型目录",
        "base_model_exists": "基底模型目录存在！",
        "base_model_not_exists": "基底模型目录不存在！",
        "no_pth_files": "基底模型目录中未找到 .pth 文件。",
        "check_checkpoint_dir": "检查 {} 目录",
        "checkpoint_exists": "{} 目录存在！",
        "checkpoint_not_exists": "{} 目录不存在！",
        "no_checkpoint_files": "{} 目录中未找到 .pth 文件。",
        "check_pissa_init_dir": "检查 PISSA 初始化目录",
        "pissa_init_exists": "PISSA 初始化目录存在！",
        "pissa_init_not_exists": "PISSA 初始化目录不存在！",
        "no_pissa_files": "PISSA 初始化目录中未找到 .pth 文件。",
        "merge_command": "合并命令",
        "run_merge": "运行合并",
        "merging": "合并进行中... 请稍候。",
        "merge_success": "合并成功完成！✨✨✨",
        "command_output": "查看命令输出",
        "error": "合并过程中发生错误：{}",
        "command_output_error": "命令输出：{}",
        "command_stderr_error": "命令标准错误：{}",
        "unexpected_error": "发生意外错误：{}"
    }
}

# Add sidebar
with st.sidebar:
    st.sidebar.page_link('app.py', label='Home', icon='🏠')
    st.sidebar.page_link('pages/training.py', label='Training', icon='🎈')
    st.sidebar.page_link('pages/merge.py', label='Merge', icon='🔀')

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
        if 'merge' not in cache:
            cache['merge'] = {}
        # Update only the specified keys in the 'training' section
        cache['merge'].update(data)
    # Write back to the cache file
    with open(cache_file, 'w') as file:
        yaml.safe_dump(cache, file)


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

def update_language_cache():
    lang_code = "en" if st.session_state.language == "English" else "zh"
    # Update only the 'language' key in the 'training' section
    write_cache({'language': lang_code}, os.path.join(get_project_root() + '/web', 'cache.yml'), is_public=True)
        
# Language selection in the sidebar
language = st.sidebar.selectbox(
    "", 
    ["English", "中文"], 
    index=0 if read_cache(os.path.join(get_project_root() + '/web', 'cache.yml')).get('public', {}).get('language', 'en') == 'en' else 1,
    key='language',
    on_change=update_language_cache
)
lang_code = "en" if language == "English" else "zh"
class Merge:
    def __init__(self):
        self.config = {}
        self.project_root = get_project_root()
        self.cache_name = 'cache.yml'
        # Load the merge section from the cache
        self.cache = read_cache(os.path.join(self.project_root + '/web', self.cache_name)).get('merge', {})

    def render(self):
        self.setup_page()
        self.setup_config()
        self.show_merge_command()

    def setup_page(self):
        st.title(language_dict[lang_code]["title"])
        st.write(language_dict[lang_code]["welcome"])

    def setup_config(self):
        col1, col2 = st.columns([1,2])
        with col1:
            with st.container(border=True):
                st.subheader(language_dict[lang_code]["basic_config"])
                self.config["merge_type"] = st.selectbox("Select Merge Type", ("bone", "pissa", "lora", "state"))
                self.config["quant"] = st.selectbox("Quant", ["none", "int8", "nf4"], index=0)
                output_path = st.text_input(
                    "Output Path", 
                    self.cache.get('output_path', f"/home/rwkv/model/meta{self.config['merge_type']}-1.6b.pth")
                )
                if st.button(language_dict[lang_code]["save_output_path"]):
                    # Save to cache
                    self.cache['output_path'] = output_path
                    write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    st.success(language_dict[lang_code]["output_saved"])
                self.config["output"] = output_path
        with col2:
            with st.container(border=True):
                st.subheader(language_dict[lang_code]["model_config"])
                # Base Model Path
                base_model_directory = st.text_input(
                    "Base Model Directory", 
                    self.cache.get('base_model_directory', "/home/rwkv/model")
                )
                if 'base_model_files' not in st.session_state:
                    st.session_state.base_model_files = []
                if st.button(language_dict[lang_code]["check_base_model_dir"]):
                    if os.path.exists(base_model_directory):
                        st.success(language_dict[lang_code]["base_model_exists"])
                        st.session_state.base_model_files = get_model_files(base_model_directory)
                        # Save to cache
                        self.cache['base_model_directory'] = base_model_directory
                        write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    else:
                        st.error(language_dict[lang_code]["base_model_not_exists"])
                        st.session_state.base_model_files = []
                if st.session_state.base_model_files:
                    self.config["base_model"] = st.selectbox(
                        "Base Model Path",
                        options=st.session_state.base_model_files,
                        index=0,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning(language_dict[lang_code]["no_pth_files"])
                    self.config["base_model"] = st.text_input("Base Model Path", "")

                # Checkpoint Path (LoRA or State)
                checkpoint_label = "State Checkpoint" if self.config["merge_type"] == "state" else "LoRA Checkpoint"
                checkpoint_directory = st.text_input(
                    f"{checkpoint_label} Directory", 
                    self.cache.get('checkpoint_directory', "/home/rwkv/out_model/metabone")
                )
                if 'checkpoint_files' not in st.session_state:
                    st.session_state.checkpoint_files = []
                if st.button(language_dict[lang_code]["check_checkpoint_dir"].format(checkpoint_label)):
                    if os.path.exists(checkpoint_directory):
                        st.success(language_dict[lang_code]["checkpoint_exists"].format(checkpoint_label))
                        st.session_state.checkpoint_files = get_model_files(checkpoint_directory)
                        # Save to cache
                        self.cache['checkpoint_directory'] = checkpoint_directory
                        write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    else:
                        st.error(language_dict[lang_code]["checkpoint_not_exists"].format(checkpoint_label))
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
                    st.warning(language_dict[lang_code]["no_checkpoint_files"].format(checkpoint_label.lower()))
                    checkpoint_key = "state_checkpoint" if self.config["merge_type"] == "state" else "lora_checkpoint"
                    self.config[checkpoint_key] = st.text_input(f"{checkpoint_label} Path", "")

                # PISSA specific configuration
                if self.config["merge_type"] == "pissa":
                    pissa_init_directory = st.text_input(
                        "PISSA Init Directory", 
                        self.cache.get('pissa_init_directory', "/home/rwkv/out_model/metapissa")
                    )
                    if 'pissa_init_files' not in st.session_state:
                        st.session_state.pissa_init_files = []
                    if st.button(language_dict[lang_code]["check_pissa_init_dir"]):
                        if os.path.exists(pissa_init_directory):
                            st.success(language_dict[lang_code]["pissa_init_exists"])
                            st.session_state.pissa_init_files = get_model_files(pissa_init_directory)
                            # Save to cache
                            self.cache['pissa_init_directory'] = pissa_init_directory
                            write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                        else:
                            st.error(language_dict[lang_code]["pissa_init_not_exists"])
                            st.session_state.pissa_init_files = []
                    if st.session_state.pissa_init_files:
                        self.config["lora_init"] = st.selectbox(
                            "PISSA Init Path",
                            options=st.session_state.pissa_init_files,
                            index=0,
                            format_func=lambda x: os.path.basename(x)
                        )
                    else:
                        st.warning(language_dict[lang_code]["no_pissa_files"])
                        self.config["lora_init"] = st.text_input("PISSA Init Path", "")

                # LoRA specific configuration
                if self.config["merge_type"] == "lora":
                    self.config["lora_alpha"] = st.number_input("LoRA Alpha", value=32, min_value=1)

    def show_merge_command(self):
        with st.container():
            st.subheader(language_dict[lang_code]["merge_command"])
            command = self.generate_merge_command()
            st.code(command, language="bash")

        if st.button(language_dict[lang_code]["run_merge"]):
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
            with st.spinner(language_dict[lang_code]["merging"]):
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
            st.success(language_dict[lang_code]["merge_success"])
            
            # 可选：显示命令输出
            with st.expander(language_dict[lang_code]["command_output"]):
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(language_dict[lang_code]["error"].format(e))
            st.error(language_dict[lang_code]["command_output_error"].format(e.output))
            st.error(language_dict[lang_code]["command_stderr_error"].format(e.stderr))
        except Exception as e:
            st.error(language_dict[lang_code]["unexpected_error"].format(str(e)))

def get_model_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]

if __name__ == "__main__":
    merge = Merge()
    merge.render()
