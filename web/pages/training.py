import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT-Training")
import os
import json
import subprocess
import tempfile
import signal
import threading
import time
import GPUtil
import pandas as pd
import plotly.express as px
import psutil
import yaml

# Language dictionary
language_dict = {
    "en": {
        "title": "🎈 RWKV-PEFT Training Interface",
        "basic_config": "Basic Configuration",
        "data_config": "Data Configuration",
        "model_config": "Model Configuration",
        "training_config": "Training Configuration",
        "save_output_path": "Save Output Path",
        "output_saved": "Output path saved!",
        "data_found": "Found {} unique file prefixes in the directory.",
        "no_files_found": "No files found in the specified directory.",
        "data_dir_not_exist": "Data directory does not exist!",
        "model_dir_exists": "Model directory exists!",
        "model_dir_not_exist": "Model directory does not exist!",
        "no_pth_files": "No .pth files found. Please check the model directory.",
        "run_script": "Run Script",
        "stop_script": "Stop Script",
        "generated_script": "Generated Script",
        "fixed_parameters": "Fixed Parameters",
        "activity_monitor": "Activity Monitor",
        "script_running": "Script is running. GPU memory usage is being updated.",
        "training_success": "Training completed successfully!✨✨✨",
        "output_dir_not_empty": "Error: Output directory '{output_dir}' already contains .pth files.",
        "config_reference": "Config Reference"
    },
    "zh": {
        "title": "🎈 RWKV-PEFT 训练界面",
        "basic_config": "基本配置",
        "data_config": "数据配置",
        "model_config": "模型配置",
        "training_config": "训练配置",
        "save_output_path": "保存输出路径",
        "output_saved": "输出路径已保存！",
        "data_found": "在目录中找到 {} 个唯一文件前缀。",
        "no_files_found": "在指定目录中未找到文件。",
        "data_dir_not_exist": "数据目录不存在！",
        "model_dir_exists": "基底模型目录存在！",
        "model_dir_not_exist": "基底模型目录不存在！",
        "no_pth_files": "未找到 .pth 文件。请检查模型目录。",
        "run_script": "运行脚本",
        "stop_script": "停止脚本",
        "generated_script": "脚本预览",
        "fixed_parameters": "固定参数",
        "activity_monitor": "活动监控",
        "script_running": "脚本正在运行。GPU 内存使用情况正在更新。",
        "training_success": "训练完成！✨✨✨",
        "output_dir_not_empty": "错误：输出目录 '{output_dir}' 已包含 .pth 文件。",
        "config_reference": "配置参考"
    }
}


def get_model_files(directory):
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    return model_files

def get_data_files(directory):
    data_files = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_prefix = os.path.splitext(file)[0]
            if '.' in file_prefix:
                file_prefix = file_prefix.split('.')[0]
            data_files.add(os.path.join(root, file_prefix))
    return sorted(list(data_files))

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
        if 'training' not in cache:
            cache['training'] = {}
        # Update only the specified keys in the 'training' section
        cache['training'].update(data)
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
        if parent_path == current_path:
            raise Exception("Could not find project root directory")
        current_path = parent_path

class Training:
    def __init__(self):
        self.config = {}
        if 'process' not in st.session_state:
            st.session_state.process = None
        self.show_sidebar()
        self.lang_code = self.show_language_selection()
        self.gpu_memory_usage = 0
        self.gpu_memory_total = 0
        self.memory_text = ""
        self.memory_bar = None
        self.stop_monitoring = False
        self.project_root = get_project_root()
        self.cache_name = 'cache.yml'
        # Load the training section from the cache
        self.cache = read_cache(os.path.join(self.project_root + '/web', self.cache_name)).get('training', {})
    
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

    def show_sidebar(self):
        # 确保 process 存在于 session_state 中
        if 'process' not in st.session_state:
            st.session_state.process = None
        
        # 检查进程是否在运行
        if st.session_state.process is not None:
            try:
                # 检查进程是否还活着
                is_running = st.session_state.process.poll() is None
            except:
                is_running = False
        else:
            is_running = False

        with st.sidebar:
            st.sidebar.page_link('app.py', label='Home', icon='🏠', disabled=is_running)
            st.sidebar.page_link('pages/training.py', label='Training', icon='🎈', disabled=is_running)
            st.sidebar.page_link('pages/merge.py', label='Merge', icon='🔀', disabled=is_running)
        
    def render(self):
        self.setup_page()
        self.setup_config()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.show_generated_script()
        with col2:
            self.show_fixed_parameters()
            run_button = st.button(language_dict[self.lang_code]["run_script"], disabled=st.session_state.process is not None)
            if run_button:
                output_dir = self.config["proj_dir"]
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    if any(file.endswith('.pth') for file in files):
                        # Ensure output_dir is defined before this block
                        output_dir = self.config.get("proj_dir", "")
                        # Use the correct placeholder name in the format method
                        st.error(language_dict[self.lang_code]["output_dir_not_empty"].format(output_dir=output_dir))
                        # proceed = st.button("仍然继续")
                        # new_output_dir = st.text_input("或指定一个新的输出路径：", output_dir)
                        # if proceed:
                        # self.run_script(self.generate_script())
                        # elif new_output_dir != output_dir:
                        #     self.config["proj_dir"] = new_output_dir
                        #     self.run_script(self.generate_script())
                    else:
                        self.run_script(self.generate_script())
                else:
                    self.run_script(self.generate_script())
            
            stop_button = st.button(language_dict[self.lang_code]["stop_script"], disabled=st.session_state.process is None)
            if stop_button:
                self.stop_script()

        # 监控GPU、loss、训练进度
        self.activity_monitor()

        if st.session_state.process is not None:
            self.update_displays()

    def setup_page(self):
        st.title(language_dict[self.lang_code]["title"])

    def setup_config(self):
        # 创建三列布局
        left_column, middle_column, right_column = st.columns([2, 2, 2])

        with left_column:
            # Basic Configuration
            with st.container(border=True):
                st.subheader(language_dict[self.lang_code]["basic_config"])
                self.config["my_testing"] = st.selectbox("RWKV Version", ["v5", "v6"], index=1)
                self.config["peft"] = st.selectbox("Select PEFT Method", ("bone", "lora", "pissa", "state"), index=0)
                if self.config["peft"] == "bone":
                    col1, col2 = st.columns(2)
                    with col1:
                        bone_load = st.text_input("Bone Load", "")
                    with col2:
                        bone_r = st.number_input("Bone R", value=64, min_value=1)
                    self.config["bone_config"] = json.dumps({"bone_load": bone_load, "bone_r": bone_r})
                self.config["quant"] = st.selectbox("Quant", ["none", "int8", "nf4"], index=0)
                proj_dir = st.text_input(
                    "Output Path", 
                    self.cache.get('proj_dir', "/home/rwkv/out_model/")
                )
                if st.button(language_dict[self.lang_code]["save_output_path"]):
                    self.cache['proj_dir'] = proj_dir
                    write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    st.success(language_dict[self.lang_code]["output_saved"])
                self.config["proj_dir"] = proj_dir
                if self.config["peft"] == "lora":
                    lora_load = st.text_input("LoRA Load", "")
                    lora_r = st.number_input("LoRA R", value=32, min_value=1)
                    col1, col2 = st.columns(2)
                    with col1:
                        lora_alpha = st.number_input("LoRA Alpha", value=32, min_value=1)
                    with col2:
                        lora_dropout = st.number_input("LoRA Dropout", value=0.0, min_value=0.0, max_value=1.0, format="%.2f")
                    self.config["lora_config"] = json.dumps({
                        "lora_load": lora_load,
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout
                    })
                if self.config["peft"] == "pissa":
                    pissa_load = st.text_input("PISSA Load", "")
                    pissa_init = st.text_input("PISSA Init", "")
                    col1, col2 = st.columns(2)
                    with col1:
                        pissa_r = st.number_input("PISSA R", value=32, min_value=1)
                    with col2:
                        svd_niter = st.number_input("SVD Niter", value=4, min_value=1)
                    self.config["pissa_config"] = json.dumps({
                        "pissa_load": pissa_load,
                        "pissa_init": pissa_init,
                        "pissa_r": pissa_r,
                        "svd_niter": svd_niter
                    })

        with middle_column:
            # Data Configuration
            with st.container(border=True):
                st.subheader(language_dict[self.lang_code]["data_config"])
                st.markdown(f"[{language_dict[self.lang_code]['config_reference']}](https://rwkv.cn/RWKV-Fine-Tuning/FT-Dataset)")
                data_file_dir = st.text_input(
                    "Data File Path", 
                    self.cache.get('data_file_dir', "/home/rwkv/data/")
                )
                if st.button("Check Data File"):
                    if os.path.exists(data_file_dir):
                        st.session_state.data_files = get_data_files(data_file_dir)
                        if st.session_state.data_files:
                            st.success(language_dict[self.lang_code]["data_found"].format(len(st.session_state.data_files)))
                            self.cache['data_file_dir'] = data_file_dir
                            write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                        else:
                            st.warning(language_dict[self.lang_code]["no_files_found"])
                    else:
                        st.error(language_dict[self.lang_code]["data_dir_not_exist"])
                        st.session_state.data_files = []
                
                if 'data_files' not in st.session_state:
                    st.session_state.data_files = []
                
                if st.session_state.data_files:
                    self.config["data_file"] = st.selectbox(
                        "Data File",
                        options=st.session_state.data_files,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning(language_dict[self.lang_code]["no_files_found"])
                    self.config["data_file"] = st.text_input("Data File", "")
                    
                col1, col2 = st.columns(2)
                with col1:
                    self.config["data_load"] = st.selectbox("Data Load", ["pad", "get", "only"])
                    self.config["data_type"] = st.selectbox("Data Type", ["binidx", "jsonl"])
                    self.config["data_shuffle"] = st.toggle("Data Shuffle", value=1)
                with col2:
                    self.config["loss_mask"] = st.selectbox("Loss Mask", ["none", "pad", "qa"], index=0)
                    self.config["vocab_size"] = st.number_input("Vocab Size", value=65536, min_value=1, disabled=True)

        with right_column:
            # Model Configuration
            with st.container(border=True):
                st.subheader(language_dict[self.lang_code]["model_config"])
                st.markdown(f"[{language_dict[self.lang_code]['config_reference']}](https://rwkv.cn/RWKV-Wiki/Model-Download)")
                model_directory = st.text_input(
                    "Base Model Directory", 
                    self.cache.get('model_directory', "/home/rwkv/model")
                )
                if st.button("Check Base Model Directory"):
                    if os.path.exists(model_directory):
                        st.success(language_dict[self.lang_code]["model_dir_exists"])
                        st.session_state.model_files = get_model_files(model_directory)
                        self.cache['model_directory'] = model_directory
                        write_cache(self.cache, os.path.join(self.project_root + '/web', self.cache_name))
                    else:
                        st.error(language_dict[self.lang_code]["model_dir_not_exist"])
                        st.session_state.model_files = []
                if 'model_files' not in st.session_state:
                    st.session_state.model_files = []
                
                if st.session_state.model_files:
                    self.config["load_model"] = st.selectbox(
                        "Load Model Path",
                        options=st.session_state.model_files,
                        index=0,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning(language_dict[self.lang_code]["no_pth_files"])
                    self.config["load_model"] = st.text_input("Load Model Path", "")
                col1, col2 = st.columns(2)
                with col1:
                    self.config["n_layer"] = st.number_input("Number of Layers", value=24, min_value=1)
                with col2:
                    self.config["n_embd"] = st.number_input("Embedding Size", value=2048, min_value=1)
                self.config["train_parts"] = st.multiselect("Train Parts", ["emb", "head", "time", "ln" ], default=["time", "ln"])

        # Training Configuration
        with st.container(border=True):
            st.subheader(language_dict[self.lang_code]["training_config"])
            st.markdown(f"[{language_dict[self.lang_code]['config_reference']}](https://rwkv.cn/RWKV-Fine-Tuning/Full-ft-Simple#%E8%B0%83%E6%95%B4%E5%85%B6%E4%BB%96%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0)")
            
            col1, col2 = st.columns(2)
            with col1:
                self.config["micro_bsz"] = st.number_input("Micro Batch Size", value=4, min_value=1)
                self.config["epoch_steps"] = st.number_input("Epoch Steps", value=50, min_value=1)
                self.config["epoch_begin"] = st.number_input("Epoch Begin", value=0, min_value=0)
                self.config["lr_init"] = st.number_input("Initial Learning Rate", value=2e-5, format="%.1e")
                self.config["strategy"] = st.selectbox("Strategy", ["deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload"])
                self.config["warmup_steps"] = st.number_input("Warmup Steps", value=0, min_value=0)
                _col1, _col2 = st.columns(2)
                with _col1:
                    default_use_fla = True if self.config["peft"] == "state" else False
                    self.config["use_fla"] = st.toggle("Use FLA", value=default_use_fla)
                    self.config["grad_cp"] = st.toggle("Gradient Checkpoint", value=True)
                with _col2:
                    self.config["wandb"] = st.toggle("Wandb", value=False)
                    self.config["train_type"] = st.toggle("Infctx", value=False)

            with col2:
                self.config["epoch_count"] = st.number_input("Epoch Count", value=1, min_value=1)
                self.config["epoch_save"] = st.number_input("Epoch Save", value=1, min_value=1)
                self.config["ctx_len"] = st.number_input("Context Length", value=512, min_value=1)
                self.config["lr_final"] = st.number_input("Final Learning Rate", value=2e-5, format="%.1e")
                self.config["precision"] = st.selectbox("Precision", ["bf16", "fp16", "fp32"])
                self.config["accumulate_grad_batches"] = st.number_input("Accumulate Grad Batches", value=0, min_value=0)
                self.config["devices"] = st.number_input("Number of Devices", value=1, min_value=1)
                if self.config['train_type']:
                    self.config["chunk_ctx"] = st.selectbox("Chunk Context", ["128", "256", "512", "1024"], index=2)
                if self.config["wandb"]:
                    self.config["wandb_project"] = st.text_input("Wandb Project", "peft-loss")

    def show_hover_image(self):
        image1 = "https://rwkv.cn/images/RWKV-Wiki-Cover.png"
        image2 = "https://rwkv.cn/images/RWKV-Prompt-Cover.png"
        # HTML 和 CSS 代码
        hover_html = f"""
                <style>
                .hover-container {{
                    position: relative;
                    width: 150px; /* 设定图片的宽度 */
                }}
                .hover-container img {{
                    width: 100%;
                    transition: opacity 0.5s ease-in-out;
                }}
                .hover-container img.image-b {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    opacity: 0;
                }}
                .hover-container:hover img.image-b {{
                    opacity: 1;
                }}
                .hover-container:hover img.image-a {{
                    opacity: 0;
                }}
                </style>

                <div class="hover-container">
                    <img src="{image1}" class="image-a" alt="Image A">
                    <img src="{image2}" class="image-b" alt="Image B">
                </div>
                """
            
        # 在 Streamlit 中显示 HTML 代码
        st.markdown(hover_html, unsafe_allow_html=True)

    def show_generated_script(self):
        st.subheader(language_dict[self.lang_code]["generated_script"])
        script = self.generate_script()
        st.code(script, language="bash")

    def show_fixed_parameters(self):
        st.subheader(language_dict[self.lang_code]["fixed_parameters"])
        # st.write("The following parameters are fixed and cannot be changed:")
        st.code("""
        pre_ffn = 0
        head_qk = 0
        beta1 = 0.9
        beta2 = 0.99
        adam_eps = 1e-8
        accelerator = gpu
        """)

    def generate_script(self):
        fixed_args = """--pre_ffn 0 --head_qk 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu"""
        fla_arg = " --fla" if self.config.get('use_fla', False) else ""
        common_args = f"""--load_model {self.config['load_model']} \\
--proj_dir {self.config['proj_dir']} --data_file {self.config['data_file']} \\
--data_type {self.config['data_type']} --vocab_size {self.config['vocab_size']} \\
--ctx_len {self.config['ctx_len']} --epoch_steps {self.config['epoch_steps']} --epoch_count {self.config['epoch_count']} --epoch_begin {self.config['epoch_begin']} --epoch_save {self.config['epoch_save']} --micro_bsz {self.config['micro_bsz']} \\
--n_layer {self.config['n_layer']} --n_embd {self.config['n_embd']} \\
--lr_init {self.config['lr_init']} --lr_final {self.config['lr_final']} --warmup_steps {self.config['warmup_steps']} \\
{fixed_args} \\
--devices {self.config['devices']} --precision {self.config['precision']} --strategy {self.config['strategy']} {'--grad_cp 0' if not self.config['grad_cp'] else '--grad_cp 1'} \\
{'--my_testing "x060"' if self.config['my_testing'] == "v6" else ''} \\
--dataload {self.config['data_load']} --loss_mask {self.config['loss_mask']} \\
{f"--peft {self.config['peft']}" if self.config['peft'] != 'state' else ''} {f"--bone_config '{self.config['bone_config']}'" if self.config['peft'] == 'bone' else ''}{f" --lora_config '{self.config['lora_config']}'" if self.config['peft'] == 'lora' else ''} {f" --pissa_config '{self.config['pissa_config']}'" if self.config['peft'] == 'pissa' else ''} \\
{f"--data_shuffle 1" if self.config['data_shuffle'] == True else '--data_shuffle 0'}{f" --accumulate_grad_batches {self.config['accumulate_grad_batches']}" if self.config['accumulate_grad_batches'] > 0 else ''}{f"{' --train_type state' if self.config['peft'] == 'state' else ' --train_type infctx' if self.config['train_type'] else ''}"}{f" --chunk_ctx {self.config['chunk_ctx']}" if self.config['train_type'] else ""}{fla_arg}{f" --quant {self.config['quant']}" if self.config['quant'] else ''}{f" --wandb {self.config['wandb_project']}" if self.config['wandb'] else ''}"""

        return f"""python train.py {common_args}"""
    
    def run_script(self, script):
        st.session_state.process = True
        # with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as temp_file:
        #     temp_file.write(script)
        #     temp_file_path = temp_file.name

        # try:
        #     os.chmod(temp_file_path, 0o755)
        #     st.session_state.process = subprocess.Popen(['bash', temp_file_path], 
        #                            cwd=self.project_root,
        #                            preexec_fn=os.setsid)
        #     self.start_gpu_monitoring()
        # except Exception as e:
        #     st.error(f"An error occurred: {str(e)}")
        # finally:
        #     os.unlink(temp_file_path)

    def stop_script(self):
        print("Attempting to stop script", st.session_state.process)
        if st.session_state.process:
            try:
                parent = psutil.Process(st.session_state.process.pid)
                
                children = parent.children(recursive=True)
                
                for child in children:
                    child.terminate()
                
                gone, alive = psutil.wait_procs(children, timeout=3)
                
                for p in alive:
                    p.kill()
                
                parent.terminate()
                parent.wait(5)
                
                if parent.is_running():
                    parent.kill()
                    parent.wait(5)
            
            except psutil.NoSuchProcess:
                print("Process already terminated")
            except Exception as e:
                print(f"Error stopping script: {e}")
            finally:
                st.session_state.process = None
                self.stop_gpu_monitoring()
                print("Script stopped")
        else:
            print("No script running")

    def monitor_gpu_memory(self):
        while not self.stop_monitoring:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_memory_usage = gpus[0].memoryUsed
                self.gpu_memory_total = gpus[0].memoryTotal
            time.sleep(1)

    def activity_monitor(self):
        st.subheader(language_dict[self.lang_code]["activity_monitor"])
        self.setup_gpu_monitoring()
        self.setup_loss_chart()
        self.setup_training_progress()
        
    def start_gpu_monitoring(self):
        self.stop_monitoring = False
        threading.Thread(target=self.monitor_gpu_memory, daemon=True).start()

    def stop_gpu_monitoring(self):
        self.stop_monitoring = True

    def read_data(self, proj_dir):
        loss_file = os.path.join(proj_dir, "loss_data.json")
        loss_data = []
        t_cost = 0
        kt_s = 0
        loss = 0
        if os.path.exists(loss_file):
            with open(loss_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    loss_data.append(data['loss'])
                    t_cost = data['t_cost']
                    kt_s = data['kt_s']
                    loss = data['loss']
        return loss_data, t_cost, kt_s, loss

    def setup_gpu_monitoring(self):
        self.memory_bar = st.empty()
        self.memory_text = st.empty()

    def setup_loss_chart(self):
        self.loss_chart = st.empty()

    def setup_training_progress(self):
        self.rate_bar = st.empty()
        self.rate_text = st.empty()

    def update_displays(self):
        placeholder = st.empty()
        loss_data = []
        current_epoch = 0
        last_progress = 0
        last_t_cost = 0
        last_kt_s = 0
        last_loss = 0
        while st.session_state.process:
            print(1111)
            memory_percentage = self.gpu_memory_usage / self.gpu_memory_total if self.gpu_memory_total > 0 else 0
            self.memory_text.text(f"GPU Memory: {self.gpu_memory_usage:.2f} MB / {self.gpu_memory_total:.2f} MB")
            self.memory_bar.progress(memory_percentage)
            placeholder.text(language_dict[self.lang_code]["script_running"])
            
            new_loss_data, t_cost, kt_s, loss = self.read_data(self.config['proj_dir'])
            
            current_epoch = min(int(len(new_loss_data) / self.config['epoch_steps']), self.config['epoch_count'] - 1)
            total_progress = min(len(new_loss_data) / (self.config['epoch_steps'] * self.config['epoch_count']), 1.0)
            
            if total_progress > last_progress:
                last_progress = total_progress
                last_t_cost = t_cost
                last_kt_s = kt_s
                last_loss = loss
            
            self.rate_bar.progress(last_progress)
            self.rate_text.text(f"Epoch {current_epoch + 1}/{self.config['epoch_count']}: {last_progress:.2%} complete "
                           f"| it/s: {last_t_cost:.2f} | Kt/s: {last_kt_s:.2f} | Loss: {last_loss:.4f}")
            
            if len(new_loss_data) > len(loss_data):
                loss_data = new_loss_data
                steps = range(1, len(loss_data) + 1)
                df = pd.DataFrame({'step': steps, 'loss': loss_data})
                
                fig = px.line(df, x='step', y='loss', title='Training Loss')
                fig.update_layout(xaxis_title='Epoch Step', yaxis_title='Loss')
                if self.config['accumulate_grad_batches'] > 0:
                    fig.update_xaxes(range=[1, (self.config['epoch_steps'] * self.config['epoch_count']) // self.config['accumulate_grad_batches']])
                else:
                    fig.update_xaxes(range=[1, self.config['epoch_steps'] * self.config['epoch_count']])
                self.loss_chart.plotly_chart(fig, use_container_width=True)

            time.sleep(1)
            placeholder.empty()
        
        self.rate_bar.progress(1.0)
        self.rate_text.text(f"Training Complete: 100.00% | it/s: {last_t_cost:.2f} | Kt/s: {last_kt_s:.2f} | Loss: {last_loss:.4f}")
        
        self.stop_gpu_monitoring()
        st.balloons()
        st.success(language_dict[self.lang_code]["training_success"])
        st.session_state.process = None

if __name__ == "__main__":
    training = Training()
    training.render()

