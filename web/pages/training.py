import streamlit as st
st.set_page_config(layout="wide", page_title="RWKV-PEFT Training")
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
            # 获取文件名（不包括扩展名）
            file_prefix = os.path.splitext(file)[0]
            # 如果文件名包含点号，取第一个点号之前的部分作为前缀
            if '.' in file_prefix:
                file_prefix = file_prefix.split('.')[0]
            data_files.add(os.path.join(root, file_prefix))
    return sorted(list(data_files))

class Training:
    def __init__(self):
        self.config = {}
        if 'process' not in st.session_state:
            st.session_state.process = None
        self.gpu_memory_usage = 0
        self.gpu_memory_total = 0
        self.stop_monitoring = False

    def render(self):
        self.setup_page()
        self.setup_config()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.show_generated_script()
        with col2:
            self.show_fixed_parameters()
            run_button = st.button("Run Script", disabled=st.session_state.process is not None)
            if run_button:
                self.run_script(self.generate_script(), self.config['work_dir'])
            stop_button = st.button("Stop Script", disabled=st.session_state.process is None)
            if stop_button:
                self.stop_script()

        # 监控GPU、loss、训练进度
        self.activity_monitor()

        if st.session_state.process is not None:
            self.update_displays()

    def setup_page(self):
        st.title("🎈 RWKV-PEFT Training Interface")

    def setup_config(self):
        # 创建三列布局
        left_column, middle_column, right_column = st.columns([2, 2, 2])

        with left_column:
            # Basic Configuration
            with st.container(border=True):
                st.subheader("Basic Configuration")
                self.config["my_testing"] = st.selectbox("RWKV Version", ["v5", "v6"], index=1)
                self.config["peft"] = st.selectbox("Select PEFT Method", ("bone", "lora", "pissa", "state"), index=0)
                if self.config["peft"] == "bone":
                    col1, col2 = st.columns(2)
                    with col1:
                        bone_load = st.text_input("Bone Load", "")
                    with col2:
                        bone_r = st.number_input("Bone R", value=64, min_value=1)
                    self.config["bone_config"] = json.dumps({"bone_load": bone_load, "bone_r": bone_r})
                self.config["work_dir"] = st.text_input("Working Directory", "/home/ryan/code/RWKV-PEFT-WEB")
                self.config["quant"] = st.selectbox("Quant", ["none", "int8", "nf4"], index=0)
                self.config["proj_dir"] = st.text_input("Project Directory", "/home/ryan/code/out_model/metabone")

        with middle_column:
            # Data Configuration
            with st.container(border=True):
                st.subheader("Data Configuration")
                st.markdown("[配置参考](https://rwkv.cn/RWKV-Fine-Tuning/FT-Dataset)")
                data_file_dir = st.text_input("Data File Path", "/home/ryan/code/data/")
                if st.button("Verify Data File"):
                    if os.path.exists(data_file_dir):
                        st.session_state.data_files = get_data_files(data_file_dir)
                        if st.session_state.data_files:
                            st.success(f"Found {len(st.session_state.data_files)} unique file prefixes in the directory.")
                        else:
                            st.warning("No files found in the specified directory.")
                    else:
                        st.error("Data directory does not exist!")
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
                    st.warning("No data files found. Please check the data directory.")
                    self.config["data_file"] = st.text_input("Data File", data_file_dir)
                    
                col1, col2 = st.columns(2)
                with col1:
                    self.config["data_load"] = st.selectbox("Data Load", ["pad", "get", "only"])
                    self.config["data_type"] = st.selectbox("Data Type", ["binidx", "jsonl"])
                    self.config["data_shuffle"] = st.toggle("Data Shuffle", value=1)
                with col2:
                    self.config["loss_mask"] = st.selectbox("Loss Mask", ["pad", "qa"])
                    self.config["vocab_size"] = st.number_input("Vocab Size", value=65536, min_value=1, disabled=True)
            
            # PEFT-specific configurations
            if self.config["peft"] == "lora":
                self.setup_lora_config()
            elif self.config["peft"] == "pissa":
                self.setup_pissa_config()

        with right_column:
            # Model Configuration
            with st.container(border=True):
                st.subheader("Model Configuration")
                st.markdown("[配置参考](https://rwkv.cn/RWKV-Wiki/Model-Download)")
                model_directory = st.text_input("Model Directory", "/home/ryan/code/model")
                if 'model_files' not in st.session_state:
                    st.session_state.model_files = []
                if st.button("Check Model Directory"):
                    if os.path.exists(model_directory):
                        st.success("Model directory exists!")
                        st.session_state.model_files = get_model_files(model_directory)
                    else:
                        st.error("Model directory does not exist!")
                        st.session_state.model_files = []
                if st.session_state.model_files:
                    self.config["load_model"] = st.selectbox(
                        "Load Model Path",
                        options=st.session_state.model_files,
                        index=0,
                        format_func=lambda x: os.path.basename(x)
                    )
                else:
                    st.warning("No .pth files found. Please check the model directory.")
                    self.config["load_model"] = st.text_input("Load Model Path", "")
                col1, col2 = st.columns(2)
                with col1:
                    self.config["n_layer"] = st.number_input("Number of Layers", value=24, min_value=1)
                with col2:
                    self.config["n_embd"] = st.number_input("Embedding Size", value=2048, min_value=1)
        
        # Training Configuration
        with st.container(border=True):
            st.subheader("Training Configuration")
            st.markdown("[配置参考](https://rwkv.cn/RWKV-Fine-Tuning/Full-ft-Simple#%E8%B0%83%E6%95%B4%E5%85%B6%E4%BB%96%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0)")
            
            col1, col2 = st.columns(2)
            with col1:
                self.config["micro_bsz"] = st.number_input("Micro Batch Size", value=4, min_value=1)
                self.config["epoch_steps"] = st.number_input("Epoch Steps", value=50, min_value=1)
                self.config["epoch_begin"] = st.number_input("Epoch Begin", value=0, min_value=0)
                self.config["lr_init"] = st.number_input("Initial Learning Rate", value=2e-5, format="%.1e")
                self.config["grad_cp"] = st.number_input("Gradient Checkpoint", value=1, min_value=0)
                self.config["strategy"] = st.selectbox("Strategy", ["deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload"])
                self.config["warmup_steps"] = st.number_input("Warmup Steps", value=0, min_value=0)
                
                default_use_fla = True if self.config["peft"] == "state" else False
                self.config["use_fla"] = st.toggle("Use FLA", value=default_use_fla)
                self.config["train_type"] = st.toggle("Infctx", value=False)
                self.config["wandb"] = st.toggle("Wandb", value=False)

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
        
    def setup_lora_config(self):
        with st.container(border=True):
            st.subheader("LoRA Configuration")
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

    def setup_pissa_config(self):
        with st.container(border=True):
            st.subheader("PISSA Configuration")
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

    def show_generated_script(self):
        st.subheader("Generated Script")
        script = self.generate_script()
        st.code(script, language="bash")

    def show_fixed_parameters(self):
        st.subheader("Fixed Parameters")
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
--devices {self.config['devices']} --precision {self.config['precision']} --strategy {self.config['strategy']} --grad_cp {self.config['grad_cp']} \\
{'--my_testing "x060"' if self.config['my_testing'] == "v6" else ''} \\
--dataload {self.config['data_load']} --loss_mask {self.config['loss_mask']} \\
{f"--peft {self.config['peft']}" if self.config['peft'] != 'state' else ''} {f"--bone_config '{self.config['bone_config']}'" if self.config['peft'] == 'bone' else ''}{f" --lora_config '{self.config['lora_config']}'" if self.config['peft'] == 'lora' else ''} {f" --pissa_config '{self.config['pissa_config']}'" if self.config['peft'] == 'pissa' else ''} \\
{f" --accumulate_grad_batches {self.config['accumulate_grad_batches']}" if self.config['accumulate_grad_batches'] > 0 else ''}{f"{' --train_type "state"' if self.config['peft'] == 'state' else ' --train_type "infctx"' if self.config['train_type'] else ''}"}{f" --chunk_ctx {self.config['chunk_ctx']}" if self.config['train_type'] else ""}{fla_arg}{f" --quant {self.config['quant']}" if self.config['quant'] else ''}{f" --train_parts {self.config['train_parts']}" if self.config.get('train_parts') else ''}{f" --wandb {self.config['wandb_project']}" if self.config['wandb'] else ''}{f" --data_shuffle 1" if self.config['data_shuffle'] == True else ' --data_shuffle 0'}"""

        return f"""python train.py {common_args}"""
    
    def run_script(self, script, working_directory):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as temp_file:
            temp_file.write(script)
            temp_file_path = temp_file.name

        try:
            os.chmod(temp_file_path, 0o755)
            st.session_state.process = subprocess.Popen(['bash', temp_file_path], 
                                   cwd=working_directory,
                                   preexec_fn=os.setsid)
            self.start_gpu_monitoring()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            os.unlink(temp_file_path)

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
        st.subheader("Activity Monitor")
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
        while st.session_state.process and st.session_state.process.poll() is None:
            memory_percentage = self.gpu_memory_usage / self.gpu_memory_total if self.gpu_memory_total > 0 else 0
            self.memory_text.text(f"GPU Memory: {self.gpu_memory_usage:.2f} MB / {self.gpu_memory_total:.2f} MB")
            self.memory_bar.progress(memory_percentage)
            placeholder.text("Script is running. GPU memory usage is being updated.")
            
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
        st.success("Script has finished running.✨✨✨")
        st.session_state.process = None

if __name__ == "__main__":
    training = Training()
    training.render()