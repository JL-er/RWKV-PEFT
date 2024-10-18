import streamlit as st
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

class TrainingPage:
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
        self.show_fixed_parameters()

        working_directory = st.text_input("Working Directory", "/home/ryan/code/RWKV-PEFT-WEB")

        script = self.generate_script()
        st.header("Generated Script")
        st.code(script, language="bash")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Script"):
                loss_file = os.path.join(self.config['proj_dir'], "loss_data.json")
                if os.path.exists(loss_file):
                    os.remove(loss_file)
                self.run_script(script, working_directory)
        with col2:
            if st.button("Stop Script"):
                self.stop_script()

        self.setup_gpu_monitoring()
        self.setup_loss_chart()
        self.setup_training_progress()

        if st.session_state.process is not None:
            self.update_displays()

    def setup_page(self):
        st.title("🎈 RWKV-PEFT Training Interface")
        st.write("Welcome to the RWKV-PEFT training interface!")

    def setup_config(self):
        st.header("Configuration")
        self.config["peft"] = st.selectbox("Select PEFT Method", ("bone", "lora", "pissa", "state"), index=0)
        
        # Common configuration options
        self.config["load_model"] = st.text_input("Load Model Path", "/home/ryan/code/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")
        self.config["proj_dir"] = st.text_input("Project Directory", "/home/ryan/code/out_model/metabone")
        self.config["data_file"] = st.text_input("Data File Path", "/home/ryan/code/data/roleplay")
        self.config["n_layer"] = st.number_input("Number of Layers", value=24, min_value=1)
        self.config["n_embd"] = st.number_input("Embedding Size", value=2048, min_value=1)
        self.config["micro_bsz"] = st.number_input("Micro Batch Size", value=4, min_value=1)
        self.config["epoch_count"] = st.number_input("Epoch Count", value=1, min_value=1)
        self.config["epoch_begin"] = st.number_input("Epoch Begin", value=0, min_value=0)
        self.config["epoch_save"] = st.number_input("Epoch Save", value=1, min_value=1)
        self.config["epoch_steps"] = st.number_input("Epoch Steps", value=50, min_value=1)
        self.config["ctx_len"] = st.number_input("Context Length", value=512, min_value=1)
        self.config["lr_init"] = st.number_input("Initial Learning Rate", value=2e-5, format="%.1e")
        self.config["lr_final"] = st.number_input("Final Learning Rate", value=2e-5, format="%.1e")
        self.config["devices"] = st.number_input("Number of Devices", value=1, min_value=1)
        self.config["precision"] = st.selectbox("Precision", ["bf16", "fp16", "fp32"])
        self.config["strategy"] = st.selectbox("Strategy", ["deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"])
        self.config["grad_cp"] = st.number_input("Gradient Checkpoint", value=1, min_value=0)
        self.config["accumulate_grad_batches"] = st.number_input("Accumulate Grad Batches", value=0, min_value=0)

        # PEFT-specific configurations
        if self.config["peft"] == "state":
            self.setup_state_config()
        elif self.config["peft"] == "bone":
            self.setup_bone_config()
        elif self.config["peft"] == "lora":
            self.setup_lora_config()
        elif self.config["peft"] == "pissa":
            self.setup_pissa_config()

    def setup_state_config(self):
        st.subheader("State Configuration")
        self.config["use_fla"] = st.toggle("Use FLA", value=True)

    def setup_bone_config(self):
        st.subheader("Bone Configuration")
        bone_load = st.text_input("Bone Load", "")
        bone_r = st.number_input("Bone R", value=64, min_value=1)
        self.config["bone_config"] = json.dumps({"bone_load": bone_load, "bone_r": bone_r})

    def setup_lora_config(self):
        st.subheader("LoRA Configuration")
        lora_load = st.text_input("LoRA Load", "")
        lora_r = st.number_input("LoRA R", value=32, min_value=1)
        lora_alpha = st.number_input("LoRA Alpha", value=32, min_value=1)
        lora_dropout = st.number_input("LoRA Dropout", value=0.0, min_value=0.0, max_value=1.0, format="%.2f")
        self.config["lora_config"] = json.dumps({
            "lora_load": lora_load,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout
        })

    def setup_pissa_config(self):
        st.subheader("PISSA Configuration")
        pissa_load = st.text_input("PISSA Load", "")
        pissa_init = st.text_input("PISSA Init", "")
        pissa_r = st.number_input("PISSA R", value=32, min_value=1)
        svd_niter = st.number_input("SVD Niter", value=4, min_value=1)
        self.config["pissa_config"] = json.dumps({
            "pissa_load": pissa_load,
            "pissa_init": pissa_init,
            "pissa_r": pissa_r,
            "svd_niter": svd_niter
        })

    def show_fixed_parameters(self):
        st.header("Fixed Parameters")
        st.write("The following parameters are fixed and cannot be changed:")
        st.code("""
        pre_ffn = 0
        head_qk = 0
        beta1 = 0.9
        beta2 = 0.99
        adam_eps = 1e-8
        accelerator = gpu
        warmup_steps = 0
        """)

    def generate_script(self):
        fixed_args = """--pre_ffn 0 --head_qk 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \\
--accelerator gpu --warmup_steps 0"""

        common_args = f"""--data_type binidx --vocab_size 65536 \\
--ctx_len {self.config['ctx_len']} --epoch_steps {self.config['epoch_steps']} --epoch_count {self.config['epoch_count']} --epoch_begin {self.config['epoch_begin']} --epoch_save {self.config['epoch_save']} --micro_bsz {self.config['micro_bsz']} \\
--n_layer {self.config['n_layer']} --n_embd {self.config['n_embd']} \\
--lr_init {self.config['lr_init']} --lr_final {self.config['lr_final']} \\
--devices {self.config['devices']} --precision {self.config['precision']} --strategy {self.config['strategy']} --grad_cp {self.config['grad_cp']} \\
{fixed_args} \\
--my_testing "x060" \\
--dataload pad {f"--accumulate_grad_batches {self.config['accumulate_grad_batches']}" if self.config['accumulate_grad_batches'] > 0 else ''}"""

        if self.config['peft'] == 'state':
            fla_arg = "--fla" if self.config.get('use_fla', False) else ""
            script = f"""python train.py --load_model {self.config['load_model']} \\
--proj_dir {self.config['proj_dir']} --data_file {self.config['data_file']} \\
{common_args} \\
--train_type "state" {fla_arg} --wandb peft-test
"""
        elif self.config['peft'] == 'bone':
            script = f"""python train.py --load_model {self.config['load_model']} \\
--proj_dir {self.config['proj_dir']} --data_file {self.config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft bone --bone_config '{self.config['bone_config']}' --wandb peft-loss
"""
        elif self.config['peft'] == 'lora':
            script = f"""python train.py --load_model {self.config['load_model']} \\
--proj_dir {self.config['proj_dir']} --data_file {self.config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft lora --lora_config '{self.config['lora_config']}' \\
--wandb peft-loss
"""
        elif self.config['peft'] == 'pissa':
            script = f"""python train.py --load_model {self.config['load_model']} \\
--proj_dir {self.config['proj_dir']} --data_file {self.config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft pissa --pissa_config '{self.config['pissa_config']}' \\
--wandb peft-loss
"""
        return script

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
        st.header("GPU Memory Usage")
        self.memory_bar = st.empty()
        self.memory_text = st.empty()

    def setup_loss_chart(self):
        self.loss_chart = st.empty()

    def setup_training_progress(self):
        st.subheader("Training Progress")
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
        st.success("Script has finished running.✨✨✨")
        st.session_state.process = None
