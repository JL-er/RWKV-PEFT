import streamlit as st
import os
import json
import subprocess
import tempfile
import signal
from collections import deque
import threading
import time
import GPUtil
import pandas as pd
import plotly.express as px

# 初始化 session state
if 'process' not in st.session_state:
    st.session_state.process = None
    

# Add these new global variables
gpu_memory_usage = 0
gpu_memory_total = 0
stop_monitoring = False

def stop_script():
    if st.session_state.process:
        os.killpg(os.getpgid(st.session_state.process.pid), signal.SIGTERM)
        st.session_state.process = None
        # 自动停止 GPU 监控
        stop_gpu_monitoring()

def generate_script(config):
    # 固定参数
    fixed_args = """--pre_ffn 0 --head_qk 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \\
--accelerator gpu --warmup_steps 0"""

    common_args = f"""--data_type binidx --vocab_size 65536 \\
--ctx_len {config['ctx_len']} --epoch_steps {config['epoch_steps']} --epoch_count {config['epoch_count']} --epoch_begin {config['epoch_begin']} --epoch_save {config['epoch_save']} --micro_bsz {config['micro_bsz']} \\
--n_layer {config['n_layer']} --n_embd {config['n_embd']} \\
--lr_init {config['lr_init']} --lr_final {config['lr_final']} \\
--devices {config['devices']} --precision {config['precision']} --strategy {config['strategy']} --grad_cp {config['grad_cp']} \\
{fixed_args} \\
--my_testing "x060" \\
--dataload pad {f"--accumulate_grad_batches {config['accumulate_grad_batches']}" if config['accumulate_grad_batches'] > 0 else ''}"""

    if config['peft'] == 'state':
        fla_arg = "--fla" if config.get('use_fla', False) else ""
        script = f"""python train.py --load_model {config['load_model']} \\
--proj_dir {config['proj_dir']} --data_file {config['data_file']} \\
{common_args} \\
--train_type "state" {fla_arg} --wandb peft-test
"""
    elif config['peft'] == 'bone':
        script = f"""python train.py --load_model {config['load_model']} \\
--proj_dir {config['proj_dir']} --data_file {config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft bone --bone_config '{config['bone_config']}' --wandb peft-loss
"""
    elif config['peft'] == 'lora':
        script = f"""python train.py --load_model {config['load_model']} \\
--proj_dir {config['proj_dir']} --data_file {config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft lora --lora_config '{config['lora_config']}' \\
--wandb peft-loss
"""
    elif config['peft'] == 'pissa':
        script = f"""python train.py --load_model {config['load_model']} \\
--proj_dir {config['proj_dir']} --data_file {config['data_file']} \\
{common_args} \\
--loss_mask pad \\
--peft pissa --pissa_config '{config['pissa_config']}' \\
--wandb peft-loss
"""
    return script

def run_script(script, working_directory):
    # 创建一个临时文件来存储脚本
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as temp_file:
        temp_file.write(script)
        temp_file_path = temp_file.name

    try:
        # 使脚本可执行
        os.chmod(temp_file_path, 0o755)

        # 使用subprocess运行脚本，设置工作目录
        st.session_state.process = subprocess.Popen(['bash', temp_file_path], 
                                   cwd=working_directory,
                                   preexec_fn=os.setsid)  # 使用 os.setsid 创建新的进程组
        
        # 自动启动 GPU 监控
        start_gpu_monitoring()
       
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    finally:
        # 删除临时文件
        os.unlink(temp_file_path)

def monitor_gpu_memory():
    global stop_monitoring, gpu_memory_usage, gpu_memory_total
    while not stop_monitoring:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory_usage = gpus[0].memoryUsed
            gpu_memory_total = gpus[0].memoryTotal
        time.sleep(1)

def start_gpu_monitoring():
    global stop_monitoring
    stop_monitoring = False
    threading.Thread(target=monitor_gpu_memory, daemon=True).start()

def stop_gpu_monitoring():
    global stop_monitoring
    stop_monitoring = True

def read_data(proj_dir):
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

def main():
    st.title("🎈 RWKV-PEFT Web Interface")
    st.write("Welcome to the RWKV-PEFT web interface!")

    # Configuration inputs
    st.header("Configuration")
    config = {}
    
    # PEFT选项改为下拉框，并放在最前面
    config["peft"] = st.selectbox(
        "Select PEFT Method",
        ("bone", "lora", "pissa", "state"),
        index=0  # 默认选择bone
    )

    # 通用配置选项
    config["load_model"] = st.text_input("Load Model Path", "/home/ryan/code/model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")
    config["proj_dir"] = st.text_input("Project Directory", "/home/ryan/code/out_model/metabone")
    config["data_file"] = st.text_input("Data File Path", "/home/ryan/code/data/roleplay")
    config["n_layer"] = st.number_input("Number of Layers", value=24, min_value=1)
    config["n_embd"] = st.number_input("Embedding Size", value=2048, min_value=1)
    config["micro_bsz"] = st.number_input("Micro Batch Size", value=4, min_value=1)
    config["epoch_count"] = st.number_input("Epoch Count", value=1, min_value=1)
    config["epoch_begin"] = st.number_input("Epoch Begin", value=0, min_value=0)
    config["epoch_save"] = st.number_input("Epoch Save", value=1, min_value=1)
    config["epoch_steps"] = st.number_input("Epoch Steps", value=50, min_value=1)
    config["ctx_len"] = st.number_input("Context Length", value=512, min_value=1)
    config["lr_init"] = st.number_input("Initial Learning Rate", value=2e-5, format="%.1e")
    config["lr_final"] = st.number_input("Final Learning Rate", value=2e-5, format="%.1e")
    config["devices"] = st.number_input("Number of Devices", value=1, min_value=1)
    config["precision"] = st.selectbox("Precision", ["bf16", "fp16", "fp32"])
    config["strategy"] = st.selectbox("Strategy", ["deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"])
    config["grad_cp"] = st.number_input("Gradient Checkpoint", value=1, min_value=0)
    config["accumulate_grad_batches"] = st.number_input("Accumulate Grad Batches", value=0, min_value=0)

    # 根据选择的PEFT方法显示相应的配置选项
    if config["peft"] == "state":
        st.subheader("State Configuration")
        config["use_fla"] = st.toggle("Use FLA", value=True)
    elif config["peft"] == "bone":
        st.subheader("Bone Configuration")
        bone_load = st.text_input("Bone Load", "")
        bone_r = st.number_input("Bone R", value=64, min_value=1)
        config["bone_config"] = json.dumps({"bone_load": bone_load, "bone_r": bone_r})
    elif config["peft"] == "lora":
        st.subheader("LoRA Configuration")
        lora_load = st.text_input("LoRA Load", "")
        lora_r = st.number_input("LoRA R", value=32, min_value=1)
        lora_alpha = st.number_input("LoRA Alpha", value=32, min_value=1)
        lora_dropout = st.number_input("LoRA Dropout", value=0.0, min_value=0.0, max_value=1.0, format="%.2f")
        config["lora_config"] = json.dumps({
            "lora_load": lora_load,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout
        })
    elif config["peft"] == "pissa":
        st.subheader("PISSA Configuration")
        pissa_load = st.text_input("PISSA Load", "")
        pissa_init = st.text_input("PISSA Init", "")
        pissa_r = st.number_input("PISSA R", value=32, min_value=1)
        svd_niter = st.number_input("SVD Niter", value=4, min_value=1)
        config["pissa_config"] = json.dumps({
            "pissa_load": pissa_load,
            "pissa_init": pissa_init,
            "pissa_r": pissa_r,
            "svd_niter": svd_niter
        })

    # 示固定参数（只读）
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

    # 添加工作目录输入
    working_directory = st.text_input("Working Directory", "/home/ryan/code/RWKV-PEFT-WEB")

    # Generate and display the script
    script = generate_script(config)
    st.header("Generated Script")
    st.code(script, language="bash")

    # Run and Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Script"):
            # 监测loss数据文件是否存在，存在则删除
            loss_file = os.path.join(config['proj_dir'], "loss_data.json")
            if os.path.exists(loss_file):
                os.remove(loss_file)
            run_script(script, working_directory)
    with col2:
        if st.button("Stop Script"):
            stop_script()

    # GPU 内存监控
    st.header("GPU Memory Usage")
    # 创建占位符
    memory_bar = st.empty()
    memory_text = st.empty()
    
    # 添加Loss图表
    # st.header("Training Loss")
    loss_chart = st.empty()
    
    # 添加速率统计占位
    st.subheader("Training Progress")
    rate_bar = st.empty()
    rate_text = st.empty()

    # 更新 GPU 内存使用情况和Loss图表
    if st.session_state.process is not None:
        placeholder = st.empty()
        loss_data = []
        current_epoch = 0
        last_progress = 0
        last_t_cost = 0
        last_kt_s = 0
        last_loss = 0
        while st.session_state.process.poll() is None:  # 检查进程是否还在运行
            memory_percentage = gpu_memory_usage / gpu_memory_total if gpu_memory_total > 0 else 0
            memory_text.text(f"GPU Memory: {gpu_memory_usage:.2f} MB / {gpu_memory_total:.2f} MB")
            memory_bar.progress(memory_percentage)
            placeholder.text("Script is running. GPU memory usage is being updated.")
            
            # 更新Loss图表和速率统计
            new_loss_data, t_cost, kt_s, loss = read_data(config['proj_dir'])
            
            # 计算当前epoch和总体进度
            current_epoch = min(int(len(new_loss_data) / config['epoch_steps']), config['epoch_count'] - 1)
            total_progress = min(len(new_loss_data) / (config['epoch_steps'] * config['epoch_count']), 1.0)
            
            # 更新进度条和统计信息
            if total_progress > last_progress:
                last_progress = total_progress
                last_t_cost = t_cost
                last_kt_s = kt_s
                last_loss = loss
            
            rate_bar.progress(last_progress)
            rate_text.text(f"Epoch {current_epoch + 1}/{config['epoch_count']}: {last_progress:.2%} complete "
                           f"| it/s: {last_t_cost:.2f} | Kt/s: {last_kt_s:.2f} | Loss: {last_loss:.4f}")
            
            if len(new_loss_data) > len(loss_data):
                loss_data = new_loss_data
                steps = range(1, len(loss_data) + 1)
                df = pd.DataFrame({'step': steps, 'loss': loss_data})
                
                # 创建图表
                fig = px.line(df, x='step', y='loss', title='Training Loss')
                fig.update_layout(xaxis_title='Epoch Step', yaxis_title='Loss')
                # 如果accumulate_grad_batches不为0，则将x轴范围设置为总步数除以accumulate_grad_batches 取整
                if config['accumulate_grad_batches'] > 0:
                    fig.update_xaxes(range=[1, (config['epoch_steps'] * config['epoch_count']) // config['accumulate_grad_batches']])
                else:
                    fig.update_xaxes(range=[1, config['epoch_steps'] * config['epoch_count']])  # 设置x轴范围为总步数
                loss_chart.plotly_chart(fig, use_container_width=True)

            time.sleep(1)
            placeholder.empty()
        
        # 确保训练结束后显示 100% 进度
        rate_bar.progress(1.0)
        rate_text.text(f"Training Complete: 100.00% | it/s: {last_t_cost:.2f} | Kt/s: {last_kt_s:.2f} | Loss: {last_loss:.4f}")
        
        # 进程结束后自动停止 GPU 监控
        stop_gpu_monitoring()
        st.success("Script has finished running.✨✨✨")
        st.session_state.process = None
    else:
        st.warning("Script is not running.")

if __name__ == "__main__":
    main()
