import gradio as gr
import subprocess
from utils import get_files, get_project_root, read_training_data, clean_training_data, set_error_message, get_error_message, delete_error
import json
from pathlib import Path
import threading
import signal
import plotly.graph_objects as go
import time
import GPUtil
import json
import queue
import os
os.environ['PYTHONUNBUFFERED'] = '1'

class TrainingInterface:
    def __init__(self):
        self.model_config = {
            "0.1B": {"n_layer": 12, "n_embd": 768},
            "0.4B": {"n_layer": 24, "n_embd": 1024},
            "1.6B": {"n_layer": 24, "n_embd": 2048},
            "3B": {"n_layer": 32, "n_embd": 2560},
            "7B": {"n_layer": 32, "n_embd": 4096},
            "14B": {"n_layer": 61, "n_embd": 4096}
        }
        
        # Initialize components dictionary
        self.components = {}
        self.process = None
        self.gpu_loss_thread = None
        self.stop_monitoring = True
        self.update_queue = queue.Queue()  # 添加队列用于更新信号
        self.interface = self.create_interface()

    # 更新数据文件列表
    def update_data_file_list(self, file_path, file_type):
        path = Path(file_path)
        if not path.exists():
            return gr.update(choices=[], value=None)
        
        # 文件扩展名
        file_extensions = {
            "binidx": (".bin", ".idx"),
            "jsonl": (".jsonl",)
        }
        
        required_exts = file_extensions.get(file_type, [])
        valid_files = []
        
        if file_type == "binidx":
            base_names = {f.rsplit('.', 1)[0] for f in os.listdir(file_path) if any(f.endswith(ext) for ext in required_exts)}
            valid_files = sorted([f for f in base_names if all(os.path.exists(os.path.join(file_path, f"{f}{ext}")) for ext in required_exts)])
        else:
            valid_files = sorted([f.rsplit('.', 1)[0] for f in os.listdir(file_path) if f.endswith('.jsonl')])
        
        return gr.update(choices=valid_files, value=valid_files[0] if valid_files else None)

    # 更新文件列表
    def update_file_list(self, file_path, file_type):
        file_list = get_files(file_path, file_type)
        return gr.update(choices=file_list, value=file_list[0] if file_list else None)
    
    # 更新 PEFT 方法配置的可见性
    def update_peft_method(self, peft_method):
        visibility_map = {
            "bone": [True, True, False, False, False, False, False, False, False, False],
            "lora": [False, False, True, True, True, True, False, False, False, False],
            "pissa": [False, False, False, False, False, False, True, True, True, True],
            "state": [False] * 10
        }
        
        visibility_settings = visibility_map.get(peft_method, [False] * 10)
        return tuple(gr.update(visible=v) for v in visibility_settings)

    # 更新模型配置
    def update_model_config(self, model_size):
        return gr.update(value=self.model_config[model_size]["n_layer"]), gr.update(value=self.model_config[model_size]["n_embd"])
    
    # 更新组件的可见性
    def update_visibility(self, is_visible):
        return gr.update(visible=is_visible)
    
    # 事件监听
    def listen_events(self):
        # 监听 peft_method 的变化
        self.components['peft_method'].change(
            fn=self.update_peft_method,
            inputs=[self.components['peft_method']],
            outputs=[self.components['bone_load'], self.components['bone_r'], self.components['lora_load'], self.components['lora_r'], self.components['lora_alpha'], self.components['lora_dropout'], self.components['pissa_load'], self.components['pissa_init'], self.components['pissa_r'], self.components['svd_niter']]
        )
        # 监听 data_path 变化，更新 data_file 列表
        self.components['data_path'].change(
            fn=self.update_data_file_list,
            inputs=[self.components['data_path'], self.components['data_type']],
            outputs=[self.components['data_file']]
        )
        # 监听 data_type 变化，更新 data_file 列表
        self.components['data_type'].change(
            fn=self.update_data_file_list,
            inputs=[self.components['data_path'], self.components['data_type']],
            outputs=[self.components['data_file']]
        )
        # 监听 model_path 变化，更新 model_file 列表
        self.components['model_path'].change(
            fn=self.update_file_list,
            inputs=[self.components['model_path'], gr.State("pth")],
            outputs=[self.components['model_file']]
        )
        # 监听 model_size 变化，更新 n_layer 和 n_embd
        self.components['model_size'].change(
            fn=self.update_model_config,
            inputs=[self.components['model_size']],
            outputs=[self.components['n_layer'], self.components['n_embd']]
        )
        # 监听 wandb 变化，更新 wandb_project 的可见性
        self.components['wandb'].change(
            fn=self.update_visibility,
            inputs=[self.components['wandb']],
            outputs=[self.components['wandb_project']]
        )
        # 监听 train_type 变化，更新 train_type_ctx 的可见性
        self.components['train_type'].change(
            fn=self.update_visibility,
            inputs=[self.components['train_type']],
            outputs=[self.components['train_type_ctx']]
        )
        # 监听 loss_cha
    
    # 创建界面
    def create_interface(self):
        with gr.Blocks() as training_interface:
            with gr.Row():
                with gr.Accordion("Basic Configurations"):
                    with gr.Group():
                        with gr.Row():
                            self.components['rwkv_version'] = gr.Dropdown(
                                choices=["v5", "v6"],
                                label="RWKV Version",
                                value="v6"
                            )
                            self.components['peft_method'] = gr.Dropdown(
                                choices=["bone", "lora", "pissa", "state"],
                                label="PEFT Method",
                                value="bone"
                            )
                        with gr.Row():
                            # Conditional components based on PEFT method
                            self.components['bone_load'] = gr.Textbox(label="Bone Load", visible=True)
                            self.components['bone_r'] = gr.Number(label="Bone R", value=64, minimum=1, visible=True)
                        
                        self.components['quant'] = gr.Dropdown(
                            choices=["none", "int8", "nf4"],
                            label="Quant",
                            value="none"
                        )
                        self.components['output_path'] = gr.Textbox(
                            label="Output Path",
                            value="/home/rwkv/out_model/your_model_name"
                        )
                        
                        # LoRA
                        self.components['lora_load'] = gr.Textbox(label="LoRA Load", visible=False)

                        with gr.Row():
                            self.components['lora_r'] = gr.Number(label="LoRA R", value=32, minimum=1, visible=False)
                            self.components['lora_alpha'] = gr.Number(label="LoRA Alpha", value=32, minimum=1, visible=False)
                            self.components['lora_dropout'] = gr.Number(label="LoRA Dropout", minimum=0, value=0, visible=False)

                        # PISSA
                        self.components['pissa_load'] = gr.Textbox(label="PISSA Load", visible=False)
                        self.components['pissa_init'] = gr.Textbox(label="PISSA Init", visible=False)
                        with gr.Row():
                            self.components['pissa_r'] = gr.Number(label="PISSA R", value=32, minimum=1, visible=False)
                            self.components['svd_niter'] = gr.Number(label="SVD Niter", value=4, minimum=1, visible=False)

                with gr.Accordion("Data Configurations"):
                    with gr.Group():
                        with gr.Row():
                            self.components['data_path'] = gr.Textbox(
                                label="Data Path",
                                value="/home/rwkv/your_data_path"
                            )
                            self.components['data_file'] = gr.Dropdown(
                                label="Data File",
                                choices=[],
                                interactive=True
                            )
                        with gr.Row():
                            self.components['data_load'] = gr.Dropdown(
                                label="Data Load",
                                choices=["pad", "get", "only"],
                                interactive=True
                            )
                            self.components['loss_mask'] = gr.Dropdown(
                                label="Loss Mask",
                                choices=["none", "pad", "qa"],
                                interactive=True
                            )
                        with gr.Row():
                            with gr.Column():
                                self.components['data_type'] = gr.Dropdown(
                                    label="Data Type",
                                    choices=["binidx", "jsonl"],
                                    interactive=True
                                )
                            with gr.Column():
                                self.components['vocab_size'] = gr.Textbox(
                                    label="Vocab Size",
                                    value="65536",
                                    interactive=False
                                )
                        self.components['data_shuffle'] = gr.Checkbox(
                            label="Data Shuffle",
                            value=True,
                            interactive=True
                        )

                with gr.Accordion("Model Configurations"):
                    with gr.Group():
                        with gr.Row():
                            self.components['model_path'] = gr.Textbox(
                                label="Model Path",
                                value="/home/rwkv/your_model_path"
                            )
                            self.components['model_file'] = gr.Dropdown(
                                label="Model File",
                                choices=[],
                                interactive=True
                            )
                        self.components['model_size'] = gr.Dropdown(
                            choices=list(self.model_config.keys()),
                            label="Model Size",
                            value="1.6B"
                        )
                        self.components['n_layer'] = gr.Number(
                            label="Number of Layers",
                            value=self.model_config["1.6B"]["n_layer"],
                            minimum=1
                        )
                        self.components['n_embd'] = gr.Number(
                            label="Embedding Size",
                            value=self.model_config["1.6B"]["n_embd"],
                            minimum=1
                        )
                        self.components['train_parts'] = gr.CheckboxGroup(
                            choices=["emb", "head", "time", "ln"],
                            label="Train Parts",
                            value=["time", "ln"]
                        )
            
            with gr.Accordion("Training Configurations"):
                with gr.Group():
                    with gr.Row():
                        self.components['micro_bsz'] = gr.Number(label="Micro Batch Size", value=4, minimum=1)
                        self.components['epoch_count'] = gr.Number(label="Epoch Count", value=1, minimum=1)
                    with gr.Row():
                        self.components['epoch_steps'] = gr.Number(label="Epoch Steps", value=1000, minimum=1)
                        self.components['epoch_save'] = gr.Number(label="Epoch Save", value=1, minimum=1)
                    with gr.Row():
                        self.components['epoch_begin'] = gr.Number(label="Epoch Begin", value=0, minimum=0)
                        self.components['ctx_len'] = gr.Number(label="Context Length", value=512, minimum=1)
                    with gr.Row():
                        self.components['lr_init'] = gr.Number(label="Initial Learning Rate", value=2e-5)
                        self.components['lr_final'] = gr.Number(label="Final Learning Rate", value=2e-5)
                    with gr.Row():
                        self.components['strategy'] = gr.Dropdown(
                            choices=[
                                "deepspeed_stage_1",
                                "deepspeed_stage_2",
                                "deepspeed_stage_2_offload",
                                "deepspeed_stage_3",
                                "deepspeed_stage_3_offload"
                            ],
                            label="Strategy",
                            value="deepspeed_stage_1"
                        )
                        self.components['precision'] = gr.Dropdown(
                            choices=["bf16", "fp16", "fp32"],
                            label="Precision",
                            value="bf16"
                        )
                    with gr.Row():
                        self.components['warmup_steps'] = gr.Number(label="Warmup Steps", value=0, minimum=0)
                        self.components['accumulate_grad_batches'] = gr.Number(label="Accumulate Grad Batches", value=0, minimum=0)
                        self.components['devices'] = gr.Number(label="Number of Devices", value=1, minimum=1)
                    with gr.Row():
                        self.components['use_fla'] = gr.Checkbox(label="Use FLA", value=False)
                        self.components['grad_cp'] = gr.Checkbox(label="Gradient Checkpoint", value=True)
                    with gr.Row():
                        self.components['wandb'] = gr.Checkbox(label="Use Wandb", value=False)
                        self.components['wandb_project'] = gr.Textbox(
                            label="Wandb Project",
                            value="peft-loss",
                            visible=False
                        )
                        self.components['train_type'] = gr.Checkbox(label="Infctx", value=False)
                        self.components['train_type_ctx'] = gr.Dropdown(label="Chunk Context", choices=[128, 256, 512, 1024, 2048, 4096], value=512, visible=False)

            # 活动监控部分
            with gr.Accordion("Activity Monitor"):
                with gr.Row():
                    # Loss图表
                    with gr.Column():
                        loss_chart = gr.Plot(label="Training Loss")
                with gr.Row():
                    with gr.Column(min_width=950):
                        # 训练进度
                        progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Training Progress (%)", interactive=False)
                        progress_text = gr.Textbox(label="Training Status", interactive=False)
                        gpu_text = gr.Textbox(label="GPU Status", interactive=False)
                    # GPU监控
                    with gr.Column(min_width=150):
                        gpu_chart = gr.Plot(label="GPU Memory Usage")

            # 保存组件引用
            self.components.update({
                'gpu_chart': gpu_chart,
                'gpu_text': gpu_text,
                'loss_chart': loss_chart,
                'progress_bar': progress_bar,
                'progress_text': progress_text
            })

            # Command Preview and Run Buttons
            with gr.Row():
                preview_button = gr.Button("Preview Command")
                run_button = gr.Button("Run Training")
                stop_button = gr.Button("Stop Training")

            command_output = gr.Code(label="Command Preview", language="shell")
            training_status = gr.Textbox(label="Training Status")

            # 监听事件
            self.listen_events()

            # 预览命令
            preview_button.click(
                fn=self.generate_command,
                inputs=list(self.components.values()),
                outputs=[command_output]
            )
            # 修改运行按钮的处理函数
            run_button.click(
                fn=self.start_and_monitor,
                inputs=list(self.components.values()),
                outputs=[
                    self.components['gpu_chart'],
                    self.components['gpu_text'],
                    self.components['progress_bar'],
                    self.components['progress_text'],
                    self.components['loss_chart'],
                    training_status
                ]
            )
            stop_button.click(
                fn=self.stop_monitoring_updates,
                outputs=[training_status]
            )

        return training_interface

    # 启动训练并监控进度
    def start_and_monitor(self, *args):
        gr.Info("Starting process...")
        # 创建配置字典
        config = dict(zip(self.components.keys(), args))
        
        # 启动训练
        initial_status = self.start_training(*args)
        # 创建初始的图表
        initial_fig = go.Figure()
        initial_fig.update_layout(title='Training Loss')
        initial_gpu_fig = go.Figure()
        initial_gpu_fig.update_layout(
            margin=dict(l=10, r=10, t=15, b=10),
            height=250
        )
        
        # 清理训练数据
        clean_training_data(self.components['output_path'].value)
        
        # 启动 GPU 和 loss 监控线程
        self.gpu_loss_thread = threading.Thread(
            target=self.monitor_gpu_loss,
            args=(config,),  # 传入配置字典
            daemon=True
        )
        self.gpu_loss_thread.start()
        
        # 监控训练进度
        while not self.stop_monitoring:
            try:
                if get_error_message():
                    yield initial_gpu_fig, None, 0, "Training stopped", initial_fig, get_error_message()
                    break
                # 检查队列更新
                update_data = self.check_updates()
                if update_data[0] is not None:  # 如果有更新数据
                    yield update_data + (initial_status,)
                time.sleep(0.1)  # 减少检查间隔
            except Exception as e:
                yield initial_gpu_fig, None, 0, "Training stopped", initial_fig, f"Error: {str(e)}"
        if get_error_message():
            err_txt = get_error_message()
            delete_error()
            raise gr.Error(err_txt)
        # 如果监控停止，返回最后一次的状态
        return initial_gpu_fig, None, 0, "Training stopped", initial_fig, "Training process has been stopped."

    def start_training(self, *args):
        """启动训练进程"""
        try:
            # 生成训练脚本
            script = self.generate_command(*args)
            config = dict(zip(self.components.keys(), args))
            os.makedirs(config['output_path'], exist_ok=True)
            script_path = os.path.join(config['output_path'], 'training_script.sh')
            with open(script_path, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)
            
            # 执行脚本
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            self.process = subprocess.Popen(
                ['bash', script_path],
                cwd=get_project_root("train.py"),
                preexec_fn=os.setsid,
            )
            self.stop_monitoring = False
            return "Training started successfully! please wait..."
        
        except Exception as e:
            set_error_message(f"Error starting training: {str(e)}")
            return f"Error starting training: {str(e)}"

    # 停止监控更新
    def stop_monitoring_updates(self):
        """停止监控更新"""
        try:
            self.stop_monitoring = True
            if self.process:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            
            # 等待监控线程结束
            if hasattr(self, 'gpu_loss_thread') and self.gpu_loss_thread and self.gpu_loss_thread.is_alive():
                self.gpu_loss_thread.join(timeout=5)
            
            self.process = None
            return "Training process has been stopped."
        except Exception as e:
            return f"Error stopping training process: {str(e)}"

    # 检查队列中的更新数据
    def check_updates(self):
        """检查队列中的更新数据"""
        try:
            update_data = self.update_queue.get_nowait()
            return (
                update_data.get('gpu_fig', None),
                update_data.get('gpu_status', None),
                update_data.get('progress', None),
                update_data.get('status', None),
                update_data.get('loss_fig', None)
            )
        except queue.Empty:
            return None, None, None, None, None

    # 生成命令
    def generate_command(self, *args):
        """
        生成训练脚本命令
        """
        config = dict(zip(self.components.keys(), args))
        try:
            cmd_parts = [
                "python train.py",
                f"--load_model {os.path.join(config['model_path'], config['model_file'])}",
                f"--proj_dir {config['output_path']} --data_file {os.path.join(config['data_path'], config['data_file'])}",
                f"--data_type {config['data_type']} --vocab_size {config['vocab_size']}",
                f"--ctx_len {config['ctx_len']} --epoch_steps {config['epoch_steps']} --epoch_count {config['epoch_count']}",
                f"--epoch_begin {config['epoch_begin']} --epoch_save {config['epoch_save']} --micro_bsz {config['micro_bsz']}",
                f"--n_layer {config['n_layer']} --n_embd {config['n_embd']}",
                f"--lr_init {config['lr_init']} --lr_final {config['lr_final']} --warmup_steps {config['warmup_steps']}",
                "--pre_ffn 0 --head_qk 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --accelerator gpu",
                f"--devices {config['devices']} --precision {config['precision']} --strategy {config['strategy']} --grad_cp {1 if config['grad_cp'] else 0}",
                f"--dataload {config['data_load']} --loss_mask {config['loss_mask']} --data_shuffle {1 if config['data_shuffle'] else 0}"
            ]
            
            optional_flags = {
                "use_fla": "--fla",
                "rwkv_version": '--my_testing "x060"' if config['rwkv_version'] == "v6" else "",
                "accumulate_grad_batches": f"--accumulate_grad_batches {config['accumulate_grad_batches']}" if config['accumulate_grad_batches'] > 0 else "",
                "wandb": f"--wandb {config['wandb_project']}" if config['wandb'] else "",
                "quant": f"--quant {config['quant']}" if config['quant'] != "none" else ""
            }
            
            cmd_parts.extend(value for flag, value in optional_flags.items() if value)
            
            if config['peft_method'] != 'state':
                cmd_parts.append(f"--peft {config['peft_method']}")
                
                # Add PEFT-specific settings
                if config['peft_method'] == 'bone':
                    bone_config = {
                        'bone_load': config['bone_load'],
                        'bone_r': config['bone_r']
                    }
                    cmd_parts.append(f"--bone_config '{json.dumps(bone_config)}'")
                    
                elif config['peft_method'] == 'lora':
                    lora_config = {
                        'lora_load': config['lora_load'],
                        'lora_r': config['lora_r'],
                        'lora_alpha': config['lora_alpha'],
                        'lora_dropout': config['lora_dropout']
                    }
                    cmd_parts.append(f"--lora_config '{json.dumps(lora_config)}'")
                    
                elif config['peft_method'] == 'pissa':
                    pissa_config = {
                        'pissa_load': config['pissa_load'],
                        'pissa_init': config['pissa_init'],
                        'pissa_r': config['pissa_r'],
                        'svd_niter': config['svd_niter']
                    }
                    cmd_parts.append(f"--pissa_config '{json.dumps(pissa_config)}'")
            
            # Train type conditional arguments
            if config['train_type']:
                cmd_parts.append("--train_type infctx")
                if config['train_type_ctx']:
                    cmd_parts.append(f"--chunk_ctx {config['train_type_ctx']}")
        
            # 合并所有参数并返回完整命令
            return ' \\\n'.join(cmd_parts)
        except Exception as e:
            # 检查 data_file、model_file 是否为空
            if not config['data_file'] or not config['model_file']:
                raise gr.Error("Error: Data file or model file is not selected")
            else:
                raise gr.Error(f"Error: {str(e)}")

    # 监听GPU状态和训练损失
    def monitor_gpu_loss(self, config):
        """监控GPU使用情况和训练损失"""   
        while not self.stop_monitoring:
            try:
                # 读取GPU数据
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_used = gpu.memoryUsed
                    memory_total = gpu.memoryTotal
                    
                    # 创建GPU使用率图表，使用Bar图表，memory_total为最大值，memory_used为当前使用量，单位为GB，
                    gpu_fig = go.Figure(go.Bar(x=[gpu.name], y=[memory_used/1024], name='GPU Memory Used (GB)'))
                    gpu_fig.update_layout(
                        margin=dict(l=10, r=10, t=15, b=10),  # 设置左、右、上、下的边距
                        yaxis=dict(range=[0, memory_total/1024]),
                        height=250  # 设置图表高度为200像素
                    )
                    
                    # 准备更新数据
                    update_data = {
                        'gpu_fig': gpu_fig,
                        'gpu_status': f"GPU Memory: {memory_used:.2f}MB / {memory_total:.2f}MB"
                    }
                    
                    # 读取训练数据
                    loss_data, t_cost, kt_s, loss = read_training_data(config['output_path'])
                    if loss_data:
                        # 计算当前进度
                        total_steps = config['epoch_steps'] * config['epoch_count']
                        current_step = len(loss_data)
                        total_progress = min(current_step / (total_steps // config['devices']), 1.0)
                        current_epoch = min(int(current_step / config['epoch_steps']), config['epoch_count'] - 1)
                        # 创建loss图表
                        loss_fig = go.Figure()
                        loss_fig.add_trace(go.Scatter(y=loss_data, mode='lines', name='Training Loss'))
                        loss_fig.update_layout(
                            title='Training Loss',
                            xaxis_title='Step',
                            yaxis_title='Loss',
                            template="plotly_white"
                        )
                        
                        # 添加训练相关的更新数据
                        update_data.update({
                            'progress': total_progress * 100,
                            'status': f"Epoch {current_epoch + 1}/{config['epoch_count']}: {total_progress:.2%} complete | it/s: {t_cost:.2f} | Kt/s: {kt_s:.2f} | Loss: {loss:.4f}",
                            'loss_fig': loss_fig
                        })
                    # 将所有更新数据一次性放入队列
                    self.update_queue.put(update_data)
                
                time.sleep(1)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)

    # 停止训练
    def stop_training(self):
        """停止训练"""
        try:
            if self.process is None:
                return "No training process is running."
            # 停止训练进程
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # 等待进程结束，最多等待5秒
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果进程没有响应SIGTERM，则强制终止
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            
            # 停止所有监控线程
            self.stop_monitoring = True
            if self.gpu_thread:
                self.gpu_thread.join()
            if self.progress_thread:
                self.progress_thread.join()
            if hasattr(self, 'output_thread') and self.output_thread:
                self.output_thread.join()
            
            self.process = None
            
            # 重置UI
            self.components['progress_bar'].value = 0
            self.components['progress_text'].value = "Training stopped"
            self.components['gpu_text'].value = ""
            self.components['gpu_chart'].value = None
            self.components['loss_chart'].value = None
            
            return "Training process has been stopped."
        except Exception as e:
            return f"Error stopping training process: {str(e)}"
    
    # 初始化组件
    def init_file_selection(self):
        # Get the updates
        data_files_update = self.update_data_file_list(
            self.components['data_path'].value,
            self.components['data_type'].value
        )
        model_files_update = self.update_file_list(
            self.components['model_path'].value,
            "pth"
        )
        # Access the dictionary values instead of attributes
        self.components['data_file'].choices = data_files_update.get('choices', [])
        self.components['data_file'].value = data_files_update.get('value', None)
        self.components['model_file'].choices = model_files_update.get('choices', [])
        self.components['model_file'].value = model_files_update.get('value', None)

    # 启动界面
    def launch(self, *args, **kwargs):
        # 首次加载时，执行一次组件的初始化
        self.interface.launch(*args, **kwargs)
