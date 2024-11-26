import gradio as gr
import os
import subprocess
from utils import get_files

class MergeInterface:
    def __init__(self):
        self.components = {
            "merge_type": None,
            "quant": None,
            "output_path": None,
            "model_path": None,
            "checkpoint_path": None,
            "pissa_init": None,
            "lora_alpha": None
        }
        self.interface = self.create_interface()
        
    def generate_command(self, merge_type=None, quant=None, output_path=None, model_path=None, model_file=None, checkpoint_path=None, checkpoint_file=None, pissa_init=None, pissa_file=None, lora_alpha=None):
        # 如果参数为 None，则使用组件的值
        merge_type = merge_type or self.components['merge_type'].value
        quant = quant or self.components['quant'].value
        output_path = output_path or self.components['output_path'].value
        model_path = model_path or self.components['model_path'].value
        model_file = model_file or self.components['model_file'].value
        checkpoint_path = checkpoint_path or self.components['checkpoint_path'].value
        checkpoint_file = checkpoint_file or self.components['checkpoint_file'].value
        pissa_init = pissa_init or self.components['pissa_init'].value
        pissa_file = pissa_file or self.components['pissa_file'].value
        lora_alpha = lora_alpha or self.components['lora_alpha'].value
        
        # 验证必要参数
        if not all([merge_type, quant, output_path, model_path, model_file, checkpoint_path, checkpoint_file]):
            return "Please fill in all required fields (Merge Type, Quant, Output Path, Model Path, Model File, Checkpoint Path, Checkpoint File)"
        
        merge_types_dict = {
            "bone": {
                "script": "merge/merge_bone.py",
                "specific_args": f"--lora_checkpoint {checkpoint_path + '/' + checkpoint_file}"
            },
            "pissa": {
                "script": "merge/merge.py",
                "specific_args": f"--type {merge_type} \\\n--lora_checkpoint {checkpoint_path + '/' + checkpoint_file} \\\n--lora_init {pissa_init + '/' + str(pissa_file)}"
            },
            "lora": {
                "script": "merge/merge.py",
                "specific_args": f"--type {merge_type} \\\n--lora_checkpoint {checkpoint_path + '/' + checkpoint_file} \\\n--lora_alpha {lora_alpha}"
            },
            "state": {
                "script": "merge/merge_state.py",
                "specific_args": f"--state_checkpoint {checkpoint_path + '/' + checkpoint_file}"
            }
        }
        script = merge_types_dict[merge_type]["script"]
        specific_args = merge_types_dict[merge_type]["specific_args"]
        quant_arg = f"--quant {quant}" if merge_type != "state" else ""
        
        command = f"python {script} \\\n--base_model {model_path + '/' + str(model_file)}"
        command += f" \\\n--output {output_path}"
        if quant_arg:
            command += f" \\\n{quant_arg}"
        command += f" \\\n{specific_args}"
        
        return command
    
    # 更新文件列表
    def update_file_list(self, file_path, file_type):
        file_list = get_files(file_path, file_type)
        return gr.update(choices=file_list, value=file_list[0] if file_list else None)
    
    def update_visibility(self, merge_type):
        pissa_visible = merge_type == "pissa"
        lora_visible = merge_type == "lora"
        return [
            gr.update(visible=pissa_visible),
            gr.update(visible=lora_visible),
            gr.update(visible=pissa_visible)
        ]

    def run_merge(self, merge_type, quant, output_path, model_path, model_file, checkpoint_path, checkpoint_file, pissa_init, pissa_file, lora_alpha):
        # Now kwargs will contain all the inputs
        command = self.generate_command(merge_type, quant, output_path, model_path, model_file, checkpoint_path, checkpoint_file, pissa_init, pissa_file, lora_alpha)
        
        try:
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            return f"Merge completed successfully!\n\nCommand output:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error during merge:\n{e.stderr}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def create_interface(self):
        with gr.Blocks() as merge_interface:
            with gr.Row():
                with gr.Column():
                    # gr.Markdown("## Basic Configurations")
                    self.components['merge_type'] = gr.Dropdown(
                        choices=["bone", "pissa", "lora", "state"],
                        label="Merge Type",
                        value="bone",
                        interactive=True
                    )
                    self.components['quant'] = gr.Dropdown(
                        choices=["none", "int8", "nf4"],
                        label="Quant",
                        value="none",
                        interactive=True
                    )
                    self.components['output_path'] = gr.Textbox(
                        label="Output Path",
                        value="/home/rwkv/model/output.pth",
                        interactive=True
                    )
                
                with gr.Column():
                    # gr.Markdown("## Model Configurations")
                    # Model Path
                    self.components['model_path'] = gr.Textbox(
                        label="Model Path",
                        value="/home/rwkv/your_model_directory",
                        interactive=True,
                    )
                    self.components['model_file'] = gr.Dropdown(
                        label="Model File",
                        choices=[],
                        interactive=True
                    )

                    # Checkpoint Path
                    self.components['checkpoint_path'] = gr.Textbox(
                        label="LoRA Checkpoint Path",
                        value="/home/rwkv/your_checkpoint_directory",
                        interactive=True,
                    )
                    self.components['checkpoint_file'] = gr.Dropdown(
                        label="LoRA Checkpoint File",
                        choices=[],
                        interactive=True
                    )
                    
                    # PISSA Init Path
                    self.components['pissa_init'] = gr.Textbox(
                        label="PISSA Init Path",
                        value="/home/rwkv/your_pissa_init_directory",
                        interactive=True,
                        visible=False
                    )
                    self.components['pissa_file'] = gr.Dropdown(
                        label="PISSA Init File",
                        choices=[],
                        interactive=True,
                        visible=False
                    )
                    
                    # LoRA Alpha
                    self.components['lora_alpha'] = gr.Number(
                        label="LoRA Alpha",
                        interactive=True,
                        visible=False,
                        value=32,
                        minimum=1,
                    )
                    
            # 显示 command 预览 折叠
            with gr.Accordion("Command Preview"):
                # 增加按钮，点击刷新
                check_button = gr.Button("Check Command")
                command_textbox = gr.Code(label="shell", language="shell", interactive=False, lines=6, value=self.generate_command())

            merge_button = gr.Button("Run Merge")
            # Merge 状态
            with gr.Accordion("Merge Status"):
                output_text = gr.Code(label="", lines=10, interactive=False, value="No merge run yet")

            # 设置可见性更新事件
            self.components['merge_type'].change(
                fn=self.update_visibility,
                inputs=[self.components['merge_type']],
                outputs=[self.components['pissa_init'], self.components['lora_alpha'], self.components['pissa_file']]
            )

            # 添加 model_path 的 change 事件
            self.components['model_path'].change(
                fn=self.update_file_list,
                inputs=[self.components['model_path'], gr.State("pth")],
                outputs=[self.components['model_file']]
            )
            # 添加 checkpoint_path 的 change 事件
            self.components['checkpoint_path'].change(
                fn=self.update_file_list,
                inputs=[self.components['checkpoint_path'], gr.State("pth")],
                outputs=[self.components['checkpoint_file']]
            )
            # 添加 pissa_init 的 change 事件
            self.components['pissa_init'].change(
                fn=self.update_file_list,
                inputs=[self.components['pissa_init'], gr.State("pth")],
                outputs=[self.components['pissa_file']]
            )
            # 添加 check_button 的 click 事件
            check_button.click(
                fn=self.generate_command,
                inputs=[
                    self.components['merge_type'],
                    self.components['quant'],
                    self.components['output_path'],
                    self.components['model_path'],
                    self.components['model_file'],
                    self.components['checkpoint_path'],
                    self.components['checkpoint_file'],
                    self.components['pissa_init'],
                    self.components['pissa_file'],
                    self.components['lora_alpha']
                ],
                outputs=[command_textbox]
            )

            # 使用字典解包来简化参数传递
            merge_button.click(
                fn=self.run_merge,
                inputs=[
                    self.components['merge_type'],
                    self.components['quant'],
                    self.components['output_path'],
                    self.components['model_path'],
                    self.components['model_file'],
                    self.components['checkpoint_path'],
                    self.components['checkpoint_file'],
                    self.components['pissa_init'],
                    self.components['pissa_file'],
                    self.components['lora_alpha']
                ],
                outputs=[output_text]
            )

        return merge_interface

    def launch(self, *args, **kwargs):
        self.interface.launch(*args, **kwargs)
