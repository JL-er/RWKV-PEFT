import re
import gradio as gr
import os
import subprocess
from utils import get_project_root, get_files, calculate_epoch_steps

class DataInterface:
    def __init__(self):
        self.components = {
            "data_path": None,
            "data_file": None,
            "output_name": None,
            "vocab": os.path.join(get_project_root('train.py') + '/json2binidx_tool', 'rwkv_vocab_v20230424.txt'),
            "append_eod": True
        }
        self.interface = self.create_interface()
    
    # 更新文件列表
    def update_file_list(self, file_path, file_type):
        file_list = get_files(file_path, file_type)
        return gr.update(choices=file_list, value=file_list[0] if file_list else None)
    
    # 生成数据命令
    def generate_command(self, data_path=None, data_file=None, output_name=None, append_eod=None):
        # 使用传入的参数值，如果没有传入则使用组件中保存的值
        data_path = data_path or self.components['data_path'].value
        data_file = data_file or self.components['data_file'].value
        output_name = output_name or self.components['output_name'].value
        append_eod = append_eod if append_eod is not None else self.components['append_eod'].value
    
        # 验证必要参数
        if not all([data_path, data_file, output_name]):
            return "Please fill in all required fields (Data Path, Data File, and Output Name)"

        script = "json2binidx_tool/tools/preprocess_data.py"
        common_args = f"--input {os.path.join(data_path, data_file)} \\\n--output-prefix {os.path.join(data_path, output_name)} \\\n--vocab {self.components['vocab']} \\\n--dataset-impl mmap \\\n--tokenizer-type RWKVTokenizer"
        if append_eod:
            common_args += " \\\n--append-eod"
        return f"""python {script} {common_args}"""
    
    # 运行数据
    def run_data(self, data_path=None, data_file=None, output_name=None, append_eod=None, batch_size=1):
        command = self.generate_command(data_path, data_file, output_name, append_eod)
        try:
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            # 提取数据条数
            data_nums_pattern = r"data_nums: (\d+)"
            data_nums_match = re.search(data_nums_pattern, result.stdout)
            if data_nums_match:
                data_count = int(data_nums_match.group(1))
                # 计算epoch步数
                epoch_steps = calculate_epoch_steps(
                    data_count, 
                    batch_size
                )
                # 返回多个值以更新界面
                return [
                    f"Data completed successfully!\n\nCommand output:\n{result.stdout}",
                    data_count,
                    epoch_steps
                ]
            return [f"Data completed successfully!\n\nCommand output:\n{result.stdout}", None, None]
        except subprocess.CalledProcessError as e:
            return [f"Error during data:\n{e.stderr}", None, None]
        except Exception as e:
            return [f"Unexpected error: {str(e)}", None, None]

    # 创建数据界面
    def create_interface(self):
        with gr.Blocks() as data_interface:
            # Data Path
            with gr.Group():
                with gr.Row():
                    self.components['data_path'] = gr.Textbox(
                        label="Data Path",
                        value="/home/rwkv/your_data_path",
                        interactive=True
                    )
                    self.components['data_file'] = gr.Dropdown(
                        label="Data File",
                        choices=[],
                        interactive=True
                    )
                # Output Name
                self.components['output_name'] = gr.Textbox(
                    label="Output Name",
                    value="your_data_output_name",
                    interactive=True
                )
                # Append EOD
                self.components['append_eod'] = gr.Checkbox(
                    label="Append EOD",
                    value=True,
                    interactive=True
                )
            
            # 显示 command 预览 折叠
            with gr.Accordion("Command Preview"):
                # 增加按钮，点击刷新
                check_button = gr.Button("Check Command")
                command_textbox = gr.Code(label="shell", language="shell", interactive=False, lines=6, value=self.generate_command())

            data_button = gr.Button("Run Data")
            with gr.Accordion("Step Calculator"):
                with gr.Row():
                    self.components['data_count'] = gr.Number(
                        label="Data Count (Auto)",
                        value=0,
                        interactive=False
                    )
                    self.components['batch_size'] = gr.Number(
                        label="Micro Batch Size",
                        value=1,
                        interactive=True
                    )
                    # 不允许主动输入
                    self.components['epoch_step'] = gr.Number(
                        label="Epoch Step (Auto)",
                        value=0,
                        interactive=False
                    )
            # 数据状态
            with gr.Accordion("Data Status"):
                output_text = gr.Code(label="", lines=10, interactive=False, value="No data run yet")

            # 添加 data_path 的 change 事件
            self.components['data_path'].change(
                fn=self.update_file_list,
                inputs=[self.components['data_path'], gr.State("jsonl")],
                outputs=[self.components['data_file']]
            )
            # 添加 refresh_button 的 click 事件
            check_button.click(
                fn=self.generate_command,
                inputs=[
                    self.components['data_path'],
                    self.components['data_file'],
                    self.components['output_name'],
                    self.components['append_eod']
                ],
                outputs=[command_textbox]
            )

            # data_button 的点击事件
            data_button.click(
                fn=self.run_data,
                inputs=[
                    self.components['data_path'],
                    self.components['data_file'],
                    self.components['output_name'],
                    self.components['append_eod'],
                    self.components['batch_size']
                ],
                outputs=[
                    output_text,
                    self.components['data_count'],
                    self.components['epoch_step']
                ]
            )

        return data_interface

    # 启动数据界面
    def launch(self, *args, **kwargs):
        self.interface.launch(*args, **kwargs)
