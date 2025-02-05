import os
import torch
import yaml
import json
def get_project_root(file_name):
    current_path = os.path.abspath(__file__)
    while True:
        parent_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(parent_path, file_name)):
            return parent_path
        if parent_path == current_path:
            raise Exception("Could not find project root directory")
        current_path = parent_path

def get_files(directory, file_type):
    if os.path.exists(directory):
        # 只返回文件名，不返回路径
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith("." + file_type)]
    return []

def read_cache(cache_file):
    try:
        with open(cache_file, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}

def write_cache(data, cache_file, module, is_public=False):
    cache = read_cache(cache_file)
    if is_public:
        cache['public'] = data
    else:
        if module not in cache:
            cache[module] = {}
        cache[module].update(data)
    with open(cache_file, 'w') as file:
        yaml.safe_dump(cache, file)
        
def calculate_epoch_steps(data_count, batch_size):
    if data_count and batch_size and batch_size > 0:
        return data_count // batch_size
    return 0

def check_gpu_status():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_status = f"🟢 GPU is available with {gpu_count} GPU(s)."
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转换为 GB
            gpu_status += f"\n - GPU {i+1}: {gpu_name}, Memory: {gpu_memory:.2f} GB"
    else:
        gpu_status = "🔴 GPU is not available."

    return gpu_status

# 读取训练数据
def read_training_data(output_path):
    loss_file = os.path.join(output_path, "loss_data.json")
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

def clean_training_data(output_path):
    loss_file = os.path.join(output_path, "loss_data.json")
    if os.path.exists(loss_file):
        os.remove(loss_file)
        
# 记录错误标识，写入文件
def set_error_message(msg):
    with open("error_message.txt", "w") as f:
        f.write(msg)
        
# 读取错误标识
def get_error_message():
    if os.path.exists("error_message.txt"):
        with open("error_message.txt", "r") as f:
            return f.read()
    return ""

# 删除错误标识
def delete_error():
    if os.path.exists("error_message.txt"):
        os.remove("error_message.txt")