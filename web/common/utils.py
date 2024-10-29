import os


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