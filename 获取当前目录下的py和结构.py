import os
import chardet # 用于检测文件编码，需要 pip install chardet

# --- 配置 ---
# <<<--- 请确保这是您项目的根目录
PROJECT_ROOT = "/home/zhz/zhz_agent" 
# 输出文件的名称
OUTPUT_FILE = "project_core_snapshot.txt"
# 只扫描并提取这两个核心目录
TARGET_DIRS_TO_SCAN = ['zhz_rag_pipeline_dagster', 'zhz_rag']
# 只处理 .py 文件
INCLUDE_EXTENSIONS = ['.py']
# 限制读取的单个文件最大体积 (KB)
MAX_FILE_SIZE_KB = 200
# 在扫描时需要排除的目录
EXCLUDE_DIRS = [
    '.venv', '.git', '__pycache__', '.idea', 'build', 'dist', 
    '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 
    'node_modules', '.dart_tool', 'ios', 'android', 'linux', 'windows', 'macos'
]
# 需要排除的文件（包括脚本自身）
EXCLUDE_FILES = [
    'list_project_structure.py', 
    'refactor_imports_and_paths.py', 
    os.path.basename(__file__)
] 

# --- 函数定义 ---

def write_directory_tree(outfile, scan_path, root_for_rel_path, exclude_dirs, exclude_files):
    """为单个目录生成结构树"""
    outfile.write(f"--- Structure for: {os.path.relpath(scan_path, root_for_rel_path)}/ ---\n")
    for dirpath, dirnames, filenames in os.walk(scan_path, topdown=True):
        # 过滤掉需要排除的目录和文件
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        filenames = [f for f in filenames if f not in exclude_files]

        relative_path = os.path.relpath(dirpath, scan_path)
        if relative_path == ".":
            level = 0
        else:
            level = len(relative_path.split(os.sep))
        
        indent = '    ' * level + '|-- '
        
        # 打印当前目录下的子目录和文件
        # os.walk的后续迭代会处理这些子目录的内容
        if level == 0:
             # 对于根目录，直接列出其下的内容
            for d in sorted(dirnames):
                outfile.write(f"|-- {d}/\n")
            for f in sorted(filenames):
                outfile.write(f"|-- {f}\n")
        else:
            # 对于子目录，显示其相对路径
            current_dir_name = os.path.basename(dirpath)
            parent_path = os.path.dirname(dirpath)
            parent_relative = os.path.relpath(parent_path, scan_path)
            parent_level = 0 if parent_relative == "." else len(parent_relative.split(os.sep))
            
            tree_indent = '    ' * parent_level + '|-- '
            outfile.write(f"{tree_indent}{current_dir_name}/\n")
            
            sub_indent = '    ' * (parent_level + 1) + '|-- '
            for f in sorted(filenames):
                 outfile.write(f"{sub_indent}{f}\n")


def detect_encoding(file_path):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10240)
            if not raw_data: return 'utf-8'
            result = chardet.detect(raw_data)
            return result['encoding'].lower() if result['encoding'] else 'utf-8'
    except (FileNotFoundError, Exception):
        return 'utf-8'

def write_code_contents(outfile, scan_path, root_for_rel_path, include_exts, max_size_kb):
    """写入指定目录中所有py文件的内容"""
    max_size_bytes = max_size_kb * 1024
    outfile.write(f"\n--- Code Contents for: {os.path.relpath(scan_path, root_for_rel_path)}/ ---\n\n")

    for dirpath, _, filenames in os.walk(scan_path):
        for filename in sorted(filenames):
            _, ext = os.path.splitext(filename)
            if ext.lower() not in include_exts:
                continue

            file_path = os.path.join(dirpath, filename)
            relative_file_path = os.path.relpath(file_path, root_for_rel_path)
            
            outfile.write(f"File: {relative_file_path}\n")
            outfile.write("-" * len(f"File: {relative_file_path}") + "\n")
            
            try:
                file_size = os.path.getsize(file_path)
                if file_size > max_size_bytes:
                    outfile.write(f"--- Content skipped: File exceeds max size {max_size_kb}KB ---\n\n")
                    continue
                if file_size == 0:
                    outfile.write(f"--- File is empty ---\n\n")
                    continue

                encoding = detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                outfile.write("```python\n")
                outfile.write(content.strip())
                outfile.write("\n```\n\n")
            except Exception as e:
                outfile.write(f"--- Error reading file {filename}: {e} ---\n\n")

# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.isdir(PROJECT_ROOT):
        print(f"Error: Project root directory '{PROJECT_ROOT}' not found.")
    else:
        output_file_path = os.path.join(PROJECT_ROOT, OUTPUT_FILE)
        try:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                abs_root = os.path.abspath(PROJECT_ROOT)
                outfile.write(f"# Core Snapshot for Project: {abs_root}\n\n")

                for target_dir_name in TARGET_DIRS_TO_SCAN:
                    target_path = os.path.join(abs_root, target_dir_name)
                    if not os.path.isdir(target_path):
                        outfile.write(f"--- Directory '{target_dir_name}' not found. Skipping. ---\n\n")
                        continue
                    
                    # 1. 写入该目录的结构树
                    write_directory_tree(outfile, target_path, abs_root, EXCLUDE_DIRS, EXCLUDE_FILES)
                    
                    # 2. 写入该目录的代码内容
                    write_code_contents(outfile, target_path, abs_root, INCLUDE_EXTENSIONS, MAX_FILE_SIZE_KB)
                    
                    outfile.write("\n" + "="*80 + "\n\n")

            print(f"Core project snapshot successfully written to {output_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
