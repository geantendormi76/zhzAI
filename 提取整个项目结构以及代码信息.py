import os
import chardet # 用于检测文件编码，需要 pip install chardet

# --- 配置 ---
PROJECT_ROOT = "/home/zhz/zhz_agent"  # <<<--- 请确保这是您项目的根目录
OUTPUT_FILE = "project_snapshot.txt"    # 输出文件的名称
MAX_FILE_SIZE_KB = 200  # 限制读取的单个文件最大体积 (KB)，防止过大的二进制文件等
INCLUDE_EXTENSIONS = ['.py', '.json', '.md', '.txt', '.yaml', '.yml', '.env', '.gitignore', '.dart', '.csv'] # 需要包含的文件类型
EXCLUDE_DIRS = ['.venv', '.git', '__pycache__', '.idea', 'build', 'dist', 
                '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 
                'node_modules', 'ios', 'android', '.dart_tool', 'linux', 'windows', 'macos', # Flutter 特有的排除项
                'build'] # Flutter build 目录
EXCLUDE_FILES = ['list_project_structure.py', 'refactor_imports_and_paths.py', os.path.basename(__file__)] # 排除脚本自身

# --- 函数定义 ---

def get_relative_path(root, full_path):
    return os.path.relpath(full_path, root)

def should_ignore(path, root, exclude_dirs, exclude_files):
    relative_path = get_relative_path(root, path)
    path_parts = relative_path.split(os.sep)

    # 检查是否是需要忽略的目录或其子目录
    for part in path_parts:
        if part in exclude_dirs:
            return True
            
    # 检查是否是需要忽略的文件
    if os.path.basename(path) in exclude_files:
        return True
        
    return False

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10240) # 读取一部分用于检测
            if not raw_data:
                return 'utf-8' # 如果文件为空，默认为utf-8
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            if encoding:
                return encoding.lower() # 转为小写以方便比较
            return 'utf-8' # 如果无法检测，默认为utf-8
    except Exception:
        return 'utf-8' # 出错也默认为utf-8


def process_project(root_dir, output_file, include_extensions, exclude_dirs, exclude_files, max_file_size_kb):
    max_file_size_bytes = max_file_size_kb * 1024
    abs_root = os.path.abspath(root_dir)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Project Root: {abs_root}\n")
        outfile.write("--- Project Structure and File Contents ---\n\n")

        for dirpath, dirnames, filenames in os.walk(abs_root, topdown=True):
            # 修改 dirnames 列表以原地排除目录
            dirnames[:] = [d for d in dirnames if not should_ignore(os.path.join(dirpath, d), abs_root, exclude_dirs, [])]
            
            if should_ignore(dirpath, abs_root, exclude_dirs, []):
                continue

            relative_dir_path = get_relative_path(abs_root, dirpath)
            if relative_dir_path == ".":
                outfile.write(f"Directory: {abs_root}/\n")
            else:
                outfile.write(f"Directory: {relative_dir_path}/\n")

            for filename in sorted(filenames):
                file_path = os.path.join(dirpath, filename)
                
                if should_ignore(file_path, abs_root, exclude_dirs, exclude_files):
                    continue

                _, ext = os.path.splitext(filename)
                if ext.lower() in include_extensions:
                    outfile.write(f"  File: {filename}\n")
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > max_file_size_bytes:
                            outfile.write(f"    --- File content skipped: Exceeds max size {max_file_size_kb}KB ---\n\n")
                            continue
                        if file_size == 0:
                             outfile.write(f"    --- File is empty ---\n\n")
                             continue

                        encoding = detect_encoding(file_path)
                        if encoding in ['gbk', 'gb2312', 'big5']: # 对于常见的中文非utf-8编码，尝试用它们打开
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                            except Exception: # 如果还是失败，尝试utf-8忽略错误
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                        elif encoding and 'utf' not in encoding and encoding != 'ascii': # 其他非UTF编码，尝试UTF-8忽略错误
                             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                        else: # 默认为utf-8或检测为utf-8/ascii
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: # errors='ignore' 以防万一
                                content = f.read()
                        
                        outfile.write("    --- START OF FILE CONTENT ---\n")
                        outfile.write(content)
                        outfile.write("\n    --- END OF FILE CONTENT ---\n\n")
                    except Exception as e:
                        outfile.write(f"    --- Error reading file {filename}: {e} ---\n\n")
                else:
                    outfile.write(f"  File (skipped, extension not included): {filename}\n")
            outfile.write("\n")
        outfile.write("--- End of Project Snapshot ---")
    print(f"Project snapshot written to {output_file}")

# --- 主程序 ---
if __name__ == "__main__":
    # 确保 PROJECT_ROOT 是您项目的实际根目录
    # 例如：/home/zhz/zhz_agent
    # 如果这个脚本就放在项目根目录，可以用 os.path.dirname(os.path.abspath(__file__))
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # PROJECT_ROOT = current_script_dir # 假设脚本在项目根目录

    if not os.path.isdir(PROJECT_ROOT):
        print(f"Error: Project root directory '{PROJECT_ROOT}' not found.")
    else:
        output_file_path = os.path.join(PROJECT_ROOT, OUTPUT_FILE) # 将输出文件也放在项目根目录
        process_project(PROJECT_ROOT, output_file_path, INCLUDE_EXTENSIONS, EXCLUDE_DIRS, EXCLUDE_FILES, MAX_FILE_SIZE_KB)