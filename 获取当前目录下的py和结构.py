import os

# --- 配置 ---
# <<<--- 请确保这是您项目的根目录
PROJECT_ROOT = "/home/zhz/zhz_agent" 
# 输出文件的名称
OUTPUT_FILE = "project_core_structure_and_code.txt"
# 只提取这两个核心目录的结构和代码
TARGET_DIRS_TO_SCAN = ['zhz_rag_pipeline_dagster', 'zhz_rag']
# 在生成结构树和提取代码时需要排除的目录
EXCLUDE_DIRS = [
    '.venv', '.git', '__pycache__', '.idea', 'build', 'dist', 
    '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 
    'node_modules', '.dart_tool', 'ios', 'android', 'linux', 'windows', 'macos'
]
# 在生成结构树和提取代码时需要排除的文件（包括脚本自身）
EXCLUDE_FILES = [
    'list_project_structure.py', 
    'refactor_imports_and_paths.py', 
    os.path.basename(__file__) # 排除当前脚本文件
] 
# 需要提取代码内容的文件后缀
CODE_FILE_EXTENSIONS = ['.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md']

# --- 函数定义 ---

def write_specific_directory_tree_and_code(outfile, root_dir, target_dirs, exclude_dirs, exclude_files, code_extensions):
    """
    为指定的一组目录生成结构树，并写入到文件中。
    同时，对于特定类型的文件，将其代码内容也写入文件。
    """
    abs_root = os.path.abspath(root_dir)
    outfile.write(f"# Core Directories Structure and Code for Project: {abs_root}\n\n")

    for target_dir_name in target_dirs:
        target_path = os.path.join(abs_root, target_dir_name)

        if not os.path.isdir(target_path):
            outfile.write(f"--- Directory '{target_dir_name}' not found. Skipping. ---\n\n")
            continue
        
        outfile.write(f"--- Structure and Code for: {target_dir_name}/ ---\n")
        
        for dirpath, dirnames, filenames in os.walk(target_path, topdown=True):
            # 1. 过滤掉需要排除的目录
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            
            # 2. 过滤掉需要排除的文件
            filenames = [f for f in filenames if f not in exclude_files]

            # 3. 计算当前深度并打印目录结构
            relative_dir_path = os.path.relpath(dirpath, target_path)
            
            if relative_dir_path == ".":
                level = 0
            else:
                # 确保路径分隔符在计算层级时被正确处理
                level = len(relative_dir_path.split(os.sep))
            
            indent = '    ' * level + '|-- '
            
            # 打印子目录
            for dirname in sorted(dirnames):
                outfile.write(f"{indent}{dirname}/\n")
            
            # 打印文件并提取代码内容
            for filename in sorted(filenames):
                outfile.write(f"{indent}{filename}\n")
                
                # 检查文件后缀是否在需要提取代码的列表中
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in code_extensions:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code_content = f.read()
                        outfile.write(f"\n``` {file_ext[1:]}\n") # 写入代码块的起始标记，去掉点
                        outfile.write(code_content)
                        outfile.write(f"\n```\n\n") # 写入代码块的结束标记
                    except Exception as e:
                        outfile.write(f"\n# Error reading file {filename}: {e}\n\n")
        
        outfile.write("\n" + "="*50 + "\n\n")

# --- 主程序 ---
if __name__ == "__main__":
    # 确保项目根目录存在
    if not os.path.isdir(PROJECT_ROOT):
        print(f"Error: Project root directory '{PROJECT_ROOT}' not found.")
    else:
        output_file_path = os.path.join(PROJECT_ROOT, OUTPUT_FILE)
        try:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                write_specific_directory_tree_and_code(
                    outfile,
                    PROJECT_ROOT,
                    TARGET_DIRS_TO_SCAN,
                    EXCLUDE_DIRS,
                    EXCLUDE_FILES,
                    CODE_FILE_EXTENSIONS
                )
            print(f"Core directories structure and code successfully written to {output_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

