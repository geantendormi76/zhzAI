import os
import sys
import fnmatch

# --- 配置 ---
# 1. 输出文件的名称
OUTPUT_FILENAME = '现阶段核心代码.txt' # <--- 输出文件名

# 2. 需要包含的文件扩展名 (小写) 或完整文件名
TARGET_ITEMS = (
    '.py',      # Python 脚本
)

# 3. 需要排除的目录名称 (主要用于递归扫描，此处作用较小，但保留以防万一)
EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    'venv', '.venv', # 虚拟环境
    'tests', 'test',
    '.vscode',
    'node_modules',
    'dist', 'build',
}

# 4. 需要排除的文件名或模式 (使用 fnmatch，不区分大小写)
EXCLUDE_FILES = {
    '.DS_Store',
    '*.log',
    # 脚本自身和输出文件会自动排除
}
# ---

script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目路径现在就是脚本所在的目录
project_path = script_dir
output_file_path = os.path.join(script_dir, OUTPUT_FILENAME)

combined_content = []
processed_files_count = 0

print(f"正在读取 '{project_path}' 目录下的核心 Python 文件...")

# 自动排除脚本自身和输出文件
exclude_files_lower_patterns = {f.lower() for f in EXCLUDE_FILES}
exclude_files_lower_patterns.add(os.path.basename(__file__).lower())
exclude_files_lower_patterns.add(OUTPUT_FILENAME.lower())

exclude_dirs_lower = {d.lower() for d in EXCLUDE_DIRS} # 保留，但在此版本中作用不大

def should_exclude_dir(dir_name_full_path):
    # 此函数在此版本中基本不会被积极使用，因为我们不递归进入子目录
    dir_name = os.path.basename(dir_name_full_path)
    return dir_name.lower() in exclude_dirs_lower

def should_include_file(file_name_full_path):
    file_name = os.path.basename(file_name_full_path)
    file_name_lower = file_name.lower()

    # 检查是否在排除文件列表
    for pattern in exclude_files_lower_patterns:
        if fnmatch.fnmatchcase(file_name_lower, pattern):
            return False # 排除

    # 检查是否匹配目标扩展名或完整文件名
    for target in TARGET_ITEMS:
        if target.startswith('.'): # 是扩展名
            if file_name_lower.endswith(target):
                return True
        else: # 是完整文件名 (虽然我们现在只用扩展名)
            if file_name_lower == target.lower():
                return True
    return False # 不包含

def collect_files_in_current_dir(current_path_abs):
    global processed_files_count
    try:
        items = sorted(os.listdir(current_path_abs))
    except Exception as e:
        print(f"错误：无法读取目录 '{current_path_abs}' 的内容: {e}")
        return

    for item_name in items:
        item_path_abs = os.path.join(current_path_abs, item_name)
        # 对于当前目录的文件，其相对于 project_path (即 current_path_abs) 的路径就是文件名本身
        relative_path_to_project_root = item_name

        if os.path.isfile(item_path_abs): # 只处理文件
            if should_include_file(item_path_abs):
                print(f"  正在添加: {relative_path_to_project_root}")
                processed_files_count += 1
                combined_content.append(f"--- START OF FILE {relative_path_to_project_root.replace(os.sep, '/')} ---")
                try:
                    with open(item_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        combined_content.append(content)
                except Exception as e:
                    print(f"  *** 警告：读取文件 {relative_path_to_project_root} 时出错: {e} ***")
                    combined_content.append(f"*** ERROR READING FILE {relative_path_to_project_root}: {e} ***")
                combined_content.append(f"--- END OF FILE {relative_path_to_project_root.replace(os.sep, '/')} ---")
                combined_content.append("\n\n")
        # 我们不再递归进入子目录，所以移除了 os.path.isdir 的检查和递归调用

# 执行收集
collect_files_in_current_dir(project_path) # project_path 就是 script_dir

# --- 保存合并后的内容到文件 ---
final_output = "".join(combined_content)

if processed_files_count > 0:
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(final_output)
        print(f"\n成功！已将 {processed_files_count} 个 Python 文件的合并内容保存到文件: {output_file_path}")
    except Exception as e:
        print(f"\n错误：无法将内容写入文件 '{output_file_path}': {e}")
        sys.exit(1)
else:
    print(f"\n在当前目录 '{project_path}' 中未找到任何符合条件的 Python 文件。请检查 TARGET_ITEMS 设置。")