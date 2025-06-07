import os
import shutil # 用于备份

# --- 配置 ---
ROOT_DIR = "/home/zhz/zhz_agent"  # 您项目的根目录
OLD_STRING = "sglang_wrapper"
NEW_STRING = "llm_interface"

# 需要处理的文件扩展名
TARGET_EXTENSIONS = ('.py', '.md', '.txt', '.json', '.yaml', '.yml') # 根据需要添加或修改

# 需要排除的目录名称 (精确匹配，小写)
EXCLUDE_DIRS = {
    '.git',
    '.venv',
    '__pycache__',
    'logs', # 假设您有日志目录不想扫描
    'stored_data', # 通常这里面是数据文件，不是代码引用
    'zhz_rag_pipeline_dagster_project.egg-info', # 构建产物
    'zhz_rag_core.egg-info', # 构建产物
    # 根据您的项目结构，可能还需要添加其他如 data, models, .vscode, node_modules 等
}

# 需要排除的文件名 (精确匹配，小写)
EXCLUDE_FILES = {
    'project_snapshot.txt', # 避免修改快照文件
    # 脚本自身会自动排除 (如果它在扫描路径下)
}

# --- 备份配置 ---
BACKUP_SUFFIX = ".bak_before_rename" # 备份文件的后缀

def should_process_file(filepath, filename_lower):
    """判断是否应该处理该文件"""
    if not filename_lower.endswith(TARGET_EXTENSIONS):
        return False
    if filename_lower in EXCLUDE_FILES:
        return False
    return True

def process_file_content(filepath):
    """读取文件内容，进行替换，如果发生改变则写回"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  - Error reading file {filepath}: {e}")
        return False

    if OLD_STRING in content:
        print(f"  - Found '{OLD_STRING}' in: {filepath}")
        
        # 创建备份
        backup_filepath = filepath + BACKUP_SUFFIX
        try:
            shutil.copy2(filepath, backup_filepath)
            print(f"    - Backup created: {backup_filepath}")
        except Exception as e_backup:
            print(f"    - ERROR creating backup for {filepath}: {e_backup}. Skipping modification.")
            return False

        new_content = content.replace(OLD_STRING, NEW_STRING)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"    - Replaced in: {filepath}")
            return True
        except Exception as e_write:
            print(f"    - Error writing to file {filepath}: {e_write}")
            # 如果写回失败，可以考虑是否恢复备份，但通常保留备份让用户手动处理更好
            return False
    return False

def rename_files_and_dirs(root_path):
    """重命名包含 OLD_STRING 的文件和目录"""
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False): # topdown=False 确保先处理子目录
        # 排除特定目录
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIRS]

        # 重命名文件
        for filename in filenames:
            if OLD_STRING in filename:
                old_filepath = os.path.join(dirpath, filename)
                new_filename = filename.replace(OLD_STRING, NEW_STRING)
                new_filepath = os.path.join(dirpath, new_filename)
                if os.path.exists(new_filepath):
                    print(f"  - SKIPPING rename (file): Target '{new_filepath}' already exists.")
                    continue
                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"  - Renamed file: '{old_filepath}' to '{new_filepath}'")
                except Exception as e:
                    print(f"  - ERROR renaming file '{old_filepath}': {e}")
        
        # 重命名目录 (在处理完文件名之后)
        # 注意：os.walk 的 dirnames 是副本，直接修改它不会影响遍历
        # 我们需要在下一次 walk 时，新的目录名才会生效，或者在一次遍历后重新执行这部分
        # 为了简单，这里只重命名当前级别的目录，如果深层嵌套目录名也包含OLD_STRING，可能需要多次运行或更复杂的逻辑
        # 但通常我们只关心顶层的那个 llm 目录下的 sglang_wrapper.py
        # 如果您有 sglang_wrapper_utils 这样的目录名也想改，这个逻辑能处理
        for dirname in dirnames: # dirnames 已经是被 EXCLUDE_DIRS 过滤后的
            if OLD_STRING in dirname:
                old_dirpath = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(OLD_STRING, NEW_STRING)
                new_dirpath = os.path.join(dirpath, new_dirname)
                if os.path.exists(new_dirpath):
                     print(f"  - SKIPPING rename (dir): Target '{new_dirpath}' already exists.")
                     continue
                try:
                    # shutil.move 更安全，可以处理跨文件系统的情况，但 os.rename 对于同文件系统内的重命名通常足够
                    os.rename(old_dirpath, new_dirpath)
                    print(f"  - Renamed directory: '{old_dirpath}' to '{new_dirpath}'")
                except Exception as e:
                    print(f"  - ERROR renaming directory '{old_dirpath}': {e}")


def main():
    print(f"Starting search and replace in directory: {ROOT_DIR}")
    print(f"Replacing all occurrences of '{OLD_STRING}' with '{NEW_STRING}'")
    print(f"Target file extensions: {TARGET_EXTENSIONS}")
    print(f"Excluded directories: {EXCLUDE_DIRS}")
    print(f"Excluded files: {EXCLUDE_FILES}")
    print("-" * 30)

    # 第一步：先重命名文件和目录
    # 特别是我们知道要将 zhz_rag/llm/sglang_wrapper.py 重命名为 zhz_rag/llm/llm_interface.py
    # 这个脚本的 rename_files_and_dirs 会处理所有匹配的文件名和目录名
    print("\n--- Phase 1: Renaming files and directories ---")
    # 为了确保我们只重命名目标文件，可以先手动执行这个，或者更精确地指定路径
    # 这里我们先尝试全局重命名（除了排除项）
    # rename_files_and_dirs(ROOT_DIR) # 如果您只想重命名特定的那个文件，建议手动操作

    # 手动重命名我们最关心的那个文件
    specific_old_file = os.path.join(ROOT_DIR, "zhz_rag", "llm", f"{OLD_STRING}.py")
    specific_new_file = os.path.join(ROOT_DIR, "zhz_rag", "llm", f"{NEW_STRING}.py")
    if os.path.exists(specific_old_file):
        if os.path.exists(specific_new_file):
            print(f"  - SKIPPING rename: Target '{specific_new_file}' already exists.")
        else:
            try:
                os.rename(specific_old_file, specific_new_file)
                print(f"  - Successfully renamed: '{specific_old_file}' to '{specific_new_file}'")
            except Exception as e:
                print(f"  - ERROR renaming '{specific_old_file}': {e}")
    else:
        print(f"  - INFO: Specific file '{specific_old_file}' not found, assuming already renamed or not present.")


    print("\n--- Phase 2: Replacing content in files ---")
    files_changed_count = 0
    files_scanned_count = 0

    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # 排除特定目录
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_DIRS and not d.endswith(BACKUP_SUFFIX)]
        
        for filename in filenames:
            if filename.endswith(BACKUP_SUFFIX): # 跳过备份文件
                continue

            filepath = os.path.join(dirpath, filename)
            filename_lower = filename.lower()
            
            # 排除脚本自身
            if os.path.abspath(filepath) == os.path.abspath(__file__):
                continue

            if should_process_file(filepath, filename_lower):
                files_scanned_count += 1
                if process_file_content(filepath):
                    files_changed_count += 1
            
    print("-" * 30)
    print(f"Scan complete. {files_scanned_count} files scanned.")
    print(f"{files_changed_count} files were modified (backups created with {BACKUP_SUFFIX} suffix).")
    if files_changed_count > 0:
        print("IMPORTANT: Please review the changes carefully, for example using 'git diff'.")
        print("You can restore individual files from their .bak_before_rename backups if needed.")

if __name__ == "__main__":
    # 再次确认
    confirm = input(f"This script will modify files in '{ROOT_DIR}'.\n"
                    f"It will rename '{OLD_STRING}.py' to '{NEW_STRING}.py' in 'zhz_rag/llm/' if it exists,\n"
                    f"and then replace all occurrences of '{OLD_STRING}' with '{NEW_STRING}' in file contents.\n"
                    f"Backups of modified files will be created with '{BACKUP_SUFFIX}' suffix.\n"
                    "Are you sure you want to continue? (yes/no): ")
    if confirm.lower() == 'yes':
        main()
    else:
        print("Operation cancelled by user.")
