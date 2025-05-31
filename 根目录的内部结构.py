import os

def print_tree(dir_path, indent="", ignored_dirs=None, ignored_files=None, level=0, max_level=10, is_root_call=True):
    if ignored_dirs is None:
        ignored_dirs = {'.venv', '__pycache__', '.git', '.idea', 'build', 'dist', '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 'node_modules'}
    if ignored_files is None:
        ignored_files = {'.DS_Store'}

    base_name = os.path.basename(dir_path)

    # For non-root calls, if dir is in ignore_dirs, skip
    if not is_root_call and base_name in ignored_dirs:
        return

    if level > max_level:
        print(f"{indent}└── ... (max depth reached for {base_name})")
        return

    print(f"{indent}├── {base_name}/")
    indent += "│   "

    try:
        items = sorted(os.listdir(dir_path), key=lambda x: (not os.path.isdir(os.path.join(dir_path, x)), x.lower()))
        for item_name in items:
            item_path = os.path.join(dir_path, item_name)
            
            if os.path.isdir(item_path):
                if item_name not in ignored_dirs: # Check before recursive call
                    print_tree(item_path, indent, ignored_dirs, ignored_files, level + 1, max_level, is_root_call=False)
            else:
                if item_name not in ignored_files:
                    print(f"{indent}└── {item_name}")
    except Exception as e:
        print(f"{indent}└── Error listing contents for {base_name}: {e}")


if __name__ == "__main__":
    # --- 重要：请确保这个路径是您项目的实际根目录 ---
    # --- 例如，包含 .git, requirements.txt, 和您的主代码包 (如 zhz_rag) 的目录 ---
    project_root = "/home/zhz/zhz_agent"  #  <--- 请再次确认这个路径！

    if not os.path.isdir(project_root):
        print(f"ERROR: Specified project_root '{project_root}' does not exist or is not a directory.")
        exit()

    print(f"Analyzing Project Root: {project_root}")
    print("--- Project Structure ---")

    # Define top-level items expected directly under project_root
    # Directories that should be fully scanned if they exist
    top_level_dirs_to_scan_fully = ['zhz_rag', 'data', 'logs', 'local_agent', 'zhz_rag_pipeline_dagster']
    # Files expected directly under project_root
    top_level_files_expected = ['.env', '.envrc', 'requirements.txt', 'README.md', '.gitignore', os.path.basename(__file__), 'mcp_servers.json'] # Added mcp_servers.json here
    # Other directories to just list the name if they exist
    other_top_level_dirs_to_list = ['.venv', '.git'] # Example

    all_items_in_root = sorted(os.listdir(project_root), key=lambda x: (not os.path.isdir(os.path.join(project_root, x)), x.lower()))

    for item_name in all_items_in_root:
        item_path = os.path.join(project_root, item_name)
        is_dir = os.path.isdir(item_path)

        if item_name in top_level_dirs_to_scan_fully and is_dir:
            print_tree(item_path, indent="", level=0, max_level=7, is_root_call=True) # Start as root call
        elif item_name in top_level_files_expected and not is_dir:
            print(f"├── {item_name}")
        elif item_name in other_top_level_dirs_to_list and is_dir:
            print(f"├── {item_name}/")
        elif is_dir and item_name not in {'.venv', '__pycache__', '.git', '.idea', 'build', 'dist', '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 'node_modules'}: # Print other relevant dirs
            print(f"├── {item_name}/")
        elif not is_dir and item_name not in {'.DS_Store'}: # Print other relevant files
             if item_name not in {'.venv', '__pycache__', '.git', '.idea', 'build', 'dist', '.pytest_cache', 'htmlcov', 'zhz_rag_pipeline.egg-info', 'node_modules'}:
                 print(f"├── {item_name}")
    
    print("\n--- End of Structure ---")