import os
from pathlib import Path
from typing import List, Set, Optional

def generate_tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
    dir_include: Optional[List[str]] = None,
    dir_exclude: Optional[List[str]] = None,
    file_exclude: Optional[List[str]] = None,
    prefix: str = "",
    output_lines: Optional[List[str]] = None,
) -> List[str]:
    """
    Generates a directory tree structure.

    Args:
        dir_path (Path): The directory to start from.
        level (int): Maximum depth to traverse. -1 for no limit.
        limit_to_directories (bool): If True, only list directories.
        length_limit (int): Maximum number of lines to output.
        dir_include (Optional[List[str]]): List of directory names to explicitly include.
                                         If None, all are included (respecting exclude).
        dir_exclude (Optional[List[str]]): List of directory names to exclude.
        file_exclude (Optional[List[str]]): List of file names/extensions to exclude.
        prefix (str): Prefix for the current line (used for recursion).
        output_lines (Optional[List[str]]): List to append output lines to.

    Returns:
        List[str]: A list of strings representing the directory tree.
    """
    if output_lines is None:
        output_lines = []

    if dir_exclude is None:
        dir_exclude = []
    if file_exclude is None:
        file_exclude = []

    # Default excludes (can be customized)
    default_dir_excludes = {".venv-uv-rag",".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "build", "dist", "*.egg-info"}
    default_file_excludes = {".DS_Store"}

    current_dir_excludes = set(dir_exclude) | default_dir_excludes
    current_file_excludes = set(file_exclude) | default_file_excludes

    if not dir_path.is_dir():
        output_lines.append(f"Error: {dir_path} is not a directory or does not exist.")
        return output_lines

    if len(output_lines) >= length_limit:
        return output_lines

    # Sort entries for consistent output, directories first
    entries = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    
    pointers = ["├── "] * (len(entries) - 1) + ["└── "]

    for pointer, path in zip(pointers, entries):
        if len(output_lines) >= length_limit:
            break

        if path.is_dir():
            # Directory include/exclude logic
            if dir_include and path.name not in dir_include:
                continue
            if path.name in current_dir_excludes or \
               any(path.match(pattern) for pattern in current_dir_excludes if "*" in pattern): # Handle wildcard patterns
                continue

            output_lines.append(f"{prefix}{pointer}{path.name}/")
            if level == -1 or level > 0:
                extension = "    " if pointer == "└── " else "│   "
                generate_tree(
                    path,
                    level=(level - 1) if level > 0 else -1, # Decrement level if it's not -1
                    limit_to_directories=limit_to_directories,
                    length_limit=length_limit,
                    dir_include=dir_include,
                    dir_exclude=dir_exclude,
                    file_exclude=file_exclude,
                    prefix=(prefix + extension),
                    output_lines=output_lines,
                )
        elif not limit_to_directories:
            # File exclude logic
            if path.name in current_file_excludes or \
               any(path.match(f"*{ext}") for ext in current_file_excludes if ext.startswith(".")) or \
               any(path.match(pattern) for pattern in current_file_excludes if "*" in pattern):
                continue
            output_lines.append(f"{prefix}{pointer}{path.name}")

    return output_lines


if __name__ == "__main__":
    # 直接获取脚本文件所在的目录作为项目根目录
    project_root = Path(__file__).resolve().parent

    # --- 自定义排除项 ---
    # 您可以在这里添加更多需要排除的目录或文件/扩展名
    custom_dir_excludes = [
        "logs", 
        "data", # 通常我们不需要展示数据文件的具体内容，除非有特殊需求
        "stored_data", # 包含数据库和索引文件，通常很大且不直观
        "duckdb_extensions", # 扩展二进制文件
        ".pytest_cache",
        "*.egg-info", # 构建产物
        "poc_document_parsing/docx_tests/sample_docs", # 示例文件目录
        "poc_document_parsing/pdf_tests/sample_docs",
        "poc_document_parsing/xlsx_tests/sample_docs",
        "poc_document_parsing/md_tests/sample_docs",
        "poc_document_parsing/html_tests/sample_docs",
        "local_agent" # 如果 local_agent 是独立的，可能不需要包含
    ]
    custom_file_excludes = [
        "*.pyc", "*.log", "*.tmp", "*.swp", "*.bak",
        ".env", ".envrc", ".gitignore", # 通常这些配置文件不需要在结构树中展示给AI
        "project_snapshot.txt", # 避免包含脚本自身生成的快照
        "generate_project_tree.py", # 排除脚本自身
        "*.db", "*.db.wal", # 排除数据库文件
        "*.jsonl", # 排除jsonl日志文件
        "*.csv" # 排除csv文件
    ]
    
    # --- 仅包含特定顶层目录 ---
    # 明确指定需要包含的顶层目录
    dir_to_include_at_root = ["zhz_rag", "zhz_rag_pipeline_dagster"]
    top_level_dirs = [d for d in project_root.iterdir() if d.is_dir() and d.name in dir_to_include_at_root]
    
    all_output_lines = [f"项目结构树: {project_root.resolve()}"]

    for top_dir_path in top_level_dirs:
        # 仅在是指定子目录时，显示其名称作为分隔符
        # 这里不需要判断 top_dir_path != project_root，因为 top_level_dirs 已经过滤了
        all_output_lines.append(f"\n--- Directory: {top_dir_path.name}/ ---") # 增加更明显的分隔符
        
        tree_lines = generate_tree(
            top_dir_path,
            dir_exclude=custom_dir_excludes,
            file_exclude=custom_file_excludes,
            # level=3 # 如果想限制深度，可以设置level
        )
        all_output_lines.extend(tree_lines)
        all_output_lines.append("-" * 40) # 每个目录树结束后添加分隔线


    output_file_name = "project_structure_tree.txt"
    output_file_path = project_root / output_file_name

    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in all_output_lines:
            f.write(line + "\n")
    
    print(f"\n项目结构树已生成到: {output_file_path.resolve()}")
    print(f"总行数: {len(all_output_lines)}")

