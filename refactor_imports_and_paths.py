import os
import glob

# --- 配置替换规则 ---

# 规则格式: (旧模块名或部分路径, 新模块名或部分路径, 是否为完整模块导入替换)
# 如果 is_full_module_import_replace 为 True, 则会匹配如 "from old_module import ..." 或 "import old_module"
# 如果为 False, 则进行简单的字符串替换 (用于路径常量等)

REFACTOR_RULES = [
    # --- 导入语句替换 (包名从 zhz_agent 改为 zhz_rag) ---
    # 注意：这些规则的顺序可能很重要，特别是当一个旧路径是另一个旧路径的子串时。
    # 我们先处理最具体的，再处理更通用的。

    # 1. Utils and Database
    ("from zhz_agent.utils import", "from zhz_rag.utils.common_utils import", True),
    ("import zhz_agent.utils", "import zhz_rag.utils.common_utils", True),
    ("from zhz_agent.database import", "from zhz_rag.utils.db_utils import", True),
    ("import zhz_agent.database", "import zhz_rag.utils.db_utils", True),

    # 2. Config (Constants and Pydantic Models)
    ("from zhz_agent.constants import", "from zhz_rag.config.constants import", True),
    ("import zhz_agent.constants", "import zhz_rag.config.constants", True),
    ("from zhz_agent.pydantic_models import", "from zhz_rag.config.pydantic_models import", True),
    ("import zhz_agent.pydantic_models", "import zhz_rag.config.pydantic_models", True),

    # 3. Core RAG components
    ("from zhz_agent.chromadb_retriever import", "from zhz_rag.core_rag.retrievers.chromadb_retriever import", True),
    ("from zhz_agent.file_bm25_retriever import", "from zhz_rag.core_rag.retrievers.file_bm25_retriever import", True),
    # If using __init__.py in retrievers for easier imports:
    # ("from zhz_agent.core_rag.retrievers import ChromaDBRetriever", "from zhz_rag.core_rag.retrievers import ChromaDBRetriever", True), # Example if already somewhat new
    ("from zhz_agent.kg import", "from zhz_rag.core_rag.kg_retriever import", True), # kg.py became kg_retriever.py
    ("from zhz_agent.fusion import", "from zhz_rag.core_rag.fusion_engine import", True), # fusion.py became fusion_engine.py

    # 4. LLM related
    ("from zhz_agent.llm import", "from zhz_rag.llm.llm_interface import", True), # llm.py became llm_interface.py
    ("import zhz_agent.llm", "import zhz_rag.llm.llm_interface", True),
    ("from zhz_agent.custom_llm import", "from zhz_rag.llm.custom_crewai_llms import", True), # custom_llm.py became custom_crewai_llms.py

    # 5. Evaluation related
    ("from zhz_agent.evaluation import", "from zhz_rag.evaluation.evaluator import", True), # evaluation.py became evaluator.py
    # For scripts that were moved, their own names don't need import changes, but their internal imports do.

    # 6. Finetuning related (imports within these scripts)
    # Example: refine_answer_data.py might import from zhz_rag.llm
    # This is covered by rule #4. It might also import from utils, config, etc.

    # 7. Task Management related
    ("from zhz_agent.task_jobs import", "from zhz_rag.task_management.jobs import", True), # task_jobs.py became jobs.py
    ("from zhz_agent.database_models import", "from zhz_rag.task_management.db_models import", True), # database_models.py became db_models.py

    # 8. API related (imports within these scripts)
    # Example: main_api.py might import from task_manager_api.py
    ("from zhz_agent.task_manager_service import", "from zhz_rag.api.task_manager_api import", True), # task_manager_service.py became task_manager_api.py

    # 9. CrewAI Integration related
    ("from zhz_agent.custom_crewai_tools import", "from zhz_rag.crewai_integration.tools import", True), # custom_crewai_tools.py became tools.py

    # --- Path constant replacements (simple string replace, use with caution) ---
    # These should be specific enough not to cause unintended replacements.
    # Make sure the paths are quoted as they appear in the code.
    ('"zhz_agent/rag_eval_data/"', '"zhz_rag/stored_data/evaluation_results_logs/"', False), # For analyze_*.py (eval results)
    # For batch_eval_*.py and refine_*.py (reading RAG interaction logs)
    ('LOG_FILE_DIR = "zhz_agent/rag_eval_data/"', 'LOG_FILE_DIR = "zhz_rag/stored_data/rag_interaction_logs/"', False), 
    ('RAG_LOG_DIR = "zhz_agent/rag_eval_data/"', 'RAG_LOG_DIR = "zhz_rag/stored_data/rag_interaction_logs/"', False),
    # For refine_*.py (reading evaluation logs)
    ('EVAL_LOG_DIR = "zhz_agent/rag_eval_data/"', 'EVAL_LOG_DIR = "zhz_rag/stored_data/evaluation_results_logs/"', False),
    # For refine_*.py (writing finetune data)
    ('FINETUNE_DATA_DIR = "zhz_agent/finetune_data/"', 'FINETUNE_DATA_DIR = "zhz_rag/finetuning/generated_data/"', False),
    # For common_utils.py (RAG_EVAL_DATA_DIR definition)
    # This one is tricky because RAG_EVAL_DATA_DIR was used for both interaction logs and eval results.
    # We'll assume common_utils.py's RAG_EVAL_DATA_DIR was primarily for the interaction logs or a general base.
    # Let's make it point to the new top-level stored_data.
    ("RAG_EVAL_DATA_DIR = os.path.join(_UTILS_DIR, 'rag_eval_data')", "RAG_EVAL_DATA_DIR = os.path.join(_UTILS_DIR, '..', 'stored_data')", False), # common_utils.py is in zhz_rag/utils now
]

def refactor_file_content(content, rules):
    modified_content = content
    for old, new, is_import_rule in rules:
        if is_import_rule:
            # Regex might be better for more complex import renaming, but simple replace can work for straightforward cases
            # Handle "from module import ..."
            modified_content = modified_content.replace(f"{old} ", f"{new} ")
            # Handle "import module" and "import module as alias"
            modified_content = modified_content.replace(f"import {old.split('.')[-1]}", f"import {new.split('.')[-1]}") # Simplistic, assumes old is like "zhz_agent.utils"
            modified_content = modified_content.replace(f"import {old}", f"import {new}")
        else:
            modified_content = modified_content.replace(old, new)
    return modified_content

def process_directory(directory_path, rules):
    print(f"Processing directory: {directory_path}")
    for filepath in glob.glob(os.path.join(directory_path, '**', '*.py'), recursive=True):
        if "__pycache__" in filepath:
            continue
        
        print(f"  Processing file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = refactor_file_content(original_content, rules)
            
            if modified_content != original_content:
                print(f"    Changes made in: {filepath}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            else:
                print(f"    No changes needed in: {filepath}")
        except Exception as e:
            print(f"    Error processing file {filepath}: {e}")

if __name__ == "__main__":
    # --- IMPORTANT: Backup your project before running this script! ---
    # --- Run this script from the project root directory (/home/zhz/zhz_agent) ---
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    # If the script is in project root, zhz_rag_package_path is 'zhz_rag'
    zhz_rag_package_path = os.path.join(project_root, "zhz_rag") 

    if not os.path.isdir(zhz_rag_package_path):
        print(f"Error: The main package directory '{zhz_rag_package_path}' does not exist.")
        print("Please ensure the script is in the project root and 'zhz_rag' directory is present.")
        exit()

    print("Starting refactoring process...")
    print("This script will attempt to update import statements and some path constants.")
    print("MAKE SURE YOU HAVE BACKED UP YOUR PROJECT.")
    # proceed = input("Type 'yes' to proceed: ")
    # if proceed.lower() != 'yes':
    #     print("Refactoring aborted by user.")
    #     exit()

    process_directory(zhz_rag_package_path, REFACTOR_RULES)
    
    print("\nRefactoring process completed.")
    print("Please review the changes carefully and test your project thoroughly.")
    print("You may need to manually adjust some imports if the script missed them or made incorrect changes.")
    print("Consider using a more robust refactoring tool or IDE features for complex projects.")