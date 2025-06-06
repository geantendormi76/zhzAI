# local_agent_app.py
import sys
import os
import logging
import traceback
from typing import Dict, Any, List, Optional, Union

# --- Path and DLL setup ---
print("--- sys.path at the very beginning of local_agent_app.py ---")
for p in sys.path:
    print(p)
print(f"--- os.getcwd() at the very beginning: {os.getcwd()} ---")
print("-----------------------------------------------------------")

venv_path = os.path.dirname(sys.executable)
scripts_path = os.path.join(venv_path)
dlls_path = os.path.join(venv_path, "..", "DLLs")

if os.path.exists(scripts_path) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(scripts_path)
        print(f"INFO: Added to DLL search path: {scripts_path}")
    except Exception as e_dll_add:
        print(f"WARN: Could not add {scripts_path} to DLL search path: {e_dll_add}")

if os.path.exists(dlls_path) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(dlls_path)
        print(f"INFO: Added to DLL search path: {dlls_path}")
    except Exception as e_dll_add:
        print(f"WARN: Could not add {dlls_path} to DLL search path: {e_dll_add}")

# --- Library Imports and Global Status ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from winotify import Notification
import xml.etree.ElementTree

# Global variables to store import status and library objects
_PANDAS_IMPORTED = False
_PANDASQL_IMPORTED = False
pd = None
PandaSQL = None

try:
    import pandas as pd_temp
    pd = pd_temp
    logging.info(f"Successfully imported pandas. pd is: {pd}")
    logging.info(f"pandas version: {pd.__version__}")
    _PANDAS_IMPORTED = True
except ImportError as e_pandas_global:
    logging.critical(f"CRITICAL_IMPORT_ERROR: Failed to import pandas at global scope: {e_pandas_global}", exc_info=True)
except Exception as e_pandas_other_global:
    logging.critical(f"CRITICAL_OTHER_ERROR: Error during pandas import or version check: {e_pandas_other_global}", exc_info=True)

try:
    from pandasql import PandaSQL as PandaSQL_temp
    PandaSQL = PandaSQL_temp
    logging.info(f"Successfully imported PandaSQL. PandaSQL is: {PandaSQL}")
    _PANDASQL_IMPORTED = True
except ImportError as e_pandasql_global:
    logging.warning(f"IMPORT_WARNING: Failed to import pandasql at global scope: {e_pandasql_global}")
except Exception as e_pandasql_other_global:
    logging.warning(f"OTHER_IMPORT_WARNING: Error during pandasql import: {e_pandasql_other_global}")

# --- Configuration ---
LOCAL_AGENT_PORT = 8003
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - LocalAgent - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ZhzAI Local Agent",
    description="本地代理程序，提供处理本地文件和执行特定任务的MCP服务。",
    version="0.1.1"
)

# --- Pydantic Model Definitions ---
class ExecuteSQORequest(BaseModel):
    sqo: Dict[str, Any] = Field(description="必需。一个结构化查询对象 (SQO) 的JSON字典...")

class SQOResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_details: Optional[str] = None

class NotificationRequest(BaseModel):
    title: str = Field(default="任务提醒", description="通知的标题")
    message: str = Field(description="通知的主要内容")
    app_name: str = Field(default="终端大脑助手", description="显示在通知来源的应用名称")

class NotificationResponse(BaseModel):
    success: bool
    message: str

# --- Filter Helper Function ---
def apply_filters_to_dataframe(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    df_filtered = df.copy()
    for f_obj in filters:
        column = f_obj.get("column")
        operator = f_obj.get("operator")
        value = f_obj.get("value")
        is_date = f_obj.get("is_date", False)

        if not all([column, operator, value is not None]):
            logger.warning(f"Skipping invalid filter object: {f_obj}")
            continue

        if column not in df_filtered.columns:
            logger.warning(f"Filter column '{column}' not found in DataFrame. Available: {df_filtered.columns.tolist()}. Skipping filter: {f_obj}")
            continue

        try:
            series_to_filter = df_filtered[column]
            if is_date:
                try:
                    series_to_filter = pd.to_datetime(series_to_filter, errors='coerce')
                    filter_value_dt = pd.to_datetime(value, errors='coerce')
                    if pd.isna(filter_value_dt) and not (isinstance(value, list) and operator in ["in", "is_in_list", "not in", "is_not_in_list"]):
                        logger.warning(f"Cannot convert filter value '{value}' to date for column '{column}'. Skipping filter.")
                        continue
                    value = filter_value_dt
                except Exception as e_date:
                    logger.warning(f"Error converting column '{column}' or value '{value}' to datetime: {e_date}. Skipping filter.")
                    continue

            condition = None
            if operator in ["equals", "=="]: condition = (series_to_filter == value)
            elif operator in ["not_equals", "!="]: condition = (series_to_filter != value)
            elif operator in ["greater_than", ">"]: condition = (series_to_filter > value)
            elif operator in ["greater_than_or_equals", ">="]: condition = (series_to_filter >= value)
            elif operator in ["less_than", "<"]: condition = (series_to_filter < value)
            elif operator in ["less_than_or_equals", "<="]: condition = (series_to_filter <= value)
            elif operator == "contains":
                if is_date: logger.warning(f"'contains' not applicable to date column '{column}'. Skip."); continue
                condition = (series_to_filter.astype(str).str.contains(str(value), case=False, na=False))
            elif operator == "not_contains":
                if is_date: logger.warning(f"'not_contains' not applicable to date column '{column}'. Skip."); continue
                condition = (~series_to_filter.astype(str).str.contains(str(value), case=False, na=False))
            elif operator in ["is_in_list", "in"]:
                if not isinstance(value, list): logger.warning(f"'is_in_list' expects list value for '{column}'. Skip."); continue
                if is_date:
                    list_value_dt = pd.to_datetime(value, errors='coerce').dropna().tolist()
                    if not list_value_dt: logger.warning(f"Cannot convert list values to dates for 'is_in_list' on '{column}'. Skip."); continue
                    condition = (series_to_filter.isin(list_value_dt))
                else: condition = (series_to_filter.isin(value))
            elif operator in ["is_not_in_list", "not in"]:
                if not isinstance(value, list): logger.warning(f"'is_not_in_list' expects list value for '{column}'. Skip."); continue
                if is_date:
                    list_value_dt = pd.to_datetime(value, errors='coerce').dropna().tolist()
                    if not list_value_dt: logger.warning(f"Cannot convert list values for 'is_not_in_list' on '{column}'. Skip."); continue
                    condition = (~series_to_filter.isin(list_value_dt))
                else: condition = (~series_to_filter.isin(value))
            elif operator == "between":
                if not (isinstance(value, list) and len(value) == 2): logger.warning(f"'between' expects list of two values for '{column}'. Skip."); continue
                val1, val2 = value
                if is_date:
                    val1_dt, val2_dt = pd.to_datetime(val1, errors='coerce'), pd.to_datetime(val2, errors='coerce')
                    if pd.isna(val1_dt) or pd.isna(val2_dt): logger.warning(f"Cannot convert 'between' values to dates for '{column}'. Skip."); continue
                    condition = (series_to_filter.between(min(val1_dt, val2_dt), max(val1_dt, val2_dt)))
                else: condition = (series_to_filter.between(min(val1, val2), max(val1, val2)))
            elif operator in ["is_null", "isnull"]: condition = (series_to_filter.isnull())
            elif operator in ["is_not_null", "notnull"]: condition = (series_to_filter.notnull())
            else: logger.warning(f"Unsupported filter operator '{operator}' for column '{column}'. Skipping."); continue

            if condition is not None: df_filtered = df_filtered[condition]
            else: logger.warning(f"Condition was None for filter {f_obj}.")
        except Exception as e_filter:
            logger.error(f"Error applying filter {f_obj} on column '{column}': {e_filter}", exc_info=True)
            continue
    return df_filtered

# --- API Endpoints ---
@app.post("/notify", response_model=NotificationResponse)
async def send_desktop_notification(req: NotificationRequest):
    logger.info(f"Received notification request: Title='{req.title}', Message='{req.message}'")
    try:
        toast = Notification(app_id=req.app_name, title=req.title, msg=req.message)
        toast.show()
        logger.info(f"Desktop notification successfully shown: '{req.title}'")
        return NotificationResponse(success=True, message="Notification successfully shown.")
    except Exception as e:
        logger.error(f"Failed to show desktop notification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to show notification: {str(e)}")

@app.post("/excel_sqo_mcp/execute_operation", response_model=SQOResponse)
async def execute_excel_sqo_operation(request_data: ExecuteSQORequest):
    # Check global import status at the beginning
    if not _PANDAS_IMPORTED:
        logger.critical("Pandas (pd) was not imported successfully at global scope.")
        return SQOResponse(success=False, error="内部服务器错误：Pandas库未能加载。", error_details="Pandas (pd) is None or import failed at startup.")

    sqo = request_data.sqo
    operation_type = sqo.get("operation_type")
    file_path_from_sqo = sqo.get("file_path")
    sheet_name = sqo.get("sheet_name", 0)

    # Check for PandaSQL only when the specific operation requires it
    if operation_type == "direct_sql_query" and not _PANDASQL_IMPORTED:
        logger.critical("PandaSQL was not imported successfully at global scope and is required for 'direct_sql_query' operation.")
        return SQOResponse(success=False, error="内部服务器错误：PandaSQL库未能加载，无法执行direct_sql_query。", error_details="PandaSQL import failed at startup.")

    logger.info(f"Received SQO request: operation='{operation_type}', original_file_path='{file_path_from_sqo}', sheet='{sheet_name}'")
    logger.debug(f"Full SQO received: {sqo}")

    if not file_path_from_sqo or not operation_type:
        logger.error("SQO missing 'file_path' or 'operation_type'.")
        return SQOResponse(success=False, error="SQO中缺少 'file_path' 或 'operation_type' 参数。")

    effective_file_path = file_path_from_sqo
    if not os.path.isabs(effective_file_path):
        logger.warning(f"Received file path '{effective_file_path}' is not absolute.")
    
    if not os.path.exists(effective_file_path):
        path_alt_forward_slash = file_path_from_sqo.replace("\\", "/")
        if os.path.exists(path_alt_forward_slash):
            effective_file_path = path_alt_forward_slash
        else:
            logger.error(f"Excel file not found at path: '{file_path_from_sqo}' (and alternative '{path_alt_forward_slash}' also not found).")
            return SQOResponse(success=False, error=f"Excel文件在路径 '{file_path_from_sqo}' 未找到。")

    logger.info(f"Effective file path for pandas: '{effective_file_path}'")

    df = None  # Initialize df
    try:
        df = pd.read_excel(effective_file_path, sheet_name=sheet_name)
        logger.info(f"Successfully loaded DataFrame from '{effective_file_path}'. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        result_data = None

        # --- Operation Type Handling ---
        if operation_type == "get_unique_values":
            column_name = sqo.get("column_name")
            if not column_name or column_name not in df.columns:
                error_msg = f"'get_unique_values' 操作缺少有效 'column_name' 或列 '{column_name}' 不存在。可用列: {df.columns.tolist()}"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            df_to_process = df.copy()
            filters_from_sqo = sqo.get("filters")
            if filters_from_sqo and isinstance(filters_from_sqo, list):
                df_to_process = apply_filters_to_dataframe(df_to_process, filters_from_sqo)
            unique_values_series = df_to_process[column_name].unique()
            if sqo.get('options', {}).get('drop_na', False):
                unique_values = pd.Series(unique_values_series).dropna().tolist()
            else:
                unique_values = unique_values_series.tolist()
            result_data = unique_values

        elif operation_type == "group_by_aggregate":
            group_by_cols = sqo.get("group_by_columns")
            agg_col = sqo.get("aggregation_column")
            agg_func = sqo.get("aggregation_function")
            if not (group_by_cols and agg_col and agg_func):
                error_msg = "'group_by_aggregate' 操作缺少 'group_by_columns', 'aggregation_column', 或 'aggregation_function' 参数。"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            columns_to_check = []
            if isinstance(group_by_cols, list): columns_to_check.extend(group_by_cols)
            elif isinstance(group_by_cols, str): columns_to_check.append(group_by_cols)
            if agg_col: columns_to_check.append(agg_col)
            for col_check in columns_to_check:
                if col_check not in df.columns:
                    error_msg = f"列 '{col_check}' 在Excel中未找到。可用列: {df.columns.tolist()}"
                    logger.error(error_msg)
                    return SQOResponse(success=False, error=error_msg)
            df_to_group = df.copy()
            filters_from_sqo_group = sqo.get("filters")
            if filters_from_sqo_group and isinstance(filters_from_sqo_group, list):
                df_to_group = apply_filters_to_dataframe(df_to_group, filters_from_sqo_group)
            grouped_data = df_to_group.groupby(group_by_cols)[agg_col].agg(agg_func)
            output_col_name = sqo.get('options', {}).get('output_column_name')
            if output_col_name:
                if isinstance(grouped_data, pd.Series): grouped_data = grouped_data.rename(output_col_name)
                elif isinstance(grouped_data, pd.DataFrame) and len(grouped_data.columns)==1: grouped_data.columns = [output_col_name]
            result_data = grouped_data.reset_index().to_dict(orient='records')

        elif operation_type == "find_top_n_rows":
            select_columns = sqo.get("select_columns")
            condition_col = sqo.get("condition_column")
            sort_order_str = sqo.get("sort_order", "descending").lower()
            n_rows_param = sqo.get("n_rows", 1)
            if not (select_columns and condition_col):
                error_msg = "'find_top_n_rows' 操作缺少 'select_columns' 或 'condition_column' 参数。"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            columns_to_check_top = []
            if isinstance(select_columns, list): columns_to_check_top.extend(select_columns)
            elif isinstance(select_columns, str): columns_to_check_top.append(select_columns)
            if condition_col: columns_to_check_top.append(condition_col)
            for col_check_top in columns_to_check_top:
                 if col_check_top not in df.columns:
                    error_msg = f"列 '{col_check_top}' 在Excel中未找到。可用列: {df.columns.tolist()}"
                    logger.error(error_msg)
                    return SQOResponse(success=False, error=error_msg)
            if sort_order_str not in ['ascending', 'descending']:
                error_msg = f"无效的 sort_order: {sort_order_str}"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            if not isinstance(n_rows_param, int) or n_rows_param <= 0:
                error_msg = f"n_rows 必须是正整数，但收到: {n_rows_param}"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            df_to_sort = df.copy()
            filters_from_sqo_sort = sqo.get("filters")
            if filters_from_sqo_sort and isinstance(filters_from_sqo_sort, list):
                df_to_sort = apply_filters_to_dataframe(df_to_sort, filters_from_sqo_sort)
            ascending_order = True if sort_order_str == 'ascending' else False
            sorted_df = df_to_sort.sort_values(by=condition_col, ascending=ascending_order)
            result_df = sorted_df.head(n_rows_param)
            result_data = result_df[select_columns].to_dict(orient='records')

        elif operation_type == "direct_sql_query":
            sql_query_from_sqo = sqo.get("sql_query")
            if not sql_query_from_sqo:
                error_msg = "'direct_sql_query' 操作缺少 'sql_query' 参数。"
                logger.error(error_msg)
                return SQOResponse(success=False, error=error_msg)
            if not _PANDASQL_IMPORTED: # Double-check
                 logger.critical("PandaSQL is required for 'direct_sql_query' but was not imported.")
                 return SQOResponse(success=False, error="内部服务器错误：PandaSQL库未能加载。", error_details="PandaSQL is None for direct_sql_query.")
            pdsql = PandaSQL(persist=False)
            logger.debug(f"Executing direct_sql_query: {sql_query_from_sqo} on columns: {df.columns.tolist()}")
            df_for_sql = df.copy()
            filters_from_sqo_sql = sqo.get("filters")
            if filters_from_sqo_sql and isinstance(filters_from_sqo_sql, list):
                df_for_sql = apply_filters_to_dataframe(df_for_sql, filters_from_sqo_sql)
            query_result_df = pdsql(sql_query_from_sqo, env={'df': df_for_sql})
            if query_result_df is None: result_data = "SQL查询成功执行，但没有返回数据。"
            elif query_result_df.empty: result_data = "SQL查询成功执行，但未找到符合条件的数据。"
            else:
                result_list_of_dicts = query_result_df.to_dict(orient='records')
                if len(result_list_of_dicts) == 1:
                    if len(result_list_of_dicts[0]) == 1: result_data = list(result_list_of_dicts[0].values())[0]
                    else: result_data = result_list_of_dicts[0]
                else:
                    if result_list_of_dicts and len(result_list_of_dicts[0]) == 1:
                        single_col_name = list(result_list_of_dicts[0].keys())[0]
                        result_data = [row[single_col_name] for row in result_list_of_dicts]
                    else: result_data = result_list_of_dicts
        else:
            error_msg = f"不支持的操作类型 '{operation_type}'。"
            logger.error(error_msg)
            return SQOResponse(success=False, error=error_msg)

        logger.info(f"SQO operation '{operation_type}' executed successfully.")
        return SQOResponse(success=True, result=result_data)

    except FileNotFoundError as e_fnf:
        logger.error(f"FileNotFoundError during pandas operation for file '{effective_file_path}': {e_fnf}", exc_info=True)
        return SQOResponse(success=False, error=f"Pandas操作时未找到文件: '{effective_file_path}'. 错误: {str(e_fnf)}", error_details=traceback.format_exc())
    except xml.etree.ElementTree.ParseError as e_xml:
        logger.error(f"XML ParseError reading Excel file '{effective_file_path}': {e_xml}", exc_info=True)
        return SQOResponse(success=False, error=f"读取Excel文件 '{os.path.basename(effective_file_path)}' 失败：文件格式错误或已损坏 (XML解析错误)。", error_details=traceback.format_exc())
    except pd.errors.EmptyDataError as e_empty:
        logger.error(f"Pandas EmptyDataError for file {effective_file_path}, sheet {sheet_name}: {e_empty}", exc_info=True)
        return SQOResponse(success=False, error=f"无法读取Excel文件 '{os.path.basename(effective_file_path)}' (工作表: {sheet_name})，文件可能为空或格式不正确。", error_details=traceback.format_exc())
    except KeyError as e_key:
        available_cols_str = df.columns.tolist() if df is not None and isinstance(df, pd.DataFrame) else '未知 (DataFrame未成功加载)'
        logger.error(f"KeyError during operation '{operation_type}': {e_key}. SQO: {sqo}", exc_info=True)
        return SQOResponse(success=False, error=f"操作失败：列名 '{str(e_key)}' 可能不存在或不正确。请检查SQO参数和Excel列名。可用列: {available_cols_str}", error_details=traceback.format_exc())
    except Exception as e:
        error_message = f"执行SQO操作 '{operation_type}' 时发生内部错误: {type(e).__name__} - {str(e)}"
        logger.error(error_message, exc_info=True)
        return SQOResponse(success=False, error=error_message, error_details=traceback.format_exc())

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Local Agent server on http://0.0.0.0:{LOCAL_AGENT_PORT}")
    uvicorn.run("local_agent_app:app", host="0.0.0.0", port=LOCAL_AGENT_PORT, reload=True)