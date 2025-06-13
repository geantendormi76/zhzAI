# 测试长代码块分割

这是一个包含长代码块的Markdown文档，用于测试我们基于空行分割代码的策略。

```python
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# 这是一个比较长的函数，包含多个逻辑部分和空行
def process_complex_data_structure(data: Dict[str, Any], threshold: float = 0.5, verbose: bool = False) -> Optional[List[str]]:
    """
    处理一个复杂的数据结构，进行过滤和转换。
    这个函数包含多个内部步骤，通过空行分隔。
    我们期望这些空行能成为PyInstaller分割的依据。
    """
    if not isinstance(data, dict):
        if verbose:
            logging.error("Input data must be a dictionary.")
        return None

    results: List[str] = []
    
    # 第一部分：数据验证和初步处理
    if verbose:
        logging.info(f"Processing data with {len(data)} top-level keys.")
    
    if 'items' not in data or not isinstance(data['items'], list):
        if verbose:
            logging.warning("Data does not contain a valid 'items' list. Skipping item processing.")
    else:
        for index, item in enumerate(data['items']):
            if not isinstance(item, dict) or 'value' not in item or 'name' not in item:
                if verbose:
                    logging.debug(f"Skipping invalid item at index {index}: {item}")
                continue
            
            item_value = item.get('value', 0.0)
            item_name = item.get('name', f'Unnamed Item {index}')
            
            if isinstance(item_value, (int, float)) and item_value > threshold:
                processed_name = item_name.strip().upper()
                results.append(f"VALID: {processed_name} - Value: {item_value:.2f}")
                if verbose:
                    logging.info(f"Item '{processed_name}' passed threshold with value {item_value}.")
            else:
                if verbose:
                    logging.debug(f"Item '{item_name}' did not pass threshold (value: {item_value}, type: {type(item_value)}).")


    # 第二部分：元数据处理和汇总
    metadata_section = data.get('metadata', {})
    report_id = metadata_section.get('report_id', 'N/A')
    creation_date = metadata_section.get('creation_date', 'Unknown')
    
    summary_line = f"Report ID: {report_id}, Created: {creation_date}, Valid Items Found: {len(results)}"
    results.insert(0, summary_line) # 将摘要放在结果列表的开头
    
    if verbose:
        logging.info("Finished processing metadata and generating summary.")
        logging.info(summary_line)

    # 第三部分：一些额外的计算或清理（模拟更多代码）
    # 这个部分故意留有很多空行，以测试分割器对多个连续空行的处理
    
    
    
    final_output_length = sum(len(s) for s in results)
    if verbose:
        logging.info(f"Total length of formatted results (excluding newlines): {final_output_length}")

    if not results:
        return ["No processable data found or all items filtered out."]
        
    return results

# 这是另一个独立的函数，也比较长
def utility_function_for_string_manipulation(input_string: str, mode: str = "default") -> str:
    """
    一个用于字符串操作的工具函数。
    它也有一些内部逻辑，并用空行分隔。
    """
    if not input_string:
        return ""

    if mode == "uppercase":
        return input_string.upper()
    
    elif mode == "lowercase":
        return input_string.lower()
        
    elif mode == "reverse":
        return input_string[::-1]
        
    elif mode == "capitalize_words":
        return ' '.join(word.capitalize() for word in input_string.split())
        
    # 默认行为或更多逻辑
    # ...
    # ...
    # ...
    
    
    
    
    # 确保这个函数也足够长
    temp_storage = []
    for i in range(5): # 模拟一些循环和操作
        temp_storage.append(f"Line {i}: {input_string[:10]}...")
    
    return f"Processed ({mode}): {input_string} -> Result based on complex internal logic and temp_storage: {' | '.join(temp_storage)}"

# 主调用部分（模拟）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_data = {
        "items": [
            {"name": "  item Alpha  ", "value": 0.8},
            {"name": "item Beta", "value": 0.2},
            {"name": "item Gamma", "value": 0.95, "extra_info": "important"},
            "not_a_dict_item", # 无效项
            {"name": "item Delta"} # 缺少value
        ],
        "metadata": {
            "report_id": "XYZ123",
            "creation_date": "2025-06-12"
        }
    }
    
    processed_results = process_complex_data_structure(sample_data, threshold=0.7, verbose=True)
    if processed_results:
        for line in processed_results:
            print(line)
            
    print("\\n--- Utility Function Test ---")
    print(utility_function_for_string_manipulation("hello world example for testing", mode="capitalize_words"))
    print(utility_function_for_string_manipulation("another test string", mode="reverse"))