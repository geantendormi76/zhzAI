# /home/zhz/zhz_agent/core/tools/calculator_tool.py
import logging
import re
# --- 添加开始 ---
import math 
# --- 添加结束 ---
from typing import Type, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

class CalculateToolInput(BaseModel):
    expression: str = Field(description="要计算的数学表达式字符串。例如：'(10 + 20) * 3 / 5 - 10'")

class CalculateTool(BaseTool):
    name: str = "calculate_tool"
    description: str = (
        "【计算器工具】执行数学表达式的计算并返回数值结果。支持常见的算术运算（加、减、乘、除、括号、幂运算、取模）以及部分数学函数 (如 sqrt, factorial, sin, cos, tan, log, log10, exp, degrees, radians)。" # <--- 更新描述
    )
    args_schema: Type[BaseModel] = CalculateToolInput

    def _run(self, expression: str, **kwargs: Any) -> str:
        logger.info(f"CalculateTool._run called with expression: '{expression}'")
        
        # 安全性检查：允许字母（用于函数名如sqrt, factorial, sin等）
        # 但仍然需要小心，更安全的做法是解析表达式并只允许白名单中的函数
        if not re.match(r"^[0-9a-zA-Z_+\-*/().\s%^!]+$", expression):
            logger.error(f"Invalid characters in expression: '{expression}'")
            return "错误：表达式中包含无效字符。"
        
        if len(expression) > 200: # 稍微放宽长度限制
            logger.error(f"Expression too long: '{expression}'")
            return "错误：表达式过长。"

        try:
            # --- 修改开始：准备安全的命名空间 ---
            safe_globals = {
                "__builtins__": {}, # 限制内置函数
                "sqrt": math.sqrt,
                "factorial": math.factorial,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "pow": pow, # math.pow 和内置 pow 行为类似，但内置的更通用
                "pi": math.pi,
                "e": math.e,
                "degrees": math.degrees,
                "radians": math.radians,
                # 你可以根据需要添加更多 math 模块中的安全函数
            }
            # 为了支持 x^y 这样的幂运算，eval本身用 **
            # 如果用户输入了 ^, 我们可以在这里替换一下，或者告知用户使用 **
            expression_to_eval = expression.replace('^', '**')
            # 移除阶乘符号 `!` 后的空格，并替换为 `factorial()`
            # 注意：这只是一个简单的替换，可能无法处理所有复杂情况，例如 `(5+2)!`
            expression_to_eval = re.sub(r"(\d+)\s*!", r"factorial(\1)", expression_to_eval)

            result = eval(expression_to_eval, safe_globals, {}) # 使用受限的命名空间
            # --- 修改结束 ---
            logger.info(f"Expression '{expression}' (evaluated as '{expression_to_eval}') evaluated to: {result}")
            return f"计算结果: {expression} = {result}"
        except NameError as e:
            logger.error(f"Error evaluating expression '{expression}': NameError - {e}")
            return f"错误：表达式中使用了未定义的函数或变量 (例如 '{e.name}')。支持的函数包括：sqrt, factorial, sin, cos, tan, log, log10, exp。"
        except ZeroDivisionError:
            logger.error(f"Error evaluating expression '{expression}': Division by zero")
            return "错误：表达式中存在除以零的操作。"
        except SyntaxError:
            logger.error(f"Error evaluating expression '{expression}': Syntax error")
            return "错误：数学表达式语法错误。"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}", exc_info=True)
            return f"计算表达式时发生错误: {str(e)}"