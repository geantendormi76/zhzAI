# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/custom_io_managers.py
import json
import os
from typing import List, Type, Union, get_args, get_origin, Any, Optional 
from dagster import UPathIOManager, InputContext, OutputContext, DagsterInvariantViolationError
from pydantic import BaseModel as PydanticBaseModel
from upath import UPath

class PydanticListJsonIOManager(UPathIOManager):
    extension: str = ".jsonl"

    def __init__(self, base_dir: Optional[str] = None): # Changed base_path to base_dir for clarity
        resolved_base_dir: UPath
        if base_dir:
            resolved_base_dir = UPath(base_dir).resolve() # Resolve to absolute path
        else:
            # Default to <DAGSTER_HOME>/storage/pydantic_jsonl_io
            # DAGSTER_HOME defaults to ~/.dagster, but can be overridden by env var
            dagster_home_str = os.getenv("DAGSTER_HOME", os.path.join(os.path.expanduser("~"), ".dagster"))
            resolved_base_dir = UPath(dagster_home_str) / "storage" / "pydantic_jsonl_io"
        
        # Ensure the directory exists
        try:
            resolved_base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Log this error appropriately, perhaps using a direct print if logger isn't set up yet
            # or re-raise as a Dagster-specific error.
            print(f"[PydanticListJsonIOManager __init__] ERROR: Could not create bafef __init__(self, base_dir: Optse directory {resolved_base_dir}: {e}")
            # Depending on Dagster's init sequence, context.log might not be available here.
            # It's safer to let UPathIOManager handle its own base_path or ensure dir exists before.
            # For now, we proceed, UPathIOManager might handle it or fail later.

        super().__init__(base_path=resolved_base_dir)
        # Log the final base path used by the UPathIOManager instance
        # self.log available after super().__init__() in ConfigurableIOManager context
        # For direct instantiation, we might need to pass a logger or use a global one.
        # print(f"[PydanticListJsonIOManager __init__] Initialized with resolved base_path: {self.base_path}")


    def dump_to_path(self, context: OutputContext, obj: List[PydanticBaseModel], path: UPath):
        context.log.info(f"[PydanticListJsonIOManager dump_to_path] Attempting to dump to resolved path: {path.resolve()}")
        
        if not isinstance(obj, list):
            msg = f"Expected a list of Pydantic models, got {type(obj)}"
            context.log.error(msg)
            raise TypeError(msg)
        
        # Optional: More robust type checking for list items if needed, using context.dagster_type
        # For now, assume obj is List[PydanticBaseModel] based on upstream asset's type hint.

        try:
            with path.open("w", encoding="utf-8") as f:
                for model_instance in obj:
                    if not isinstance(model_instance, PydanticBaseModel):
                        context.log.warning(f"Item in list is not a Pydantic model: {type(model_instance)}. Skipping.")
                        continue
                    json_str = model_instance.json() # Pydantic V1
                    f.write(json_str + "\n")
            context.log.info(f"[PydanticListJsonIOManager dump_to_path] Successfully dumped {len(obj)} items to {path.resolve()}")
        except Exception as e:
            context.log.error(f"[PydanticListJsonIOManager dump_to_path] Failed to dump object to {path.resolve()}: {e}", exc_info=True)
            raise

    def load_from_path(self, context: InputContext, path: UPath) -> List[PydanticBaseModel]:
        context.log.info(f"[PydanticListJsonIOManager load_from_path] Attempting to load from resolved path: {path.resolve()}")
        
        list_typing_type = context.dagster_type.typing_type
        origin = get_origin(list_typing_type)
        args = get_args(list_typing_type)

        if not (origin is list and args and issubclass(args[0], PydanticBaseModel)):
            msg = (
                f"PydanticListJsonIOManager can only handle inputs of type List[PydanticModel], "
                f"but got {list_typing_type} for input '{context.name}'."
            )
            context.log.error(msg)
            raise DagsterInvariantViolationError(msg) # Use Dagster specific error
        
        model_type: Type[PydanticBaseModel] = args[0]
        context.log.info(f"[PydanticListJsonIOManager load_from_path] Target model type for list items: {model_type.__name__}")

        loaded_models: List[PydanticBaseModel] = []
        if not path.exists():
            context.log.warning(f"[PydanticListJsonIOManager load_from_path] File not found at {path.resolve()}, returning empty list for input '{context.name}'.")
            return loaded_models

        try:
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line_content = line.strip()
                    if not line_content:
                        continue
                    try:
                        model_instance = model_type.parse_raw(line_content) # Pydantic V1
                        loaded_models.append(model_instance)
                    except Exception as e_parse:
                        context.log.error(
                            f"[PydanticListJsonIOManager load_from_path] Failed to parse JSON line {line_number} "
                            f"into {model_type.__name__} from {path.resolve()}: {e_parse}. "
                            f"Line content (first 100 chars): '{line_content[:100]}...'",
                            exc_info=True
                        )
                        # Optionally re-raise or decide to skip problematic lines
                        # For now, we'll skip
            context.log.info(f"[PydanticListJsonIOManager load_from_path] Successfully loaded {len(loaded_models)} instances of {model_type.__name__} from {path.resolve()}")
        except Exception as e_read:
            context.log.error(f"[PydanticListJsonIOManager load_from_path] Failed to read or process file {path.resolve()}: {e_read}", exc_info=True)
            raise # Re-raise if file reading itself fails catastrophically
            
        return loaded_models