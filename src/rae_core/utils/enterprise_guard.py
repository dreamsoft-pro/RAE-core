# rae_core/utils/enterprise_guard.py
import os
import time
import psutil
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, Optional
from datetime import datetime

from rae_core.utils.memory_bridge import RAEMemoryBridge
from rae_core.utils.context import RAEContextLocator

class FatalEnterpriseError(Exception):
    """Raised when an Enterprise Contract is violated."""
    pass

class RAE_Enterprise_Foundation:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(f"RAE.Enterprise.{module_name}")
        self.project_name = RAEContextLocator.get_project_name()
        self.bridge = RAEMemoryBridge(project_name=module_name)

    def verify_contract(self, operation_name: str, payload: Any):
        """Hardware-coded validation of Hard Frames and Security Contracts."""
        forbidden_patterns = ["vim", "nano", "ssh", "ftp", "htop"]
        
        # 1. No Interactive Commands Check
        payload_str = str(payload).lower()
        for pattern in forbidden_patterns:
            if pattern in payload_str:
                error_msg = f"CONTRACT VIOLATION: Forbidden interactive pattern '{pattern}' detected in {operation_name}."
                self.logger.error(error_msg)
                raise FatalEnterpriseError(error_msg)
        
        return True

def audited_operation(operation_name: str, impact_level: str = "low"):
    """Decorator for auditing agentic operations with automatic context tracking."""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(self_instance, *args, **kwargs):
                return await _execute_with_audit(func, self_instance, operation_name, impact_level, True, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(self_instance, *args, **kwargs):
                return _execute_with_audit(func, self_instance, operation_name, impact_level, False, *args, **kwargs)
            return sync_wrapper
            
    return decorator

async def _execute_with_audit(func, self_instance, op_name, impact, is_async, *args, **kwargs):
    if not hasattr(self_instance, "enterprise_foundation"):
        raise FatalEnterpriseError(f"Class {self_instance.__class__.__name__} must implement RAE_Enterprise_Foundation.")
    
    foundation: RAE_Enterprise_Foundation = self_instance.enterprise_foundation
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # 1. Log START event
    human_label = f"[{foundation.module_name.upper()}] Operation: {op_name.replace('_', ' ').title()} (START)"
    foundation.bridge.save_event(
        content=f"Starting operation {op_name}",
        human_label=human_label,
        metadata={"impact": impact, "status": "started"}
    )

    try:
        # Pre-execution Contract Validation
        foundation.verify_contract(op_name, {"args": args, "kwargs": kwargs})
        
        if is_async:
            result = await func(self_instance, *args, **kwargs)
        else:
            result = func(self_instance, *args, **kwargs)
            
        duration = time.time() - start_time
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 2. Log SUCCESS event
        foundation.bridge.save_event(
            content=f"Completed operation {op_name}",
            human_label=f"[{foundation.module_name.upper()}] Operation: {op_name.replace('_', ' ').title()} (TELEMETRY: SUCCESS)",
            metadata={
                "duration_s": round(duration, 4),
                "mem_delta_mb": round(mem_after - mem_before, 2),
                "status": "success"
            }
        )
        return result
    except Exception as e:
        # 3. Log FAILURE event
        foundation.bridge.save_event(
            content=f"Failed operation {op_name}: {str(e)}",
            human_label=f"[{foundation.module_name.upper()}] Operation: {op_name.replace('_', ' ').title()} (CRITICAL: FAILURE)",
            metadata={"error": str(e), "status": "failed"}
        )
        raise
