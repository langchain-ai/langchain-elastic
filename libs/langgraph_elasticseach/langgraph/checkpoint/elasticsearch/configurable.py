from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id
\
class Configurable:
    thread_id: Optional[str]
    checkpoint_ns: Optional[str]
    checkpoint_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "checkpoint_id": self.checkpoint_id,
        }

def get_configurable(config: Dict[str, Any]) -> Configurable:
    configurable = config.get("configurable", {})
    return Configurable(
        thread_id=configurable.get("thread_id"),
        checkpoint_ns=configurable.get("checkpoint_ns"),
        checkpoint_id=get_checkpoint_id(config),
    )