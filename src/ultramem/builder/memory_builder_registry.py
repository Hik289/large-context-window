from omegaconf import DictConfig
from ultramem.builder.memory_builder import MemoryBuilder
from ultramem.core.memory import AgentMemory
from ultramem.utils.llm import ChatCompletionModel


class MemoryBuilderRegistry:
    _builders: dict[str, type[MemoryBuilder]] = {}

    @classmethod
    def register(cls, file_type: str, builder_cls: type[MemoryBuilder]):
        cls._builders[file_type] = builder_cls

    @classmethod
    def get(cls, file_type: str, cfg: DictConfig, ultramem: AgentMemory, model_client: ChatCompletionModel) -> MemoryBuilder:
        builder_cls = cls._builders.get(file_type)
        if builder_cls is None:
            raise ValueError(f"No MemoryBuilder registered for '{file_type}'")
        return builder_cls(cfg, ultramem, model_client)
