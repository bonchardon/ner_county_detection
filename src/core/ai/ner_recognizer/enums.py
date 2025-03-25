from enum import Enum

class StrEnum(str, Enum):
    """A class that combines `str` and `Enum` to behave like a StrEnum."""
    pass

class AiModels(StrEnum):
    GPT_4O_MINI: str = 'gpt-4o-mini'
    LLM_MODEL_3_1_SWALLOW: str = 'tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3'
    LLM_MODEL_3_8B_SUZUME: str = 'lightblue/suzume-llama-3-8B-japanese'
    ELYZA_LLAMA: str = 'elyza/ELYZA-japanese-Llama-2-7b-instruct'
    