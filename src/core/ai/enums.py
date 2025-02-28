from enum import StrEnum


class AiModels(StrEnum):
    GPT_4O_MINI: str = 'gpt-4o-mini'
    LLM_MODEL_3_1_SWALLOW: str = 'tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3'
    LLM_MODEL_3_8B_SUZUME: str = 'lightblue/suzume-llama-3-8B-japanese'
