import torch
from loguru import logger

from transformers import (
    pipeline,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    GenerationConfig,
    Pipeline
)
from langchain import HuggingFacePipeline

from core.ai.ner_recognizer.consts import LLM_MODEL_2


class ModelBuilder:
    async def llm_builder(self) -> None:
        """ Running vLLM using Docker:
        docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
        docker run -it \
             --rm \
             --network=host \
             --cpuset-cpus=<cpu-id-list, optional> \
             --cpuset-mems=<memory-node, optional> \
             vllm-cpu-env

        :return:
        """
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_2, torch_dtype=torch.float16, trust_remote_code=True)
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(LLM_MODEL_2)
        generation_config: GenerationConfig = GenerationConfig.from_pretrained(
            pretrained_model_name=LLM_MODEL_2,
            temperature=0.8,
            top_p=0.95,
            top_k=10,
            max_new_tokens=1024,
            tensor_parallel_size=1,
            trust_remote_code=True,

        )
        text_pipeline: Pipeline = pipeline(
            task='generation',
            model=LLM_MODEL_2,
            tokenizer=tokenizer,
            return_full_text=True,
            generation_config=generation_config

        )
        llm_pipeline: HuggingFacePipeline = HuggingFacePipeline(
            pipeline=text_pipeline,
            model_kwargs={'temperature': 0}
        )
        result = llm_pipeline(
            'Sankei_news そんなの提供できるなら、日本に来てるウクライナ難民も帰国ですね'
        )
        logger.info(result)
        return
