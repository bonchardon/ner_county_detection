from loguru import logger

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain_community.llms import VLLM

from core.ai.fine_tune_model.ner_recognizer.consts import LLM_MODEL_1, LLM_MODEL_2


class ModelBuilder:
    async def llm_builder(self):
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
        tokenizer: bool | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(LLM_MODEL_2)
        llm = VLLM(
            model=LLM_MODEL_2,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_new_tokens=128,
            top_k=10,
            top_p=0.95,
            temperature=0.8,
        )
        logger.info(llm)
