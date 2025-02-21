# todo: build a RAG pipeline -->
#  1) loading data;
#  2) indexing (building vector embeddings);
#  3) storing (so no need to re-index the whole shit);
#  4) querying;
#  5) evaluation
from loguru import logger

from scipy.stats import loggamma

from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from core.ai.rag_llama.consts import EMBEDDING_MODEL


class BuildRAG:
    async def loading_data(self):
        # doc: list[Document] = SimpleDirectoryReader('').load_data()
        embed_model = FastEmbedEmbedding(model_name=EMBEDDING_MODEL)
        embeddings = embed_model.get_text_embedding('tcv2catnap この人、移民と難民、不法の区別がついていないみたいですよ。日本に難民はこない、と書いていますから。ウクライナ難民とか知らないんでしょう')
        logger.info(embeddings)


