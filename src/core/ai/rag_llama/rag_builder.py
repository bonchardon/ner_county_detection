# todo: build a RAG pipeline -->
#  1) load/collect md files with data;
#  2) break into chunks (if the file is too lengthy)

from asyncio import run
from collections.abc import Iterator

from loguru import logger

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from core.ai.rag_llama.consts import EMBEDDING_MODEL, RAG_DATA


class BuildRAG:
    async def load_documents(self) -> list[Document] | None:
        if not (data := TextLoader(RAG_DATA)):
            logger.error('There is an error when loading data.')
            return
        return data.load()

    async def embedding_data(self):
        embed_model: FastEmbedEmbedding = FastEmbedEmbedding(model_name=EMBEDDING_MODEL)
        documents: list[Document] | None = await self.load_documents()
        try:
            if not (embeddings := embed_model.get_text_embedding('\n'.join([doc.page_content for doc in documents]))):
                return
            logger.info(embeddings)
            return embeddings
        except AttributeError as exc:
            logger.error(exc)
