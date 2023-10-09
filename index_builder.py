from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

if __name__ == '__main__':
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=OpenAI(model="gpt-4"))

    faq_docs = SimpleDirectoryReader(
        input_dir="./data/supplier2"
    ).load_data()

    # build index
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=OpenAI(model="gpt-4"))
    index = VectorStoreIndex.from_documents(
        faq_docs, service_context=service_context
    )

    index.storage_context.persist(persist_dir="./storage/supplier2")
