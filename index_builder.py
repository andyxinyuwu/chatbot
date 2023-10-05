from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

if __name__ == '__main__':
    llm = OpenAI(temperature=0, model="gpt-4-0613", api_key=openai.api_key)
    service_context = ServiceContext.from_defaults(llm=llm)

    faq_docs = SimpleDirectoryReader(
        input_dir="./data/"
    ).load_data()

    # build index
    index = VectorStoreIndex.from_documents(
        faq_docs, service_context=service_context
    )

    index.storage_context.persist(persist_dir="./storage/")
