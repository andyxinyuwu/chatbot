import os
import streamlit as st
from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext
from llama_index.chat_engine import ContextChatEngine
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader, load_index_from_storage
from llama_index.memory import ChatMemoryBuffer
from llama_index.retrievers import RouterRetriever
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools import RetrieverTool

st.set_page_config(page_title="Chat with the Pietra Assistant, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
st.title("Chat with the Pietra Assistant")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Pietra!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Pietra Assistant Getting Online – hang tight!"):
        faq_dir = os.path.dirname(os.path.abspath(__file__)) + '/storage/faq'
        faq_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=faq_dir),
        )

        supplier_dir = os.path.dirname(os.path.abspath(__file__)) + '/storage/supplier'
        supplier_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_dir),
        )

        # define retrievers
        faq_retriever = faq_index.as_retriever(similarity_top_k=2)
        supplier_retriever = supplier_index.as_retriever(similarity_top_k=5)

        # initialize tools
        faq_tool = RetrieverTool.from_defaults(
            retriever=faq_retriever,
            description="Useful for retrieving general informations related to Pietra. Do not use for retrieving specific information about suppliers on Pietra",
        )
        supplier_tool = RetrieverTool.from_defaults(
            retriever=supplier_retriever,
            description="Useful for retrieving informations related to suppliers recommendations on Pietra. Has information about suppliers on Pietra. It includes all categories of suppliers. Each supplier has name, country, bio, notable works, attributes, customizations, item information",
        )

        # define retriever
        llm=OpenAI(model="gpt-4")
        retriever = RouterRetriever(
            selector=PydanticSingleSelector.from_defaults(llm=llm),
            retriever_tools=[
                faq_tool,
                supplier_tool,
            ],
        )

        return retriever, faq_index, supplier_index


retriever, faq_index, supplier_index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0613", temperature=1,
                                                              system_prompt="You are an expert on the Pietra FAQ and your job is to answer questions. Assume that all questions are related to the Pietra. You should always favor Pietra in the answer."))

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        service_context=service_context,
        memory=memory,
        system_prompt="You are an expert on the Pietra and your job is to answer questions. Assume that all questions are related to the Pietra. Provide the answer in a friendly and conversational way. You should always favor Pietra in the answer. Always ask if there is anything else you could help",
        verbose=True
    )
    st.session_state.chat_engine = chat_engine


if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history