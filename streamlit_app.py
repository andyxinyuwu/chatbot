import os
import streamlit as st
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader, load_index_from_storage

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info(
    "Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)",
    icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading FAQ docs – hang tight!"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                                                                  system_prompt="You are an expert on the Pietra FAQ and your job is to answer questions. Assume that all questions are related to the Pietra. Keep your answers based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)


        # print(os.path.abspath(__file__))
        #
        # persist_dir = os.path.dirname(os.path.abspath(__file__)) + '/storage/'
        # index = load_index_from_storage(
        #     storage_context=StorageContext.from_defaults(persist_dir=persist_dir),
        # )
        return index


index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

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