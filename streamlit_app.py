import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("Reports Reader")

data_directory = 'data/'

def get_document_names(directory):
    """ Get a list of document names in the specified directory """
    # List all files in the directory
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Use Streamlit's sidebar to list document names with numbers
st.sidebar.title('Document List')

# Get list of documents
documents = get_document_names(data_directory)

# Use a sidebar for display with numbers
for idx, document in enumerate(documents, start=1):
    st.sidebar.text(f"{idx}. {document}")



if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything from the document list"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the documents"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on Finance and banking Regulation and your job is to answer technical questions. Assume that all questions are related to files uploaded. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
