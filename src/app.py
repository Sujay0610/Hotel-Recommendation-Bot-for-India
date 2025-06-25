from data_preparation import CSVData
from langchain_openai import ChatOpenAI
from embeddings import Embeddings, EmbeddingType
from vector_databases import ChromaDB
from tools import RetrieverTool, OnlineSearchTool, ToolType
from prompts import ReactPrompt
from agents import Agents
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

import os
import openai
from dotenv import load_dotenv
import streamlit as st
from data_preparation.prepare_chroma_db import prepare_and_load_data_to_chroma

load_dotenv()

# Define ChromaDB path and collection name
CHROMADB_PERSIST_PATH = './chroma_db'
CHROMADB_COLLECTION_NAME = "hotel_recommendations"

def run(embedding_name, online_search=False):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        st.error("OPENROUTER_API_KEY not found. Please set it in your .env file.")
        st.stop()

    if online_search:
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_api_key:
            st.error("SERPAPI_API_KEY not found. Online search is enabled but the key is missing. Please set it in your .env file.")
            st.stop()
    # Ensure ChromaDB is prepared. Only run if the directory doesn't exist.
    if not os.path.exists(CHROMADB_PERSIST_PATH):
        st.info("Preparing hotel data and loading into ChromaDB. This may take a moment...")
        prepare_and_load_data_to_chroma(
            csv_path='data/raw/goibibo_com-travel_sample.csv',
            db_path=CHROMADB_PERSIST_PATH
        )
        st.success("ChromaDB prepared successfully!")

    llm = ChatOpenAI(
        model="mistralai/ministral-3b",
        openai_api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    embedding_model = Embeddings.get(embedding_name=embedding_name)

    vector_db = ChromaDB.get(
        embedding_model=embedding_model,
        persist_directory=CHROMADB_PERSIST_PATH,
        collection_name=CHROMADB_COLLECTION_NAME
    )
    retriever = vector_db.as_retriever(search_kwargs={'k': 5})
    tools = [RetrieverTool.get(retriever)]
    if online_search:
        tools.append(OnlineSearchTool.get())

    # use default setting: react prompt
    prompt = ReactPrompt(conversation_history=True).get()
    agent = Agents.get(llm=llm,
                       tools=tools,
                       prompt=prompt,
                       react=True,
                       verbose=True)
    # use default setting: chat memory
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # TODO: fix "treating as root run..'
    agent_executor = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    st.text("------------- Chatting -------------")
    question = st.text_input("Your question: ")
    full_question = f"Use the 'retriever-tool' first to answer: {question}. " \
                    f"If cannot get the answer, use other tool"
    submit = st.button("Ask!")

    # only response when all components are ready
    if agent_executor and full_question:
        print(f"[INFO] Agent is ready!")
        if submit:
            print(f"[INFO] Question: {full_question}")
            try:
                response = agent_executor.invoke(
                    {"input": full_question},
                    config={"configurable": {"session_id": "test-session"}},
                )
                st.write(response['output'])
            except openai.AuthenticationError as e:
                error_message = e.body.get('error', {}).get('message', 'Unknown authentication error.')
                st.error(f"Authentication Error: {error_message}. Please check your OPENROUTER_API_KEY.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")



if __name__ == "__main__":
    st.title("GuetGenie: A Hotel Recommendation Chatbot")

    # User input parameters


    # embedding_options = ["Sentence Transformer"]
    # REGISTRY_EMBEDDING = {
    #     embedding_options[0]: EmbeddingType.SENTENCE_TRANSFORMER
    # }
    # selected_embedding = st.selectbox("Select embedding: ", embedding_options)
    # selected_embedding = REGISTRY_EMBEDDING[selected_embedding]
    selected_embedding = EmbeddingType.SENTENCE_TRANSFORMER
    REGISTRY_SEARCH = {"No": False, "Yes": True}
    use_online_search = st.selectbox("Use online search? ", ["No", "Yes"])
    use_online_search = REGISTRY_SEARCH[use_online_search]

    run(embedding_name=selected_embedding,
        online_search=use_online_search)

