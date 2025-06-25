# Hotel-Recommendation-ChatBot-for-India

This project implements a hotel recommendation chatbot using a combination of Large Language Models (LLMs), vector databases, and various tools to provide personalized hotel suggestions to users.

## Dataset

The core of this chatbot's knowledge base is derived from the [Indian Hotels on Goibibo](https://www.kaggle.com/datasets/PromptCloudHQ/hotels-on-goibibo?resource=download) dataset. This dataset contains information about hotels in India, featuring crucial information such as hotel names, addresses, and other details relevant for recommendations.

### Data Preparation

Before being utilized by the chatbot, the raw dataset undergoes a cleaning and processing phase. The processed data is then used to create documents, which are vectorized and stored in a ChromaDB vector database. This vector database is integral to the Retrieval Augmented Generation (RAG) mechanism, enabling the LLM to retrieve relevant information efficiently.

This implementation focuses on recommending hotels within India, leveraging the provided dataset. The data preparation process involves cleaning and processing this dataset to create documents for the ChromaDB.

## Implementation Details

This chatbot is built using the Langchain library, integrating several key components:

### 1. Large Language Models (LLM)

The system is designed to work with various LLMs, using the Openrouter API.

### 2. Prompts

The chatbot utilizes a ReAct (Reasoning and Acting) prompting style. This approach allows the LLM to engage in a structured thinking process, including selecting appropriate tools, before formulating a final answer. This thinking process involves:
- **Thought**: The LLM's internal reasoning about the user's query.
- **Action**: The decision to use a specific tool.
- **Action Input**: The input provided to the chosen tool.
- **Observation**: The result obtained from the tool's execution.
- **Final Answer**: The ultimate response to the user.

### 3. Embeddings

Sentence Transformer models are used to generate embeddings for the hotel review documents, facilitating efficient similarity searches within the vector database.

### 4. Vector Database

ChromaDB serves as the vector database, storing the vectorized hotel review documents. It plays a crucial role in the RAG process, allowing the LLM to retrieve contextually relevant information to answer user queries.

### 5. Tools

The chatbot integrates several tools to enhance its capabilities:
- **Retriever Tool**: Accesses information from the pre-processed hotel review documents stored in ChromaDB.
- **Online Search Tool**: Enables the chatbot to search for information not available in its internal knowledge base, providing up-to-date or external data.

### 6. Chat Memory

Conversation history is maintained to provide context for ongoing interactions. This memory allows the chatbot to understand and respond to queries that refer to previous parts of the conversation, ensuring a more natural and coherent user experience.

## Running the App

To run the application:

1.  Ensure you have a `.env` file in the project's root directory (same level as `src`) containing your OpenAI API key, HuggingFace API key, and SerpAPI key.
2.  Navigate to the `src` directory:
    ```commandline
    cd src
    ```
3.  Run the Streamlit application:
    ```commandline
    streamlit run app.py
    ```

## Project Structure

-   `src/`: Contains the main application scripts.
-   `notebook/`: Includes demo notebooks and exploratory data analysis.
-   `data/`: Stores raw and processed datasets, including the ChromaDB data.
-   `chroma_db/`: Houses the ChromaDB vector store.

## Future Improvements

-   Enhance the dataset with more diverse information to reduce reliance on online searches.
-   Further customize and refine the tools for improved accuracy and efficiency.
-   Explore larger and more advanced LLMs (e.g., ChatGPT 4, Llama 3) if computational resources permit.
