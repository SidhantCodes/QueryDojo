
# Superior DocuDojo

## Overview

This application implements a document-based question answering (QA) chatbot, 'Superior DocuDojo' using FastAPI and Azure OpenAI services. It allows users to upload PDF documents, extract text, create embeddings using Azure OpenAI, and query the chatbot for answers related to the uploaded documents.

## Frameworks and Libraries Used

- **FastAPI**: FastAPI is used to create the REST API endpoints for uploading PDFs and querying the chatbot.
- **Azure OpenAI**: Azure OpenAI services are used for both embeddings (text similarity) and chat responses.
- **langchain**: Various components from langchain library are used for document loading, text splitting, and question answering chain setup.

## Features

- **Document Upload**: Users can upload PDF files containing educational content.
- **Text Extraction and Splitting**: Uploaded PDFs are processed to extract text and split into manageable chunks using RecursiveCharacterTextSplitter.
- **Embeddings Creation**: Text chunks are converted into embeddings using Azure OpenAI services and stored in a local FAISS index for efficient similarity search.
- **Question Answering Chatbot**: Users can query the chatbot with questions related to the uploaded documents. The chatbot uses a predefined conversation template and maintains context memory using ConversationBufferMemory to provide relevant answers based on past interactions.
- **Error Handling**: The API handles errors gracefully, returning appropriate HTTP status codes and error messages.

## Implementation of Context Memory

The chatbot implements context memory using `ConversationBufferMemory`. This allows the chatbot to remember past interactions (chat history) and use them to provide more contextually relevant responses in future interactions. The memory is stored and managed within the application runtime, facilitating a more coherent and human-like conversation experience.

## Installation

To run the application, follow these steps:

1. **Create a Virtual Environment**:
   ```
   python -m venv venv
   ```

2. **Create `.env` file**:
   Populate the `.env` file with the following Azure OpenAI credentials obtained from Azure OpenAI Studio:
   ```
   AZURE_OPENAI_ENDPOINT=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=xxxxxxxxxxxxxxxx
   ```

3. **Install Required Packages**:
   Activate the virtual environment and install dependencies:
   ```
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   Start the FastAPI application using uvicorn:
   ```
   uvicorn main:app --reload
   ```

5. **Interact with the API**:
   - Use Postman or similar tools to test API endpoints (`/upload_pdfs/` for uploading PDFs and `/query/` for querying the chatbot).
   - Alternatively, visit `http://localhost:8000/docs` in your browser to access the Swagger UI documentation and interact with the API.

By following these steps, you can deploy and interact with the Superior DocuDojo's API.

