# QueryDojo

## Overview

QueryDojo is an advanced question-answering (QA) application that leverages FastAPI and Azure OpenAI services to provide intelligent responses to user queries. Users can upload PDF documents, audio files, YouTube video links, and images, extract text or relevant information, create embeddings using Azure OpenAI, and query the chatbot for answers related to the uploaded content.

## Frameworks and Libraries Used

- **FastAPI**: To create REST API endpoints for uploading content and querying the chatbot.
- **Azure OpenAI**: For embeddings (text similarity) and chat responses.
- **langchain**: For document loading, text splitting, and question-answering chain setup.
- **PyPDF2**: For extracting text from PDF documents.
- **azure.cognitiveservices.speech**: For audio transcription.
- **YouTubeTranscriptApi**: For fetching YouTube video transcripts.
- **azure.ai.vision.imageanalysis**: For image analysis and text extraction.
- **FAISS**: For efficient similarity search using embeddings.

## Features

- **Content Upload**: Users can upload PDF documents, audio files, YouTube video links, and images.
- **Text Extraction and Splitting**: Extracts text from various content types and splits it into manageable chunks using RecursiveCharacterTextSplitter.
- **Embeddings Creation**: Converts text chunks into embeddings using Azure OpenAI services and stores them in a local FAISS index for efficient similarity search.
- **Question Answering Chatbot**: Users can query the chatbot with questions related to the uploaded content. The chatbot uses predefined conversation templates and maintains context memory using ConversationBufferMemory to provide relevant answers based on past interactions.
- **Error Handling**: The API handles errors gracefully, returning appropriate HTTP status codes and error messages.

## Implementation of Context Memory

The chatbot implements context memory using `ConversationBufferMemory`. This allows the chatbot to remember past interactions (chat history) and use them to provide more contextually relevant responses in future interactions. The memory is stored and managed within the application runtime, facilitating a more coherent and human-like conversation experience.

## Installation

To run the application, follow these steps:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Create `.env` file**:
   Populate the `.env` file with the following Azure credentials obtained from Azure Portal:
   ```plaintext
   AZURE_OPENAI_ENDPOINT=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=xxxxxxxxxxxxxxxx
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=xxxxxxxxxxxxxxxx
   AZURE_SPEECH_KEY=xxxxxxxxxxxxxxxx
   AZURE_SPEECH_REGION=xxxxxxxxxxxxxxxx
   VISION_ENDPOINT=xxxxxxxxxxxxxxxx
   VISION_KEY=xxxxxxxxxxxxxxxx
   ```

3. **Install Required Packages**:
   Activate the virtual environment and install dependencies:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   Start the FastAPI application using uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

5. **Interact with the API**:
   - Use Postman or similar tools to test API endpoints (`/upload_pdfs/`, `/upload_audio/`, `/youtube_vid/`, `/upload_image/` for uploading content and `/query/` for querying the chatbot).
   - Alternatively, visit `http://localhost:8000/docs` in your browser to access the Swagger UI documentation and interact with the API.

By following these steps, you can deploy and interact with QueryDojo's API, leveraging its capabilities to process and query various types of content using advanced AI services.