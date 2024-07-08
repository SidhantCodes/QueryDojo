import os
from dotenv import load_dotenv
import shutil

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

import azure.cognitiveservices.speech as speechsdk

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  

# Load environment variables from a .env file
load_dotenv()

# Initialize FastAPI instance
app = FastAPI()

# Middleware for handling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to extract text from uploaded PDF files
def get_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_from_audio(audio):
    speech_config=speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))

    speech_config.speech_recognition_language="en-US"
    audio_config = speechsdk.audio.AudioConfig(filename="test.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # print("Recognized: {}".format(speech_recognition_result.text))
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized: {}".format(speech_recognition_result.no_match_details)
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        return "Speech Recognition canceled: {}".format(cancellation_details.reason)
    

# Function to split text into manageable chunks
def split_text(text):
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = txt_splitter.split_text(text)

    return chunks

# Function to create and save FAISS index from text chunks
def get_embeddings(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Create FAISS index from text chunks and save locally
    vec_store = FAISS.from_texts(chunks, embedding=embeddings)
    vec_store.save_local("index")
    return "Creation of Vector Store completed"


# Define the conversation prompt template
def get_conversation_chain():
    # Template for conversation between chatbot and human
    prompt_tempt = """You are a chatbot, who goes by the name 'Superior DocuDojo' having a conversation with a human.

    Given the following extracted parts of a text book of class 10th science and a question, create a final answer. Do not give a wrong answer. If you dont know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'. Dont provide wrong answers. If a user asks a questions that goes beyond the provided context, kindly ask the user to ask questions that are relevant to the provided context

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    # Initialize Azure OpenAI chat model
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    # Create prompt template for the conversation
    prompt = PromptTemplate(template=prompt_tempt, input_variables=["chat_history", "human_input", "context"])

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    # Load question answering chain
    chain = load_qa_chain(
        model,
        chain_type="stuff",
        memory=memory,
        prompt=prompt
    )
    return chain

# Function to handle user input and query
def user_input(user_query, chain):
    # Initialize Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    # Load FAISS index from local storage
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search in FAISS index
    docs = db.similarity_search(user_query)
    # Invoke the conversation chain with input documents and user query
    res = chain.invoke({"input_documents": docs, "human_input": user_query})
    return res["output_text"]

# Event handler to initialize conversation chain on application startup
@app.on_event("startup")
async def startup_event():
    global chain
    chain = get_conversation_chain()

# Endpoint to upload PDF files and create text chunks and embeddings
@app.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    pdf_files = []

    # Copy uploaded files to temporary location and open as PDF files
    for file in files:
        with open(file.filename, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        pdf_files.append(open(file.filename, "rb"))
        file.file.close()

    try:
        # Extract text from PDF files and split into chunks
        text = get_text_from_pdf(pdf_files)
        chunks = split_text(text)

        # Create embeddings and save FAISS index
        result = get_embeddings(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Close and remove temporary files
        for pdf in pdf_files:
            pdf.close()
        for file in files:
            os.remove(file.filename)

    return JSONResponse(content={"message": result})

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only .wav files are allowed")
    
    try:
        # Save uploaded audio file temporarily
        with open("temp_audio.wav", "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        # Extract text from the audio file
        text = get_text_from_audio("temp_audio.wav")
        # Split extracted text into chunks
        chunks = split_text(text)
        # Create embeddings and save FAISS index
        result = get_embeddings(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove temporary audio file
        os.remove("temp_audio.wav")

    return JSONResponse(content={"message": result})

# Endpoint to query the chatbot with user input
@app.post("/query/")
async def query(user_query: str):
    if chain is None:
        raise HTTPException(status_code=500, detail="Chain is not initialized")
    elif 'index' not in os.listdir('.'):
        raise HTTPException(status_code=500, detail="Please create vector store")
    
    try:
        # Invoke user input processing with conversation chain
        result = user_input(user_query, chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"answer": result})