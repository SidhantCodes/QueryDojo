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

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from youtube_transcript_api import YouTubeTranscriptApi

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
# Function to transcribe audio to text
def get_text_from_audio(audio):
    speech_config=speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))

    speech_config.speech_recognition_language="en-US"
    audio_config = speechsdk.audio.AudioConfig(filename=audio)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized: {}".format(speech_recognition_result.no_match_details)
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        return "Speech Recognition canceled: {}".format(cancellation_details.reason)
# Function to get transcript of youtube videos
def get_ytvid_transcript(link):
    vid_id = link.split("=")[1]
    transcript = ""

    transcript_list = YouTubeTranscriptApi.get_transcript(vid_id)

    for obj in transcript_list:
        transcript+=obj['text']
    
    return transcript
# Function to analyse image and generate relevant information in text
def get_image_data_text(image_data):
    try:
        endpoint = os.getenv("VISION_ENDPOINT")
        key = os.getenv("VISION_KEY")
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()
    
    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    visual_features =[
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.READ,
        VisualFeatures.SMART_CROPS,
        VisualFeatures.PEOPLE,
    ]
    result = client._analyze_from_image_data(
        image_content=image_data,
        visual_features=visual_features,
        gender_neutral_caption=True
    )
    tags_txt = "Tags: "
    obj_txt = "Objects: "
    caption_txt = "Caption: "
    dense_cap_txt = "Dense Captions: "
    read_txt = "Read: "

    if result.tags is not None:
        for obj in result.tags['values']:
            tags_txt += obj['name']+","
        tags_txt=tags_txt[:-1]

    if result.objects is not None:
        for obj in result.objects['values'][0]['tags']:
            obj_txt += f"{obj['name']}: confidence score: {obj['confidence']},"
        obj_txt=obj_txt[:-1]

    if result.caption is not None:
        caption_txt+=f"{result.caption['text']}- confidence score: {result.caption['confidence']}"

    if result.dense_captions is not None:
        for obj in result.dense_captions['values']:
            dense_cap_txt+=f"\n{obj['text']}- confidence score: {obj['confidence']}\n"

    if result.read is not None:
        for obj in result.read['blocks'][0]['lines']:
            read_txt+=f"\n{obj['text']}\n"

    formatted_image_analysis_result = f"Image analysis results:\n{tags_txt}\n{obj_txt}\n{caption_txt}\n{dense_cap_txt}\n{read_txt}\n"
    
    return formatted_image_analysis_result

# Function to split text into manageable chunks
def split_text(text):
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = txt_splitter.split_text(text)

    return chunks

# Function to create and save FAISS index from text chunks
def create_vectorstore(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Create FAISS index from text chunks and save locally
    vec_store = FAISS.from_texts(chunks, embedding=embeddings)
    vec_store.save_local("index")
    return "Creation of Vector Store completed"

# Define prompt templates for different file formats
pdf_prompt_template = """You are a chatbot named 'QueryDojo' having a conversation with a human.

Given the following extracted parts of a document (PDF) and a question, create a final answer. Do not give a wrong answer. If you don't know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'. Don't provide wrong answers. If a user asks a question that goes beyond the provided context, kindly ask the user to ask questions that are relevant to the provided context.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

audio_prompt_template = """You are a chatbot named 'QueryDojo' having a conversation with a human.

Given the following extracted parts of an audio file transcription and a question, create a final answer. Do not give a wrong answer. If you don't know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'. Don't provide wrong answers. If a user asks a question that goes beyond the provided context, kindly ask the user to ask questions that are relevant to the provided context.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

video_prompt_template = """You are a chatbot named 'QueryDojo' having a conversation with a human.

Given the following extracted parts of a youtub video transcript and a question, create a final answer. Do not give a wrong answer. If you don't know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'. Don't provide wrong answers. If a user asks a question that goes beyond the provided context, kindly ask the user to ask questions that are relevant to the provided context.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

image_prompt_template = """You are a chatbot named 'QueryDojo' having a conversation with a human.

Given the following extracted parts of an image analysis and a question, create a final answer. The provided context is the result of OCR performed on the image to extract all necessary information from the image, like captions, tags, read(the texts written in it). If anything is not there such as Captions ignore it. Do not give a wrong answer. If you don't know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'. Don't provide wrong answers. If a user asks a question that goes beyond the provided context, kindly ask the user to ask questions that are relevant to the provided context.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

# Function to get the appropriate prompt template based on the file type
def get_prompt_template(file_type):
    if file_type == "pdf":
        return pdf_prompt_template
    elif file_type == "audio":
        return audio_prompt_template
    elif file_type == "video":
        return video_prompt_template
    elif file_type == "image":
        return image_prompt_template
    else:
        raise ValueError("Unsupported file type")

# Define the conversation prompt template
def get_conversation_chain(file_type):
    # Get the appropriate prompt template based on the file type
    prompt_tempt = get_prompt_template(file_type)

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
    global pdf_chain, audio_chain, video_chain, image_chain
    pdf_chain = get_conversation_chain("pdf")
    audio_chain = get_conversation_chain("audio")
    video_chain = get_conversation_chain("video")
    image_chain = get_conversation_chain("image")

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
        result = create_vectorstore(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Close and remove temporary files
        for pdf in pdf_files:
            pdf.close()
        for file in files:
            os.remove(file.filename)

    return JSONResponse(content={"message": result, "chain": "pdf"})

# Endpoint to upload audio and create vector store
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only .wav files are allowed")
    
    try:
        # Save uploaded audio file temporarily
        with open(f"temp_audio.wav", "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        # Extract text from the audio file
        text = get_text_from_audio(f"temp_audio.wav")
        # Split extracted text into chunks
        chunks = split_text(text)
        # Create embeddings and save FAISS index
        result = create_vectorstore(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove temporary audio file
        os.remove(f"temp_audio.wav")

    return JSONResponse(content={"message": result, "chain": "audio"})

# Endpoint to upload link to youtube video and create vector store based on its transcript
@app.post('/youtube_vid/')
async def youtube_video_upload(link):
    try:
        transcript = get_ytvid_transcript(link)
        chunks = split_text(transcript)
        result = create_vectorstore(chunks)

        return JSONResponse(content={"message": result, "chain": "video"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Endpoint to upload image and create vector store based on its text analysis
@app.post('/upload_image/')
async def upload_image(image: UploadFile = File(...)):
    try:
        extension = image.filename.split('.')[1]
        with open(f"temp_image.{extension}", "wb") as temp_file:
            shutil.copyfileobj(image.file, temp_file)
        with open(f"temp_image.{extension}", "rb") as f:
            image_data = f.read()
        
        image_text=get_image_data_text(image_data)
        chunks=split_text(image_text)
        result=create_vectorstore(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(f"temp_image.{extension}")

    return JSONResponse(content={"message": result, "chain": "image"})

# Endpoint to query the chatbot with user input
@app.post("/query/")
async def query(user_query: str, chain_type: str):
    if chain_type == "pdf":
        chain = pdf_chain
    elif chain_type == "audio":
        chain = audio_chain
    elif chain_type == "video":
        chain = video_chain
    elif chain_type == "image":
        chain = image_chain
    elif 'index' not in os.listdir('.'):
        raise HTTPException(status_code=500, detail="Please create vector store")
    else:
        raise HTTPException(status_code=400, detail="Invalid chain type")
    
    try:
        # Invoke user input processing with conversation chain
        result = user_input(user_query, chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"answer": result})