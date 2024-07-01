import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

load_dotenv()

def get_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text(text):
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = txt_splitter.split_text(text)

    return chunks

# Initialize embeddings
def get_embeddings(chunks):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Create and save FAISS index
    vec_store = FAISS.from_texts(chunks, embedding=embeddings)
    vec_store.save_local("index")
    return "Success:200 | Creation of Vector Store completed"

# Define the prompt template
def get_conversation_chain():
    prompt_tempt = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a text book of class 10th science and a question, create a final answer. Do not give a wrong answer. If you dont know the answer to a question, just respond with 'Sorry, the answer to the given question is not available in the provided context'

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    # Initialize the chat model
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    prompt = PromptTemplate(template=prompt_tempt, input_variables=["chat_history", "human_input", "context"])
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        memory=memory,
        prompt=prompt
    )

def user_input(user_query):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    chain = get_conversation_chain();
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_query)
    res = chain.invoke({"input_documents": docs, "human_input": user_query})
    return res["output_text"]
