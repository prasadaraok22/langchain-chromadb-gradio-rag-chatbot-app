import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.chains import create_retrieval_chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import CSVLoader

import truststore
import logging
import httpx

import gradio as gr

api_key = os.environ.get("MISTRAL_API_KEY")
print(f"API key: ", api_key)

# 1. SSL Configuration - Optional
# To disable SSL, start
#Fix SSL for 'api.mistral.ai' and 'us.i.posthog.com'
truststore.inject_into_ssl()
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
# Create a client that ignores SSL verification
custom_client = httpx.Client(verify=False)
# Disable SSL end

# 2. Configure Logging - Enable to debug when getting failures
#logging.basicConfig(level=logging.DEBUG)

# 3. Prepare data and split into chunks
## All types of document loaders
def document_loader():
    pdf_file_path = "./fda-approved-drug.pdf"
    loader = PyPDFLoader(pdf_file_path)
    loaded_document = loader.load()
    print("File loaded successfully")
    return loaded_document

# 3.1 PDF document loader
def document_loader_pdf(file):
    print(f"Loading pdf file: {file.name}")
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    print("PDF File loaded successfully")
    return loaded_document

# 3.2 JSON document loader
def document_loader_json(file):
    print(f"Loading json file: {file.name}")
    loader = JSONLoader(file_path=file.name, jq_schema=".", text_content=False,)
    loaded_document = loader.load()
    print("JSON File loaded successfully")
    return loaded_document

# 3.3 CSV document loader
def document_loader_csv(file):
    print(f"Loading csv file: {file.name}")
    loader = CSVLoader(file.name)
    loaded_document = loader.load()
    print("CSV File loaded successfully")
    return loaded_document

# Function to handle the file upload form
def upload_file_function(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".json"):
            add_documents_to_vector_database(text_splitter(document_loader_json(uploaded_file)))
        elif uploaded_file.name.endswith(".pdf"):
            add_documents_to_vector_database(text_splitter(document_loader_pdf(uploaded_file)))
        elif uploaded_file.name.endswith(".csv"):
            add_documents_to_vector_database(text_splitter(document_loader_csv(uploaded_file)))

        return "File uploaded successfully."
    else:
        return "No file uploaded."

# 4. Text Splitter
def text_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(text)
    print("Text chunks created successfully")
    return chunks

# 4. Create the vectorstore (empty)   --- later keep adding documents
def create_vector_database():
    print("Creating empty vector database")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_db = Chroma(embedding_function=embeddings)
    print("Empty vector database created successfully")
    return vector_db

# 4.1 Add documents to the vectorstore
def add_documents_to_vector_database(chunks):
    print("Adding documents to vector database")
    vector_db.add_documents(documents=chunks)
    print("Documents added successfully")

# 5. Vector Storage with Mistral Embeddings
def retriever():
    print("Creating retriever")
    return vector_db.as_retriever(search_kwargs={"k": 120})

# 6. Mistral LLM Integration
# Latest models for 2025 include 'mistral-large-latest' or 'mistral-small-latest'
def get_llm():

    # 1. Store the original __init__ method
    original_init = httpx.Client.__init__

    # 2. Define a patched version that removes 'verify' if present and forces it to False
    def patched_init(self, *args, **kwargs):
        kwargs.pop('verify', None)  # Remove any 'verify' arg to avoid TypeError
        kwargs['verify'] = False    # Force disable verification
        original_init(self, *args, **kwargs)

    # 3. Apply the patch globally to httpx.Client
    httpx.Client.__init__ = patched_init

    # 4. Initialize the model as normal
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=api_key
    )
    print("Mistral client created")
    return llm

# 7. Define the RAG Prompt
system_prompt = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. Keep it concise.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 8. Create and Execute the Retrieval Chain
def ask_question(question):
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": question})
    return response

# Function to handle the question and answer form - UI calls this
def qa_function(question):
    if question:
        # Process the question and generate an answer
        response = ask_question(question)
        return response["answer"]
    else:
        return "Please enter a question."

print("Running RAG Chain...")
print("Initializing vector database and retriever...")
vector_db = create_vector_database()
retriever = retriever()
## Uncomment this to test directly instead UI
#add_documents_to_vector_database(text_splitter(document_loader()))
#response = ask_question("What are the most common adverse reactions of Iwilfin?")
#print(response["answer"])



