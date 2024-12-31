from PyPDF2 import PdfReader
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, UploadFile, File
import openai
from pinecone import Pinecone as ps, ServerlessSpec
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO

from langchain.vectorstores import Pinecone
load_dotenv()
# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pc = ps(
    api_key="pcsk_2E7WjQ_KNr1o8NERYCFbP1WMFYj5ygbaCwRtvynTEqAxx2tJzTnQcAqmY33zEetUgmoPXc"
)

####
# Specify serverless environment
spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

# Create or connect to the index
index_name = "yardstick-qa2"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec
    )
else:
    print(f"Index '{index_name}' already exists.")  

index = pc.Index(index_name)
app = FastAPI()
# adding CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontendqa.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS for preflight
    allow_headers=["Content-Type", "Authorization"],  # List expected headers
)

# Define the input model
class QueryRequest(BaseModel):
    query: str

from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI   # deprecated in local but works on colab notebook
from langchain_community.chat_models import ChatOpenAI

def extract_text_from_pdfs(pdf_file):
    all_texts = []
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Splitting the text into smaller chunks using the RecursiveCharacterTextSplitter
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # length_function=len
    )    
    texts = txt_splitter.split_text(text)
    # all_texts.extend(texts)
    
    return texts

# from langchain.embeddings.openai import OpenAIEmbeddings  # deprecated in local but works on colab notebook
from langchain_community.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        bytes_contents = BytesIO(contents)
        all_texts = extract_text_from_pdfs(bytes_contents)
        for i, text in enumerate(all_texts):
            chunk_embedding = embedding.embed_query(text)
            index.upsert([(f"chunk-{i}", chunk_embedding, {"text": text})])
        return {"message": "PDF uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Example usage
# pdf_paths = ["./About Yardstick.pdf"]
# all_texts = extract_text_from_pdfs(pdf_paths)

# Embed and upsert each chunk into Pinecone



retriever = Pinecone(
    index=index,
    embedding=embedding.embed_query,
    text_key='text'
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Instead of directly passing 'retriever', use retriever.as_retriever()
rag_model = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever() # Call as_retriever() method
)

# print(f'Number of chunks = {len(all_texts)}')
# # print(f"First chunk:\n{all_texts[0]}")

@app.get('/')
def homePage():
    return {'HomePage'}

@app.post('/chat')
def qa_chatbot(req: QueryRequest):
    ques = req.query
    if not ques:
        raise HTTPException(status_code=400, detail='Query failed')
    
    try:
        answer = rag_model.run(ques)
        return {"query": ques, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.options("/chat")
def options_handler():
    return JSONResponse(content={"message": "Options Request OK"}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    