from PyPDF2 import PdfReader
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, UploadFile, File
import openai, os, asyncio, uvicorn
from pinecone import Pinecone as ps, ServerlessSpec
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

from concurrent.futures import ThreadPoolExecutor
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI   # deprecated in local but works on colab notebook
from langchain_community.chat_models import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings  # deprecated in local but works on colab notebook
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Dict, Any

load_dotenv()
# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAGServices:
    def __init__(self):
        self.pc = None 
        self.index = None
        self.retriever = None
        self.llm = None
        self.rag_model = None
        self.embedding = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.uploaded_documents = set()  # Track uploaded documents
        self._initialize_services()

    def _initialize_services(self):
        '''
        Initializing all the services once when its started
        '''
        try:
            self.pc = ps(
                api_key="pcsk_2E7WjQ_KNr1o8NERYCFbP1WMFYj5ygbaCwRtvynTEqAxx2tJzTnQcAqmY33zEetUgmoPXc")
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )

            # Create or connect to the index
            index_name = "yardstick-qa2"
            if index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=spec
                )
            else:
                print(f"Index '{index_name}' already exists.")  
            
            self.index = self.pc.Index(index_name)
            self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            self.retriever = Pinecone(
                index=self.index,
                embedding=self.embedding.embed_query,
                text_key='text'
            )
            
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Fixed model name and reduced temperature

            print("RAG Service initialized successfully")

        except Exception as e:
            raise HTTPException(400, detail=str(e))

    def _create_rag_chain(self, document_filter: List[str] = None):
        """Create a RAG chain with optional document filtering"""
        search_kwargs = {'k': 5}  # Increased k for better context
        
        if document_filter:
            search_kwargs["filter"] = {"doc_name": {"$in": document_filter}}
        
        retriever = self.retriever.as_retriever(search_kwargs=search_kwargs)
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=False,  # This helps with debugging
            chain_type_kwargs={
                "prompt": self._create_custom_prompt()
            }
        )

    def _create_custom_prompt(self):
        """Create a custom prompt template for better context understanding"""
        from langchain.prompts import PromptTemplate
        
        template = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based on the provided context from the uploaded documents
- If the context contains relevant information, provide a comprehensive answer
- If you cannot find relevant information in the context, then search on the net without considering context"
- Be specific and cite relevant parts of the documents when possible
- If the question refers to assignments, projects, or specific content, look for that information in the provided context

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    async def process_query_async(self, query: str, document_filter: List[str] = None) -> Dict[str, Any]:
        """
        Function to process the query asynchronously with optional document filtering
        """
        loop = asyncio.get_event_loop()
        try:
            # Create RAG chain with appropriate filtering
            if not document_filter and self.uploaded_documents:
                # If no specific filter, use all uploaded documents
                document_filter = list(self.uploaded_documents)
            
            rag_chain = self._create_rag_chain(document_filter)
            
            # Run the synchronous RAG model in a thread pool
            result = await loop.run_in_executor(
                self.executor, 
                rag_chain, 
                {"query": query}
            )
            answer = result["result"]
            # return {
                # "answer": result["result"],
                # "source_documents": [doc.metadata for doc in result.get("source_documents", [])],
                # "documents_searched": document_filter or list(self.uploaded_documents)
            # }
            return answer
        except Exception as e:
            print(f"Query processing error: {e}")
            raise e
    async def upload_pdf_async(self, contents: bytes, file_name)-> dict:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._process_pdf_sync,
                contents,
                file_name
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Pdf Upload failed {str(e)}')
    

    def _process_pdf_sync(self, contents: bytes, file_name) -> dict:
       bytes_content = BytesIO(contents)
       all_texts = self.extract_text_from_pdfs(bytes_content)
   
       document_name = file_name  # Or extract from user/file name
       batch_size = 100
       for i in range(0, len(all_texts), batch_size):
           batch = all_texts[i:i + batch_size]
           upsert_data = []
           for j, text in enumerate(batch):
               chunk_embedding = self.embedding.embed_query(text)
               upsert_data.append(
                   (
                       f"{document_name}-chunk-{i + j}",
                       chunk_embedding,
                       {
                           "text": text,
                           "doc_name": document_name,
                           "page_number": i + j
                       }
                   )
               )
           self.index.upsert(upsert_data)
   
       return {"message": f"PDF uploaded successfully. {len(all_texts)} chunks processed."}
    
    
    def extract_text_from_pdfs(self, pdf_file):
        # all_texts = []
        reader = PdfReader(pdf_file)
        all_chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
        
            page_number = i + 1
            # Splitting the text into smaller chunks using the RecursiveCharacterTextSplitter
            txt_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )    
            pages_chunks_texts = txt_splitter.split_text(text)
            # all_texts.extend(texts)
            for chunk_id, chunk in enumerate(pages_chunks_texts):
                enriched_chunk = f"[Page {page_number}] {chunk.strip()}"
                all_chunks.append(enriched_chunk)
            # return texts
            return all_chunks
    
rag_services = RAGServices()
app = FastAPI(title='Q-A bot', version='2.0')

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


@app.get('/')
async def home_page():
    return {"message": "RAG API is running", "status": "healthy"}


@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    file_name = file.filename
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=500, detail='Only pdf files are supported')
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=500, detail='File is too large. Maximum file size is 10Mb')
    result = await rag_services.upload_pdf_async(content, file_name)
    return result


@app.get('/')
def homePage():
    return {'HomePage'}

@app.post('/chat')
async def qa_chatbot(req: QueryRequest):
    ques = req.query
    if not req.query:
        raise HTTPException(status_code=400, detail='Query failed')
    if len(req.query) > 1000:
        raise HTTPException(status_code= 400, detail='Query is too long for this smaller model')
    try:
        new_query = f'Based on uploaded PDF document, {ques}'
        print(new_query)
        answer = await rag_services.process_query_async(new_query.strip())
        return {"query": new_query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, workers= 1, loop='asyncio')