# FastAPI-Based QA Chatbot with Pinecone and OpenAI
## https://frontendqa.netlify.app/
## **Overview**
This project implements a Question Answering (QA) chatbot using FastAPI as the backend framework. It integrates Pinecone for vector-based similarity search and OpenAI for language model embeddings and responses. The bot is capable of:

- Extracting and embedding text from PDFs.
- Storing embeddings in a Pinecone vector database.
- Answering user queries by retrieving relevant context from the vector database.
- Supporting CORS to allow requests from different origins.

## **Technologies Used**
- **FastAPI**: For building the REST API.
- **Pinecone**: Vector database for storing and retrieving embeddings.
- **OpenAI API**: For text embeddings and chat model (GPT-3.5-turbo).
- **PyPDF2**: For extracting text from PDF files.
- **LangChain**: For managing embeddings, text splitting, and retrieval workflows.
- **Uvicorn**: For running the FastAPI application.

---

## **Features**
1. **PDF Text Extraction**: Extracts and splits text into chunks for efficient embedding.
2. **Vector Storage**: Uses Pinecone to store and manage text embeddings.
3. **Contextual Question Answering**: Matches user queries to relevant chunks in the database and provides answers using GPT-3.5-turbo.
4. **CORS Support**: Configured to allow cross-origin requests, enabling integration with frontend applications.

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8+
- API keys for OpenAI and Pinecone.
- Pip for dependency management.

### **2. Installation**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=<paste your key here>
   PINECONE_API_KEY=<paste your key here>
   ```

### **3. Running the Application**

Start the FastAPI server:
```bash
uvicorn main:app --reload

```

The API will be accessible at `http://127.0.0.1:8000`.

---

## **API Endpoints**

### **1. Home Page**
**Endpoint**: `GET /`

- **Description**: Verifies that the server is running.
- **Response**:
  ```json
  {
      "HomePage": "HomePage"
  }
  ```

### **2. Chatbot Endpoint**
**Endpoint**: `POST /chat`

- **Description**: Fromate for handles user queries and provides answers.
- **Request Body**:
  ```json
  {
      "query": "<Your question here>"
  }
  ```
- **Response**:
  ```json
  {
      "query": "<Your question>",
      "answer": "<Generated answer>"
  }
  ```

---

## **Frontend Integration**

The backend includes CORS middleware configured for seamless integration with frontend applications. Ensure the `origins` list in the `CORSMiddleware` settings includes your frontend's URL:

```python
origins = [
    "http://localhost:3000",  # Local frontend
]
```

---

## **Known Issues**
1. **CORS Errors**: Ensure proper configuration of CORS middleware to match frontend requests.
2. **PDF Parsing Limitations**: PyPDF2 may not extract text accurately from certain PDF formats.


---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Contact**
For queries contact:
- Email: [sharadnaik001@gmail.com]


