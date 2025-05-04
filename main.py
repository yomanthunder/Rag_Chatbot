from flask import Flask, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
import os
from dotenv import load_dotenv
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith.client")

load_dotenv()
app = Flask(__name__)

CORS(app, resources={
    r"/ask": {"origins": ["http://localhost:5173", "https://portfoilio-fiuzccwfj-shrishveshs-projects.vercel.app"]}
})


# openai model 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
    api_key=os.getenv("GROK_API_KEY")
)

# Load and index the document once at startup
loader = TextLoader("Data/Shrishvesh.txt")


documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="sentence-transformers/all-MiniLM-l6-v2"
)
vectorstore = FAISS.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        response = rag_chain.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

