import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------------
# 1. Set API Key
# -----------------------------
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# -----------------------------
# 2. Load Document
# -----------------------------
loader = TextLoader("data/knowledge.txt", encoding="utf-8")
documents = loader.load()

# -----------------------------
# 3. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# 4. Create Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings()

# -----------------------------
# 5. Store in Vector DB (Persistent)
# -----------------------------
vectorstore = Chroma(
    collection_name="rag_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.add_documents(chunks)
vectorstore.persist()

# -----------------------------
# 6. Create Retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -----------------------------
# 7. Custom Prompt
# -----------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Use ONLY the context below to answer.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# -----------------------------
# 8. LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------
# 9. RAG Chain
# -----------------------------
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# -----------------------------
# 10. Ask Question
# -----------------------------
query = "What is RAG?"
result = rag_chain.invoke({"query": query})

print("Answer:\n", result["result"])

print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata)
