## pip install pypdf langchain_community langchain_text_splitters langchain_chroma
## pip install -U sentence_transformers langchain_huggingface

# import warnings
# warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


## Extracción de información PDF en partes
file_path = "./bitcoin_whitepaper.pdf"  # Ruta al archivo PDF
docs = PyPDFLoader(
    file_path
).load()  # Lo divide en partes. En este caso 9 (nº de páginas del PDF)

# print(docs)

# Utiliza RecursiveCharacterTextSplitter para dividir el texto en fragmentos
# chunk_size: Tamaño de cada fragmento de texto en caracteres
# chunk_overlap: Número de caracteres que se superponen entre fragmentos consecutivos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)  # Lo divide en partes más pequeñas

# Modelo de lenguaje de HuggingFace para generar embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma crea un almacén vectorial a partir de los fragmentos y sus embeddings
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory="./vectordb"
)

print(f"Embedding del documento {file_path} guardado en la base de datos!")

# El almacén vectorial se convierte en un sistema de recuperación de información (retriever) que permite realizar consultas y obtener resultados relevantes basados en la similitud de embeddings.
# retriever = vectorstore.as_retriever()

# Consulta al almacén vectorial
# result = retriever.invoke("Cual es el objetivo de bitcoin")
# print(result)
