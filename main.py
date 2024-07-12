# import warnings
# warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain


# Esta función crea una plantilla de prompt para el sistema de chat. La plantilla define cómo se estructuran las entradas para el modelo de lenguaje.
# system: Proporciona las instrucciones para el asistente (instructions) y el contexto o la información recuperada del almacén vectorial (context)
# human: Proporciona la pregunta del usuario (input)
def create_prompt_template(instructions):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instructions + "\n\n" "{context}"),
            ("human", "{input}"),
        ]
    )
    return prompt


def response(question, llm, chroma_local, prompt):
    # Crea el sistema de recuperación a partir del almacen vectorial para realizar consultas y obtener resultados relevantes basados en la similitud de embeddings.
    retriever = chroma_local.as_retriever()

    # Combina un modelo de lenguaje y una plantilla de prompt. Organiza los fragmentos de texto recuperados y los inserta en la plantilla de prompt. Esto incluye llenar los placeholders como {context} con el contenido real de los documentos una vez se ejecute más adelante la función que lo rellene, en este caso será la función "create_retrieval_chain". El placeholder de {input} lo rellenará la función "invoke"
    chain = create_stuff_documents_chain(llm, prompt)

    # Rellena el context
    rag = create_retrieval_chain(retriever, chain)

    # Crea una respuesta a través de rellenar el "input" con "question"
    results = rag.invoke({"input": question})
    return results["answer"]


# A más temperatura, más creativo
# A menos temperatura, más racional
llm = ChatOllama(model="gemma:2b", temperature=0)

# Modelo de lenguaje de HuggingFace para generar embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Carga el almacén vectorial previamente creado en "store_in_bdd.py" que contiene los embeddings de los fragmentos de texto extraidos del PDF
chroma_local = Chroma(
    persist_directory="./vectordb",
    embedding_function=embeddings,
)

instructions = """Tú eres un asistente para tareas de respuesta a preguntas."
    "Usa los siguientes fragmentos de contexto recuperado para responder "
    "la pregunta. Si no sabes la respuesta, di que no "
    "sabes. Usa un máximo de tres oraciones y mantén la "
    "respuesta concisa."""

# Esta condición verifica si el archivo se está ejecutando como el programa principal. Si el script se ejecuta directamente (por ejemplo, usando python script.py), la variable especial __name__ se establece a "__main__".
if __name__ == "__main__":
    print(
        response(
            input("Haz tu pregunta: "),
            llm,
            chroma_local,
            create_prompt_template(instructions),
        )
    )
