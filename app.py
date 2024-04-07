import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from docx import Document
from jinja2 import Environment, FileSystemLoader
from docxtpl import DocxTemplate

load_dotenv()

st.set_page_config("CV's")
st.header("Transformación y Análisis Enriquecido de CV’s")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
file = st.file_uploader("Carga tu documento:", type=["pdf"], on_change=st.cache_resource.clear)

@st.cache_resource 
def create_embeddings(file):
    
    file_extension = ""
    file_name = ""
    text = ""

    if file is not None:
        # Obtener el nombre del archivo
        file_name = file.name
        
        # Obtener la extensión del archivo
        file_extension = file_name.split(".")[-1]
           
    
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
            
    # st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

if file:
    knowledge_base = create_embeddings(file)
    
    doc = DocxTemplate("formato.docx")
    llm = ChatOpenAI(model_name='gpt-4')
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Nombre
    question_nombre = "Por favor, proporciona el nombre del individuo cuya CV se describe a continuación, escribe unicamente el nombre y apellido:"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_nombre, 3)
    nombre = chain.run(input_documents=docs, question=question_nombre)

    # Nacionalidad
    question_nacionalidad = "Escribe el pais donde se encuetra ubicado la persona"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_nacionalidad, 3)
    nacionalidad = chain.run(input_documents=docs, question="Cual es la nacionalidad de la persona ? Escribe unicamente el país")
 
    # Ubicacion
    question_ubicacion = "En que país se encuentra ubicada la persona ? Escribe unicamente el País y ciudad, si tiene ciudad, sino, no la escribas "
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    ubicacion = chain.run(input_documents=docs, question=question_ubicacion)

    # Resumen

    question_resumen = "Genera un resumen en un parrafor de 4 lineas del perfil de la persona"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    resumen = chain.run(input_documents=docs, question=question_resumen)

    # formacion

    question_formacion = "Genera un listado de cada una de las formaciones, educativas, como universidad, que ha realizado la persona, enunciando el nombre donde ha realizado sus estudios"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    formacion = chain.run(input_documents=docs, question=question_formacion)
    
    # cursos y certificaciones

    question_cursos = "Genera un listado de las certificaciones, cursos o estudios mencionados con las que cuenta la persona, enunciando la escuela o entidad  donde la ha realizado y en que fecha"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    cursos = chain.run(input_documents=docs, question=question_cursos)
    
    # tecnologias

    question_tecnologias = "Genera un listado de las tecnologias de programacion e ingenieria en las cuales tiene experiencia"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    tecnologias = chain.run(input_documents=docs, question=question_tecnologias)
    
    # experiencia

    question_experiencia = "Genera un listado de la experiencia laboral que tiene la persona trabajando en otras empresas, menciona el nombre de la empresa, el cargo que ocupo y las fechas en las cuales trabajó ahí, sino es clara para ti las fechas, no las escribas"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    experiencia = chain.run(input_documents=docs, question=question_experiencia)
    
    # Insights

    question_Insights = "Cuales son los insights que mas destacas de la persona"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question_ubicacion, 3)
    Insights = chain.run(input_documents=docs, question=question_Insights)
    
    
    
    context = { 
               'nombre' : nombre,
               'nacionalidad' : nacionalidad,
               'ubicacion' : ubicacion,
               'resumen': resumen,
               'formacion': formacion,
               'cursos': cursos,
               'tecnologias': tecnologias,
               'experiencia': experiencia
               }
    
    doc.render(context)
    
    doc.save(f"Cv_Soaint_{nombre}.docx")
    
    st.header("Insights mas valiosos:")  
    st.write(Insights)  
    
    user_question = st.text_input("Haz una pregunta sobre tu PDF:")

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-4')
        # llm = ChatOpenAI(model_name='gpt-4')

        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)
    
    
    
    
    