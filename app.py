from flask import Flask, render_template, request
from langchain_community.llms import Cohere
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

TEXT_FILE_PATH = "GraphicEra.txt"

def text_file_to_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def text_splitter(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, retriever):
    cohere_api_key = os.getenv('COHERE_API_KEY')
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=cohere_api_key)

    prompt_template = """
        Answer the question as precisely as possible using the provided context. You may also provide additional links if relevant.
        If the answer is not contained in the context, say "Sorry, the answer is not available in the context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """


    prompt = PromptTemplate.from_template(template=prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        raw_text = text_file_to_text(TEXT_FILE_PATH)
        text_chunks = text_splitter(raw_text)
        vectorstore = get_vector_store(text_chunks)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        answer = generate_answer(question, retriever)
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True, port=5000)