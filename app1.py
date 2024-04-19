# Using Cohere embedding,Cohere Chat model and FAISS Vectorstore,
# Using prompt template to generate the question for the user

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv, find_dotenv
from html_templates import css,bot_template,user_template
from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunk = text_splitter.split_text(raw_text)
    return chunk

def get_vectorstore(text_chunks):
    _ = load_dotenv(find_dotenv())
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0", cohere_api_key= os.environ['COHERE_API_KEY']
    )
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return db


def get_conversation_chain(vectorstore):
    llm = Cohere(model="command-r-plus", temperature=0, cohere_api_key = os.environ['COHERE_API_KEY'])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question, template_prompt, context):
    template = template_prompt.replace("{question}", user_question)
    user_question = template
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.write(st.session_state.chat_history)
    st.write(st.session_state.conversation.memory)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("<span style='color:blue'>YOU</span>" + user_template.replace("{{MSG}}", st.session_state.user_questions[int(i/2)]), unsafe_allow_html=True)
        else:
            st.write("<span style='color:green'>CHATBOT</span>" + bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your data", page_icon=":book:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your data :book:")
    user_question = st.text_input("Ask a question about your data")

    # Build prompt
    template_prompt = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use 3 sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" 
    at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    
    context = ""
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get the text from the PDFs
                raw_text = get_pdf_text(pdf_docs)

                # get chunks of text
                text_chunks = get_text_chunks(raw_text)

                # Embedding
                vectorstore = get_vectorstore(text_chunks)
                
                # thêm vào st.session_state để mỗi lần chat biến này ko thay đổi
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # Cái này lưu lại các câu hỏi của user để đem show,ko lấy từ history đc vì lúc này có thêm prompt
                st.session_state.user_questions = []
                context = "\n\n".join(text_chunks)  # concatenate the text chunks i
    

    if user_question is not None and user_question != "":
        st.session_state.user_questions.append(user_question)
        handle_user_input(user_question, template_prompt, context)


if __name__ == "__main__":
    main()