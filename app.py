# Using Cohere embedding,Cohere Chat model and FAISS Vectorstore,
# Using prompt template to generate the question for the user

import streamlit as st
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
import os
from dotenv import load_dotenv, find_dotenv
from html_templates import css,bot_template,user_template


def get_vectorstore(pdf_directory, embedding):

    # Load the PDFs
    loader = DirectoryLoader('./pdfs', glob="*.pdf", show_progress=False)
    docs = loader.load()
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    
    return vectorstore


def cohere():
    load_dotenv()
    COHERE_API_KEY = os.environ["COHERE_API_KEY"]
    llm = ChatCohere(model="command-r", cohere_api_key=COHERE_API_KEY)
    embedding = CohereEmbeddings(model="embed-multilingual-light-v3.0", cohere_api_key=COHERE_API_KEY)
    return llm, embedding

def create_history_retriever(llm,retriever):
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def create_question_answer_chain(llm):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return question_answer_chain

def handle_user_input(user_question):
    out = st.session_state.conversational_rag_chain.invoke(
                {"input": user_question},
                config={
                    "configurable": {"session_id": "thanhstar"}
                },  # constructs a key "thanhstar" in `store`.
            )

    # for i, message in enumerate(out["chat_history"]):
    #     st.write(i)
    #     st.write(message.content)
    # print(out1)
    # print(out2)
    # print(out3)

    for i, message in enumerate(out["chat_history"]):
        if i % 2 == 0:
            st.write("<span style='color:blue'>YOU</span>" + user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write("<span style='color:green'>CHATBOT</span>" + bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    st.write("<span style='color:blue'>YOU</span>" + user_template.replace("{{MSG}}", out["input"]), unsafe_allow_html=True)
    st.write("<span style='color:green'>CHATBOT</span>" + bot_template.replace("{{MSG}}", out["answer"]), unsafe_allow_html=True)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def main():
    st.set_page_config(page_title="Chat with your data", page_icon=":book:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your data :book:")
    user_question = st.text_input("Ask a question about your data")


    llm, embedding = cohere()


    ### Statefully manage chat history ###
    if "store" not in st.session_state:
        st.session_state.store = {}
    
    with st.sidebar:
        st.subheader("Your documents")

        # Create a folder to store the uploaded PDFs
        pdfs_folder = 'pdfs'
        if not os.path.exists(pdfs_folder):
            os.makedirs(pdfs_folder)

        # Upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        # Remove trc khi lưu file mới vì lỡ như còn file khác trong folder
        for filename in os.listdir(pdfs_folder):
            file_path = os.path.join(pdfs_folder, filename)
            os.remove(file_path)

        # Save the uploaded PDFs to the folder
        if pdf_docs:
            for i, pdf_doc in enumerate(pdf_docs):
                with open(os.path.join(pdfs_folder, f"uploaded_pdf_{i+1}.pdf"), 'wb') as f:
                    f.write(pdf_doc.getbuffer())
            st.success(f"{len(pdf_docs)} PDFs uploaded successfully!")

        if st.button("Process"):
            with st.spinner("Processing..."):
                    
                vectorstore = get_vectorstore('./pdfs', embedding)
                retriever = vectorstore.as_retriever()

                ### Contextualize question ###
                history_aware_retriever = create_history_retriever(llm, retriever)

                # ### Answer question ###
                question_answer_chain = create_question_answer_chain(llm)

                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                                                                        rag_chain,
                                                                        get_session_history,
                                                                        input_messages_key="input",
                                                                        history_messages_key="chat_history",
                                                                        output_messages_key="answer",
                                                                    )
                    
    if user_question is not None and user_question != "":
        handle_user_input(user_question) 



if __name__ == "__main__":
    main()