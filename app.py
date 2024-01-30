from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

def get_pdf_text(pdf_docs):
    pdf_reader = PdfReader(pdf_docs)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text    


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding = embeddings)
    return vectorstore


def main():
    load_dotenv()
    os.getenv("OPENAI_API_KEY")
    # st.set_page_config(page_title="Some questions please.. :)")
    st.header("Ask your PDF :)")
    
    # st.subheader("Ask a question about your document")
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF", type="pdf")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text 
                    
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks 
                    
                chunks = get_text_chunks(raw_text)

                # create  vector store

                st.session_state.vectorstore = get_vectorstore(chunks)

    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        with st.spinner("Wait bruh..."):
            docs = st.session_state.vectorstore.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = docs, question=user_question)
            st.write(response)

    # upload file
    # pdf = st.file_uploader("Upload your PDF", type="pdf")

    
    # if pdf is not None:
    #     pdf_reader = PdfReader(pdf)
    #     text = ""
    #     # extract the text if file is uploaded
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()

    #     # split the text into chunks
    #     text_splitter = CharacterTextSplitter(
    #         separator="\n",
    #         chunk_size=1000,
    #         chunk_overlap=200,
    #         length_function=len
    #     )

    #     chunks = text_splitter.split_text(text)

    #     # create embeddings
    #     embeddings = OpenAIEmbeddings()
    #     vectorstore = FAISS.from_texts(chunks, embeddings)
        
    #     user_question = st.text_input("Ask a question about your PDF :")
    #     if user_question:
    #         docs = vectorstore.similarity_search(user_question)

    #         llm = OpenAI()
    #         chain = load_qa_chain(llm, chain_type="stuff")
    #         response = chain.run(input_documents = docs, question=user_question)

    #         st.write(response)


            
        


if __name__ == "__main__":
    main()