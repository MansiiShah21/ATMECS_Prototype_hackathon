import sys
import os
import streamlit as st
import faiss
import certifi
# print(certifi.where())
# print("Python executable:", sys.executable)
# print("Python version:", sys.version)
# print("Virtual environment:", os.getenv('VIRTUAL_ENV'))
# print("open api key:",os.environ.get("OPENAI_API_KEY"))


from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader,DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA


def load_llm():
    global llm
    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)
    file_path="/Users/mansishah/Desktop/Prsnl/genai_study_buddy/docs/AMZN-Q2-2024-Earnings-Release.pdf"
    loader=PyPDFLoader(file_path)
    summary_pages=loader.load()
    return llm
    # print(len(summary_pages))

def summarize_doc(text,llm):
    print(llm.get_num_tokens(text))
    print("creating summary")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200
    )
    chunks=text_splitter.create_documents([text])
    print(len(chunks))
    ######
    chunks_template="""You are a bot that generates a detailed summary for the give part of financial reports that should find key details from the given part like fincnaical numbres, next plans, 
    and if statement from ceo and other heads is present include that.
    Report- {text}"""

    map_prompt_template=PromptTemplate(
        template=chunks_template,
        INPUT_VARIABLES=['text']

    )

    final_combine_prompt="""Provide a final summarized one pager report with the key heading including all important financial numbers,stamenet from ceos and other heads
    and what are the next plans and the conclusion
    Report- {text}"""

    final_combine_prompt_template=PromptTemplate(
        template=final_combine_prompt,
        INPUT_VARIABLES=['text']

    )

    chain = load_summarize_chain(llm,verbose=False,map_prompt=map_prompt_template,combine_prompt=final_combine_prompt_template, chain_type="map_reduce")
    # print(chain.run(chunks))
    response=chain.run(chunks)
    print(response)
    return response

def qna_doc(file_path):
    global vectorstore
    loader=PyPDFLoader(file_path)
    pages=loader.load()
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=3000,
    chunk_overlap=400
    )
    print(text_splitter)
    x = text_splitter.split_documents(pages)
    print("length of chunks")
    print(len(x))
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_kwargs = {"device": "cuda"}

    # embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    # embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # dimensions=1024
    )

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(x, embeddings)
    print(vectorstore)
    return vectorstore

def qna(query):
    # retriever = vectorstore.as_retriever(k=4)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    result = qa({"query": query})
    print(result)
    return result




        


UPLOAD_DIRECTORY = "/Users/mansishah/Desktop/Prsnl/genai_study_buddy/docs"

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


def main():
    global llm
    st.title("Future-Ready Decision Support: Harnessing LLMs for Business Success!!")
    if 'llm' not in st.session_state:
        print('yes')
        st.session_state.llm = load_llm() 

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

    # Move the selectbox to the sidebar
    st.sidebar.header("What insights would ou like to get!!")
    # st.sidebar.write("You can select from 1-Insights from Financial Reporting and 2-Insights from Online Review Data")
    option = st.sidebar.selectbox("Select an option:", 
                                ["Select an option", 
                                    "Insights from Financial Reporting", 
                                    "Insights from Online Review Data"])

    # Check which option was selected
    if option == "Insights from Financial Reporting":
        st.sidebar.write("Choose what you want from the report:")
        summary_option = st.sidebar.radio("Select option ", ["","Summary", "Q&A","Comparative analysis"])
        
        st.sidebar.subheader("Upload your Financial Reporting PDF")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        if summary_option == "Summary":
                llm=load_llm()
                summary=summarize_doc(text,llm)
                # print(st.session_state.llm)
                st.header("Generated Summary.......")
                st.write(summary)
                # Add your summary processing logic here
        elif summary_option == "Q&A":
            # vector=qna_doc(file_path)
            # qna(query)
            query = st.text_input("Enter your query:")
            print("query==================",query)

            if query:
                # Show loading spinner while processing the query
                with st.spinner("Processing your query..."):
                    print(file_path)
                    llm=load_llm()
                    vector=qna_doc(file_path)
                    response = qna(query)
                
                # Display the response in the center
                st.write("#### Response to Your Query")
                st.write(response['result'])


        if uploaded_file is not None:
        # Process the uploaded file
            st.sidebar.write("File uploaded successfully!")
            # st.sidebar.write(f"You uploaded: {uploaded_file.name}")
            file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
            print(file_path)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(file_path)
            summary_pages=loader.load()
            print(len(summary_pages))
            text=''
            print(st.session_state.llm )
            for doc in summary_pages:
                text+=doc.page_content

            st.sidebar.write("Choose what you want from the report:")
            summary_option = st.sidebar.radio("Select option ", ["","Summary", "Q&A"])

            if summary_option == "Summary":
                llm=load_llm()
                summary=summarize_doc(text,llm)
                # print(st.session_state.llm)
                st.header("Generated Summary.......")
                st.write(summary)
                # Add your summary processing logic here
            elif summary_option == "Q&A":
                # vector=qna_doc(file_path)
                # qna(query)
                query = st.text_input("Enter your query:")
                print("query==================",query)

                if query:
                    # Show loading spinner while processing the query
                    with st.spinner("Processing your query..."):
                        print(file_path)
                        llm=load_llm()
                        vector=qna_doc(file_path)
                        response = qna(query)
                    
                    # Display the response in the center
                    st.write("#### Response to Your Query")
                    st.write(response['result'])

if __name__ == "__main__":
    main()