# IT HelpDesk QnA 


import streamlit as st
import pandas as pd
import re
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.document_transformers import LongContextReorder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from fpdf import FPDF

# Azure OpenAI settings
api_key = "783973291a7c4a74a1120133309860c0"  # Replace with your Azure API key
azure_endpoint = "https://theswedes.openai.azure.com/"
api_version = "2024-02-01"


# CSV to PDF Conversion Function
def csv_to_pdf(dataframe, output_filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        for col in dataframe.columns:
            pdf.multi_cell(0, 10, f"{col}: {row[col]}", border=0)
        pdf.ln(10)  # Add space after each row

    pdf.output(output_filename)
    
# Streamlit UI
st.title("IT HelpDesk QnA")

# File uploader for CSV
uploaded_csv = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_csv is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_csv, encoding='ISO-8859-1')
        
        # Convert CSV to PDF
        output_pdf = "output.pdf"
        csv_to_pdf(df, output_pdf)
        st.write("CSV converted to PDF successfully!")
        
        loader = PyMuPDFLoader(output_pdf)
        data = loader.load()

        st.write("Document Ingested Successfully.")

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        texts = text_splitter.split_documents(data)

        # Initialize embeddings
        embeddings_model = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            deployment="TextEmbeddingLarge",
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key
        )
        
        # Process the documents in batches
        # batch_size = 10  # You can adjust the batch size based on your requirements
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i:i + batch_size]
        #                 # Store each batch in Chroma DB
        #     db = Chroma.from_documents(embedding=embeddings_model, documents=batch_texts)
        #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ embeddings ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        from langchain_postgres.vectorstores import PGVector
        connection = "postgresql+psycopg://citus:Admin123@c-it-helpdisk-qna.edgw3kbkcr5txn.postgres.cosmos.azure.com:5432/citus?sslmode=require"
        collection_name = "citus"
        db = PGVector(embeddings=embeddings_model, collection_name=collection_name, connection=connection, use_jsonb=True)
        
        db.delete() 

        # db.add_documents(enhanced_documents, ids=[doc.metadata["serial_number"] for doc in enhanced_documents])
        
        # Process the documents in batches
        batch_size = 100  # You can adjust the batch size based on your requirements
        for i in range(0, len(texts), batch_size): 
                batch_texts = texts[i:i + batch_size]
                db.add_documents(batch_texts)
        
        # # Create a Chroma database
        # db = Chroma.from_documents(embedding=embeddings_model, documents=texts)        
        # Ask user for problem input as a message
        user_problem = st.text_input("Describe the problem you want to find a resolution:")

        retriever = db.as_retriever(search_kwargs={"k": 10})        
    
        if st.button("Get Resolution"):
            # Create a retriever

            query = f"What is the Resolution for the below-mentioned Problem: {user_problem}"

            # Retrieve relevant documents
            docs = retriever.invoke(query)

            # Reorder the documents
            reordering = LongContextReorder()
            reordered_docs = reordering.transform_documents(docs)

            # Initialize the LLM
            llm = AzureChatOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key,
                model="gpt-4o-mini",
                base_url=None,
                azure_deployment="GPT-4o-mini"
            )
            llm.validate_base_url = False

            # Create the prompt template
            prompt_template = """
            Given these texts:
            -----
            {context}
            -----
            Please answer the following question in detail:
            {query}
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "query"],
            )

            # Create and invoke the chain
            chain = create_stuff_documents_chain(llm, prompt)
            response = chain.invoke({"context": reordered_docs, "query": query})

            # Display the response
            st.write("## Resolution:")
            st.write(response)
    except Exception as e:
        st.error(f"Error processing file: {e}")
