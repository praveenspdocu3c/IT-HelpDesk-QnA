# IT HelpDesk QnA 


import streamlit as st
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import re

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import BaseExtractor

from llama_index.core.postprocessor import LLMRerank
from llama_index.core import Settings
import logging
import sys
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Streamlit application
st.title("IT HelpDesk QnA")

# Azure OpenAI settings
api_key = "6e9d4795bb89425286669b5952afe2fe"
azure_endpoint = "https://danielingitaraj-gpt4turbo.openai.azure.com/"
api_version = "2024-02-01" 

llm = AzureOpenAI(
    model="gpt-4",
    deployment_name="GPT4Turbo",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="text-embedding-3-large",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Create the sentence window node parser with default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Base node parser is a sentence splitter
text_splitter = SentenceSplitter()

# Configure settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter


# Preprocessing function to replace Unicode values with respective characters
def preprocess_text(text):
    # Add more replacements as needed
    replacements = {
        "&ldquo;": "“",
        "&rdquo;": "”",
        "&lsquo;": "‘",
        "&rsquo;": "’",
        "&quot;": '"',
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&mdash;": "—",
        "&ndash;": "–",
        "&hellip;": "…",
        "&copy;": "©",
        "&reg;": "®",
        "&euro;": "€",
        "&cent;": "¢",
        "&pound;": "£",
        "&yen;": "¥",
        "&deg;": "°",
        # Add more HTML entity replacements as needed
    }
    
    for unicode_val, char in replacements.items():
        text = re.sub(unicode_val, char, text)
    
    return text


# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Saving uploaded file
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Loading document
    documents1 = SimpleDirectoryReader(
        input_files=["uploaded_file.pdf"]
    ).load_data()

    # Preprocess the text in the document
    for doc in documents1:
        doc.text = preprocess_text(doc.text)  # Apply preprocessing

    class CustomExtractor(BaseExtractor):
        async def aextract(self, nodes):
            keywords = [
                "Resolution",
                "Problem:",
                "PROBLEM/ISSUE:",
            ]

            metadata_list = []
            for node in nodes:
                custom_metadata = {}
                for keyword in keywords:
                    if keyword in node.metadata:
                        custom_metadata[keyword] = node.metadata[keyword]
                metadata_list.append({"custom": custom_metadata})

            return metadata_list

    extractors = [
        CustomExtractor()
    ]
    transformations = [text_splitter] + extractors

    pipeline = IngestionPipeline(transformations=transformations)
    documents = pipeline.run(documents=documents1)

    st.write("Document Ingested Successfully.")

    # Extract nodes
    nodes = node_parser.get_nodes_from_documents(documents)
    nodes1 = node_parser.get_nodes_from_documents(documents1)

    # Build sentence index
    sentence_index = VectorStoreIndex(nodes1 + nodes)

    # MetadataReplacementPostProcessor
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=30,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # Ask user for problem input as a message
    user_problem = st.text_area("Describe the problem you want to find a resolution:")

    if st.button("Get Resolution"):
        # Query engine with user problem input
        window_response = query_engine.query(
            f"""What is the Resolution for the below mentioned Problem:
                Problem: {user_problem}"""
        )

        # Display response
        st.markdown("### Query Result:")
        st.write(window_response)

        # Show the original sentence retrieved and the window
        window = window_response.source_nodes[0].node.metadata["window"]
        st.markdown("### Retrieved Window:")
        st.write(window)
