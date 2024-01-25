import os
from dotenv import load_dotenv

# Assuming you have a .env file with environment variables
load_dotenv()  # Now the environment variables can be accessed through os.environ

# We need to replace <DIR> with the actual path to the directory containing the documents.
# This path could be stored in an environment variable or hardcoded into the script.
documents_directory = os.getenv('DOCUMENTS_DIR', 'path/to/documents')  # Replace 'path/to/documents' with your actual directory path

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load the documents from the specified directory
documents = SimpleDirectoryReader(documents_directory).load_data()
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index and perform a search
query_engine = index.as_query_engine()
response = query_engine.query("Who was on the plane?")
print(response)