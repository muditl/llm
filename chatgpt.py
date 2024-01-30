import os
import sys
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"

query = sys.argv[1]

loader = TextLoader("data.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))