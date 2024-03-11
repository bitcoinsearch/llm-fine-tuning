import tiktoken
import os
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()

TOKENIZER = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = SentenceTransformer('intfloat/e5-large-v2')  # intfloat/e5-large-v2, intfloat/e5-base-v2
CHAT_COMPLETION_MODEL = "gpt-3.5-turbo"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY")

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

ES_CLOUD_ID = os.getenv("ES_CLOUD_ID")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX = os.getenv("ES_INDEX")
ES_DATA_FETCH_SIZE = 10000  # No. of data to fetch and save from elastic-search
