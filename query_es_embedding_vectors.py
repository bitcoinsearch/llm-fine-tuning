import os
from dotenv import load_dotenv
import warnings
import traceback
from loguru import logger
from sentence_transformers import SentenceTransformer
from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_INDEX
from src.utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

# logs automatically rotate log file
os.makedirs("logs", exist_ok=True)
logger.add(f"logs/query_es_embedding_vectors.py.log", rotation="23:59")

# define embedding model
model = SentenceTransformer('intfloat/e5-large-v2')

# ES_INDEX = "btc-test-1210"

if __name__ == "__main__":
    elastic_search = ElasticSearchClient(es_cloud_id=ES_CLOUD_ID, es_username=ES_USERNAME,
                                         es_password=ES_PASSWORD)

    try:
        question = 'How are transactions in discarded forks merged back into the blockchain?'

        results = elastic_search.compute_similar_docs_with_cosine_similarity(
            es_index=ES_INDEX,
            model=model,
            field_name='summary_vector',
            question=question,
            top_k=3
        )

        logger.info(f"Top n results: \n{results}")

    except Exception as ex:
        logger.error(f"Error occurred: {ex} \n{traceback.format_exc()}")
