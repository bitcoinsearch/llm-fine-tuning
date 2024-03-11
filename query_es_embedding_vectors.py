import os
from dotenv import load_dotenv
import warnings
import traceback
from loguru import logger
from src.config import ES_INDEX, EMBEDDING_MODEL
from src.elasticsearch_utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()

if __name__ == "__main__":

    # logs automatically rotate log file
    os.makedirs("logs", exist_ok=True)
    logger.add(f"logs/query_es_embedding_vectors.py.log", rotation="23:59")

    elastic_search = ElasticSearchClient()

    try:
        question = 'Bitcoin'

        results = elastic_search.compute_similar_docs_with_cosine_similarity(
            es_index=ES_INDEX,
            model=EMBEDDING_MODEL,
            field_name='summary_vector_embeddings',
            question=question,
            top_k=3
        )

        for res in results['matches']:
            score = res.get('score')
            summary = res.get('summary')
            logger.info(f'Score: {score} | Text: {summary}')

    except Exception as ex:
        logger.error(f"Error occurred: {str(ex)}\n{traceback.format_exc()}")
