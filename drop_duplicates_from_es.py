from loguru import logger
from dotenv import load_dotenv
import warnings
import tqdm
import traceback

from src.config import ES_INDEX
from src.utils import convert_to_dataframe, get_duplicated_docs_ids
from src.elasticsearch_utils import ElasticSearchClient

warnings.filterwarnings("ignore")
load_dotenv()


if __name__ == "__main__":

    elastic_search = ElasticSearchClient()

    dev_urls = [
        "https://btctranscripts.com/"
    ]

    for dev_url in dev_urls:
        docs_list = elastic_search.docs_list = elastic_search.extract_data_from_es(
            es_index=ES_INDEX, url=dev_url, start_date_str=None, current_date_str=None
        )
        logger.success(f"TOTAL THREADS RECEIVED FOR {dev_url}: {len(docs_list)}")

        if docs_list:
            df = convert_to_dataframe(docs_list)
            ids_to_drop = get_duplicated_docs_ids(df)

            if ids_to_drop:
                for idx, this_id in enumerate(tqdm.tqdm(ids_to_drop)):
                    try:
                        res = elastic_search.es_client.delete(
                            index=ES_INDEX,
                            id=this_id
                        )
                        logger.info(res)
                    except Exception as ex:
                        logger.error(f"Error deleting the doc: {traceback.format_exc()}")
        else:
            logger.info(f"NO THREADS RECEIVED FOR {dev_url}")

    logger.info("Process Complete.")
