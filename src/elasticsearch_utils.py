import time
from elasticsearch import Elasticsearch
from loguru import logger
from src.config import ES_CLOUD_ID, ES_USERNAME, ES_PASSWORD, ES_DATA_FETCH_SIZE


class ElasticSearchClient:
    def __init__(self,
                 es_cloud_id=ES_CLOUD_ID,
                 es_username=ES_USERNAME,
                 es_password=ES_PASSWORD,
                 es_data_fetch_size=ES_DATA_FETCH_SIZE
                 ) -> None:
        self._es_cloud_id = es_cloud_id
        self._es_username = es_username
        self._es_password = es_password
        self._es_data_fetch_size = es_data_fetch_size
        self._es_client = Elasticsearch(
            cloud_id=self._es_cloud_id,
            http_auth=(self._es_username, self._es_password),
        )

    @property
    def es_client(self):
        return self._es_client

    def get_domain_query(self, url):
        if isinstance(url, list):
            domain_query = {"terms": {"domain.keyword": url}}
        else:
            domain_query = {"term": {"domain.keyword": url}}
        return domain_query

    def handle_multiple_field_names(self, field_name):
        if isinstance(field_name, list):
            field_not_exists_query = [{"exists": {"field": str(field)}} for field in field_name]
        else:
            field_not_exists_query = {
                "exists": {
                    "field": str(field_name)
                }
            }
        return field_not_exists_query

    def compute_similar_docs_with_cosine_similarity(self, es_index, model, field_name, question: str, top_k: int = 3):
        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")
            logger.info(f"Querying ES index: {question}")
            question_embedding = model.encode(question, normalize_embeddings=True)
            script_query = {
                "bool": {
                    "must": [
                        {"script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"cosineSimilarity(params.query_vector, '{field_name}') + 1.0",
                                "params": {"query_vector": question_embedding.tolist()}
                            }
                        }}, {"exists": {"field": field_name}}
                    ]
                }
            }
            response = self._es_client.search(
                index=es_index,
                body={
                    "size": top_k,
                    "query": script_query,
                    "_source": {"includes": ["title", "summary", "body"]}
                }
            )
            logger.info(f"Response: {response}")
            return {
                "question": question,
                "matches": [
                    {
                        "title": hit['_source'].get('title'),
                        "summary": hit['_source'].get('summary') if hit['_source'].get('summary') else hit[
                            '_source'].get('body'),
                        "score": hit.get('_score')
                    } for hit in response['hits']['hits']
                ]
            }
        else:
            logger.info('Could not connect to Elasticsearch')
            return {
                "question": question,
                "matches": []
            }

    def fetch_data_for_empty_field(self, es_index, field_name, url=None, start_date_str=None, current_date_str=None):
        logger.info(f"fetching the data based on empty '{field_name}' ... ")
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")

            domain_query = self.get_domain_query(url)
            not_exists_query = self.handle_multiple_field_names(field_name)

            if url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                },
                                domain_query
                            ],
                            "must_not": not_exists_query
                        }
                    }
                }
            elif url and not start_date_str and not current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                domain_query
                            ],
                            "must_not": not_exists_query
                        }
                    }
                }
            elif not url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                }
                            ],
                            "must_not": not_exists_query
                        }
                    }
                }
            else:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must_not": not_exists_query
                        }
                    }
                }

            # Initialize the scroll
            scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
                                                     scroll='5m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"dumping '{es_index}' data in json...")
            while len(results) > 0:
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"dumping '{es_index}' data completed in {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return output_list

    def extract_data_from_es(self, es_index, url=None, start_date_str=None, current_date_str=None):
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")

            domain_query = self.get_domain_query(url)

            if url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                },
                                domain_query
                            ]
                        }
                    }
                }
            elif url and not start_date_str and not current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                domain_query
                            ]
                        }
                    }
                }
            elif not url and start_date_str and current_date_str:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "created_at": {
                                            "gte": f"{start_date_str}T00:00:00.000Z",
                                            "lte": f"{current_date_str}T23:59:59.999Z"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            else:
                logger.info(f"Url: {url}, Start Date: {start_date_str}, Current Date: {current_date_str}")
                query = {
                    "query": {
                        "bool": {
                        }
                    }
                }

            # Initialize the scroll
            scroll_response = self._es_client.search(index=es_index, body=query, size=self._es_data_fetch_size,
                                                     scroll='5m')
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"started dumping of {es_index} data in json...")
            while len(results) > 0:
                # Save the current batch of results
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.warning('Could not connect to Elasticsearch')
            return output_list

    def fetch_docs_with_keywords(self, es_index, url, keyword):
        output_list = []
        start_time = time.time()

        if self._es_client.ping():
            logger.info("connected to the ElasticSearch")

            domain_query = self.get_domain_query(url)

            query = {
                "min_score": 1,
                "query": {
                    "bool": {
                        "must": [
                            domain_query,
                            {
                                "match_phrase": {
                                    "summary": str(keyword)
                                }
                            },
                            {
                                "match_phrase": {
                                    "body": str(keyword)
                                }
                            },
                            {
                                "match": {
                                    "summary": {
                                        "query": str(keyword),
                                        "minimum_should_match": "95%",
                                        # "operator": "and"
                                    }
                                }
                            },
                            {
                                "match": {
                                    "body": {
                                        "query": str(keyword),
                                        "minimum_should_match": "95%",
                                        # "operator": "and"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Initialize the scroll
            scroll_response = self._es_client.search(
                index=es_index,
                body=query,
                size=self._es_data_fetch_size,
                scroll='5m'
            )
            scroll_id = scroll_response['_scroll_id']
            results = scroll_response['hits']['hits']

            # Dump the documents into the json file
            logger.info(f"Starting dumping of {es_index} data in json...")
            while len(results) > 0:
                for result in results:
                    output_list.append(result)

                # Fetch the next batch of results
                scroll_response = self._es_client.scroll(scroll_id=scroll_id, scroll='5m')
                scroll_id = scroll_response['_scroll_id']
                results = scroll_response['hits']['hits']

            logger.info(
                f"Dumping of {es_index} data in json has completed and has taken {time.time() - start_time:.2f} seconds.")

            return output_list
        else:
            logger.info('Could not connect to Elasticsearch')
            return None

    def add_vector_field(self, es_index, field_name):
        res = self._es_client.indices.put_mapping(
            index=es_index,
            body={
                "properties": {
                    f"{field_name}": {
                        "type": "dense_vector",
                        "dims": 1024
                    }
                }
            }
        )
        logger.info(f"Field updated: {res}")
