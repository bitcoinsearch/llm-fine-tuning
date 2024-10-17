# LLM Fine-Tuning

This repository was created to explore fine-tuning models using OpenAI and LLama V2. While that functionality still exists in the code, it is no longer the primary focus. Instead, the repo has shifted towards **automating the generation of topic modeling and vector embeddings for data** collected by the [scraper](https://github.com/bitcoinsearch/scraper) and stored in an Elasticsearch index.

The original fine-tuning scripts remain in place for potential future use, but the current effort centers on leveraging document summaries for efficient topic extraction and embedding generation.

## Libraries Used

The current workflows involve:

- **Cron Jobs** for automation.
- **Elasticsearch** for document storage and querying.
- **GPT** (via OpenAI API) for generating topic models.
- **SentenceTransformer** for generating vector embeddings.

The legacy fine-tuning scripts still support the following libraries:

- OpenAI
- LLama V2

## Current Functionality

The repository now focuses on automating topic modeling and vector embedding based on document summaries. Below are the details of the current functionality:

### Overview

Utilizing data collected by the [scraper](https://github.com/bitcoinsearch/scraper) and stored in an Elasticsearch index, this repository runs several nightly cron jobs to automate topic modeling and vector embeddings.

### Current Cron Jobs

1. **Daily [Topic Modeling Generation](.github/workflows/generate_topics_from_elasticsearch_cron_job.yml)** ([source](generate_topic_modeling_csv.py))  
   - Queries Elasticsearch for documents without topic modeling across specified sources. It generates primary and secondary topics for each document using GPT and a predefined list of [Bitcoin-related topics](./btc_topics.csv). The results are stored in [CSV files](./gpt_output/).

2. **Daily [Push Topic Modeling to Elasticsearch](.github/workflows/push_topics_to_elasticsearch_cron_job.yml)** ([source](push_topic_modeling_to_es.py))  
   - Reads the generated topic modeling CSV files and updates the corresponding documents in Elasticsearch with their primary and secondary topics.

3. **Daily [Update Vector Embeddings](.github/workflows/update_vector_embeddings_cron_job.yml)** ([source](update_vector_embedding_to_es.py))  
   - Queries Elasticsearch for documents without vector embeddings across specified sources. It generates vector embeddings using the document's title and summary (or body if summary is unavailable) with SentenceTransformer (`intfloat/e5-large-v2`). These embeddings are then updated in Elasticsearch.

   - **Note**: This uses an open-source model, so there's no cost for generating embeddings. The embedding size limit is 1024.

## Legacy Fine-Tuning Code

For those interested, the original fine-tuning scripts can still be found under their respective directories (`openai-finetune` and `llamav2-finetune`). These are not actively maintained but remain available for future experimentation.

## Getting Started

To use this repository, you can clone it to your local machine or download it as a zip file. Once you've done that, refer to the scripts or cron jobs relevant to your use case, and follow the instructions in the respective files.
