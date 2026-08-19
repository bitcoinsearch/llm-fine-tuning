import os
import re
# import shutil
import traceback
import pandas as pd
from dotenv import load_dotenv
from dateutil.parser import parse
from loguru import logger
import warnings
from datetime import datetime
import csv
from src.elasticsearch_utils import ElasticSearchClient
warnings.filterwarnings("ignore")
load_dotenv()


month_dict = {
    1: "Jan", 2: "Feb", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"
}

# pre-compile the patterns
regex_url = re.compile(r'http\S+|www\S+|https\S+')
regex_non_alpha = re.compile(r'[^A-Za-z0-9 ]+')
regex_spaces = re.compile(r'\s+')


def clean_text(text):
    text = regex_url.sub('', text.lower())  # remove urls and convert to lower case
    text = regex_non_alpha.sub('', text)  # remove non-alphanumeric characters
    text = regex_spaces.sub(' ', text).strip()  # replace whitespace sequences with a single space
    return text


def clean_title(xml_name):
    special_characters = ['/', ':', '@', '#', '$', '*', '&', '<', '>', '\\', '?']
    xml_name = re.sub(r'[^A-Za-z0-9]+', '-', xml_name)
    for sc in special_characters:
        xml_name = xml_name.replace(sc, "-")
    return xml_name


def get_id(id):
    return str(id).split("-")[-1]


def get_base_directory(url):
    if "bitcoin-dev" in url or "bitcoindev" in url:
        directory = "bitcoin-dev"
    elif "lightning-dev" in url:
        directory = "lightning-dev"
    elif "delvingbitcoin" in url:
        directory = "delvingbitcoin"
    else:
        directory = "others"
    return directory


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def normalize_text(s, sep_token=" \n "):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.replace("#", "")
    s = s.strip()
    return s


def preprocess_email(email_body):
    email_body = email_body.split("-------------- next part --------------")[0]
    email_lines = email_body.split('\n')
    temp_ = []
    for line in email_lines:
        if line.startswith("On"):
            line = line.replace("-", " ")
            x = re.sub('\d', ' ', line)
            if is_date(x, fuzzy=True):
                continue
            if line.endswith("> wrote:"):
                continue
        if line.endswith("> wrote:"):
            continue
        if line.startswith("Le "):
            continue
        if line.endswith("?crit :"):
            continue
        if re.match(r'\d{4}-\d{2}-\d{2}', line):
            continue
        if line.startswith("From:") or line.strip().startswith("To") or line.strip().startswith("permalink"):
            continue
        if line.startswith("Sent with Proton Mail"):
            continue
        if line and not line.startswith('>'):
            if line.startswith('-- ') or line.startswith('[') or line.startswith('_____'):
                continue
            temp_.append(line)
    email_string = "\n".join(temp_)
    normalized_email_string = normalize_text(email_string)
    if len(normalized_email_string) == 0:
        return email_body
    else:
        return normalized_email_string


def merge_multiple_csv_from_dir(dir_path, csv_save_path):
    # get a list of all the csv files
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logger.info(f"Total csv files found: {len(csv_files)}")

    # initialize a list to store dataframes
    dfs = []

    # loop through the list of csv files
    for file in csv_files:
        try:
            # read the csv file
            curr_df = pd.read_csv(dir_path + file)
            logger.info(f"individual file shape bfr:{curr_df.shape}")

            # drop duplicate rows based on the 'source_id' column
            curr_df.drop_duplicates(subset='source_id', keep='first', inplace=True)
            logger.info(f"individual file shape aft:{curr_df.shape}")

            # append the current df to the list
            dfs.append(curr_df)
        except Exception as ex:
            logger.error(f"Error Occurred: {ex}")
            logger.error(traceback.format_exc())

    # concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"main file shape bfr:{df.shape}")

    # drop duplicates from the final dataframe
    df.drop_duplicates(subset='source_id', keep='first', inplace=True)
    logger.info(f"main file shape aft:{df.shape}")

    # write the final dataframe to a csv file
    df.to_csv(f"{csv_save_path}", index=False)
    logger.success(f"file saved at path: {csv_save_path}")


def convert_to_dataframe(docs_list, save_csv=False):
    dict_list = []
    for i in docs_list:
        src = i.get("_source")
        i_data = {
            "index": str(i.get("_index", "")).strip(),
            "_id": str(i.get("_id", "")).strip(),
            "title": str(src.get("title", "")).strip(),
            "transcript_by": str(src.get("transcript_by", "")).strip(),
            "domain": str(src.get("domain", "")).strip(),
            "created_at": str(src.get("created_at", "")).strip(),
            "body_type": str(src.get("body_type", "")).strip(),
            "url": str(src.get("url", "")).strip()
        }
        dict_list.append(i_data)

    df = pd.DataFrame(dict_list)
    if save_csv:
        df.to_csv("data_list.csv", index=False)
    return df


def get_duplicated_docs_ids(df):
    logger.info(f"Shape: {df.shape}")
    cols = ['index', 'title', 'transcript_by', 'created_at', 'domain', 'body_type']
    df_grouped = df.groupby(cols)

    ids_to_keep = []
    for _, dfx in df_grouped:

        # if only one instance keep it
        if dfx.shape[0] == 1:
            this_id = dfx['_id'].values[0]
            ids_to_keep.append(this_id)

        else:
            urls_ = list(set(dfx['url'].to_list()))

            # check if all urls are same, if yes then keep the last one
            if len(urls_) == 1:
                this_id = dfx['_id'].values[-1]
                ids_to_keep.append(this_id)

            # if all urls are different
            else:
                # check if last route of each urls are same, if yes move forward with this operation
                last_route_urls = list(set([i.split("/")[-1] for i in urls_]))

                if len(last_route_urls) == 1:
                    base_url = "https://btctranscripts.com/bitcoin-core-dev-tech/"
                    pattern = re.compile("^https://btctranscripts.com/bitcoin-core-dev-tech/[0-9]{4}-[0-9]{2}/.*")

                    temp_urls = []
                    for _, r in dfx.iterrows():
                        this_id = r['_id']
                        this_url = r['url']

                        if this_url not in temp_urls:
                            if base_url in this_url:
                                if pattern.match(this_url):
                                    ids_to_keep.append(this_id)
                            else:
                                ids_to_keep.append(this_id)

                            temp_urls.append(this_url)

                # if last route of each urls are different, take all of them
                else:
                    temp_urls = []
                    for _, r in dfx.iterrows():
                        this_id = r['_id']
                        this_url = r['url']
                        if this_url not in temp_urls:
                            ids_to_keep.append(this_id)
                            temp_urls.append(this_url)

    total_ids = set(df['_id'])
    ids_to_keep_set = set(ids_to_keep)
    ids_to_drop = list(total_ids - ids_to_keep_set)

    df_to_keep = df[df['_id'].isin(ids_to_keep_set)]
    df_to_drop = df[df['_id'].isin(ids_to_drop)]

    duplicates = df_to_keep[df_to_keep.duplicated('url', keep=False)]
    df_to_drop = pd.concat([df_to_drop, duplicates])
    df_to_keep.drop(duplicates.index, inplace=True)

    ids_to_drop = df_to_drop['_id'].to_list()
    logger.info(f"Total: {len(total_ids)}, Keeping: {df_to_keep.shape[0]}, Dropping: {len(ids_to_drop)}")

    return ids_to_drop


def log_csv(file_name, url=None, inserted=0, updated=0, no_changes=0, folder_path="daily_logs",
            error="False", error_log="---"):
    date = datetime.utcnow().strftime("%d_%m_%Y")
    month_year = datetime.utcnow().strftime("%Y_%m")
    time = datetime.utcnow().strftime("%H:%M:%S")

    log_folder_path = os.path.join(folder_path, month_year)
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    csv_file_path = os.path.join(log_folder_path, f'{date}_logs.csv')
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(
                ['Date', 'Time', 'File name', 'URL', 'Inserted records', 'Updated records', 'No changes records',
                 'Total records', 'Error', 'Error log'])

        total_docs = 0

        if isinstance(url, str):
            total_docs = ElasticSearchClient().get_domain_counts(index_name=os.getenv('INDEX'), domain=url)

        elif isinstance(url, list):
            for i in url:
                t_docs = ElasticSearchClient().get_domain_counts(index_name=os.getenv('INDEX'), domain=i)
                total_docs += t_docs

        writer.writerow([date, time, file_name, url, inserted, updated, no_changes, total_docs, error, error_log])
    logger.success("CSV Update Successfully :)")
