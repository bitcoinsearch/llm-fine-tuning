import traceback
from datetime import datetime
import pytz
import xml.etree.ElementTree as ET
import os
from loguru import logger
from src.utils import get_id, clean_title, month_dict, get_base_directory


class XMLReader:

    def get_xml_summary(self, data, dev_name):
        number = get_id(data["_source"]["id"])
        title = data["_source"]["title"]
        xml_name = clean_title(title)
        published_at = datetime.strptime(data['_source']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        published_at = pytz.UTC.localize(published_at)
        month_name = month_dict[int(published_at.month)]
        str_month_year = f"{month_name}_{int(published_at.year)}"
        current_directory = os.getcwd()
        directory = get_base_directory(dev_name)
        file_path = f"static/{directory}/{str_month_year}/{number}_{xml_name}.xml"
        full_path = os.path.join(current_directory, file_path)

        try:
            if os.path.exists(full_path):
                namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
                tree = ET.parse(full_path)
                root = tree.getroot()
                summ_list = root.findall(".//atom:entry/atom:summary", namespaces)
                if summ_list:
                    summ = "\n".join([summ.text for summ in summ_list])
                    return summ
                else:
                    logger.warning(f"No summary found: {full_path}")
                    return None
            else:
                # logger.warning(f"No xml file found: {full_path}")
                return None
        except Exception as e:
            logger.error(
                f"Error: {e} \n{traceback.format_exc()} \n\nFILE PATH: {full_path}\nDOC ID: {data['_source']['id']}")
            return None
