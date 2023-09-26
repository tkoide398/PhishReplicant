from datetime import datetime, timedelta, date, timezone
from pymongo import MongoClient
import os

# MongoDB connection
client = MongoClient('localhost', 27017)
db = client['ctlog']
collection = db['domain']


def ensure_data_directory_exists():
    if not os.path.exists('data'):
        os.makedirs('data')


def get_today_filename():
    today = date.today()
    return os.path.join('data', today.strftime("%Y%m%d.txt"))


def extract_and_write_domains_to_file():
    ensure_data_directory_exists()
    today_filename = get_today_filename()
    with open(today_filename, 'w') as file:
        now = datetime.now(timezone.utc)
        twenty_four_hours_ago = now - timedelta(hours=24)
        cursor = collection.find({"datetime": {"$gte": twenty_four_hours_ago, "$lt": now}})
        for document in cursor:
            domain = document['domain']
            file.write(domain + '\n')


if __name__ == "__main__":
    extract_and_write_domains_to_file()
