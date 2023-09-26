import datetime
import certstream
from pymongo import MongoClient

client = MongoClient('mongodb://ctlog-mongo-1/', 27017)
db = client['ctlog']
collection = db['domain']
collection.create_index("domain")
collection.create_index("datetime")


def save_domain_to_mongodb(domain):
    now = datetime.datetime.now(datetime.timezone.utc)
    item = {
        "domain": domain,
        "datetime": now
    }
    collection.insert_one(item)


def bulk_insert_domains_to_mongodb(domains):
    existing_domains = collection.find({"domain": {"$in": domains}})
    existing_domains = set([d['domain'] for d in existing_domains])
    new_domains = list(set(domains) - existing_domains)
    if new_domains:
        items = [{"domain": domain, "datetime": datetime.datetime.now(datetime.timezone.utc)} for domain in new_domains]
        print(items)
        collection.insert_many(items)


def print_callback(message, context, skip_heartbeats=True):
    domains = []
    if message['message_type'] == "certificate_update":
        all_domains = message['data']['leaf_cert']['all_domains']
        if not all_domains:
            return
        domain = all_domains[0]

        # Add domain to list
        domains.append(domain)

        # Bulk insert domains to MongoDB if there are 1000 domains
        if len(domains) == 1000:
            bulk_insert_domains_to_mongodb(domains)
            domains = []

    # Bulk insert domains to MongoDB if there are less than 1000 domains
    if domains:
        bulk_insert_domains_to_mongodb(domains)

# Listen for events
certstream.listen_for_events(print_callback, url='wss://certstream.calidog.io/')