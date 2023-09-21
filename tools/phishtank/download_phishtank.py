import urllib.request
import json
import os
import bz2
import datetime
import time


class PhishTank:

    API_KEY = os.environ.get('PHISHTANK_API_KEY')
    if not API_KEY:
        raise ValueError('API key not found in environment variables.')
    URL = f"http://data.phishtank.com/data/{API_KEY}/online-valid.json.bz2"
    LATEST_DIR = "data/"
    LATEST = LATEST_DIR + "latest.json.bz2"

    def __init__(self):
        self.bz2c = bz2.BZ2Compressor()
        if not os.path.exists(self.LATEST_DIR):
            os.makedirs(self.LATEST_DIR)

    def fetch_phishtank_data(self):
        try:
            request = urllib.request.Request(self.URL)
            with urllib.request.urlopen(request) as f:
                data = f.read()
            return data
        except Exception as e:
            print(f"Failed to fetch PhishTank data: {str(e)}")
            return None

    def decompress_data(self, data):
        try:
            data_bin = bz2.BZ2Decompressor().decompress(data)
            data_str = data_bin.decode("utf-8")
            data_list = json.loads(data_str)
            return data_list
        except Exception as e:
            print(f"Failed to decompress PhishTank data: {str(e)}")
            return []

    def save_data(self, data, file_path):
        try:
            with open(file_path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"Failed to save data to {file_path}: {str(e)}")

    def run(self):
        phishtank_data = self.fetch_phishtank_data()

        if phishtank_data:
            phish_ids = [l["phish_id"] for l in self.decompress_data(phishtank_data)]
            print(len(phish_ids))

            if not os.path.exists(self.LATEST):
                self.save_data(phishtank_data, self.LATEST)
                now = datetime.datetime.now().strftime('%Y%m%d%H%M')
                self.save_data(phishtank_data, f"{self.LATEST_DIR}{now}.json.bz2")
                return

            with open(self.LATEST, "rb") as f:
                data_latest_bin = bz2.BZ2Decompressor().decompress(f.read())
                data_latest_str = data_latest_bin.decode("utf-8")
                data_latest_list = json.loads(data_latest_str)
                phish_latest_ids = [l["phish_id"] for l in data_latest_list]

            diff_phish_ids = set(phish_ids) - set(phish_latest_ids)

            diff_data_list = [d for d in self.decompress_data(phishtank_data) if d["phish_id"] in diff_phish_ids]

            diff_data_str = json.dumps(diff_data_list)
            self.bz2c.compress(diff_data_str.encode())
            diff_data_bin_bz2 = self.bz2c.flush()
            now = datetime.datetime.now().strftime('%Y%m%d%H%M')
            self.save_data(diff_data_bin_bz2, f"{self.LATEST_DIR}{now}.json.bz2")

            self.save_data(phishtank_data, self.LATEST)


if __name__ == "__main__":
    phishtank = PhishTank()

    while True:
        phishtank.run()
        time.sleep(60 * 60)  # Fetch data every 1 hour
