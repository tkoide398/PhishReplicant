import argparse
import gc
import json
import logging
import os
import pickle
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from glob import glob
from pprint import pformat
from uuid import UUID

import faiss
import numpy as np
import tld
import validators
from publicsuffixlist import PublicSuffixList
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from tld import get_fld, get_tld

import config

class Converter:
    """
    convert domain to vector
    """

    def __init__(self, model_path=None):
        if model_path is None:
            if config.model_path == "":
                model_path = "sentence-transformers/all-mpnet-base-v2"
            else:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, config.model_path)
        self.sbert = SentenceTransformer(model_path)


    def encode(self, domains, show_progress_bar=True):
        vecs = self.sbert.encode(domains, show_progress_bar=show_progress_bar)
        return vecs

    def calc_similarity(self, domain1, domain2):
        vec1 = self.encode([domain1], show_progress_bar=False)[0]
        vec2 = self.encode([domain2], show_progress_bar=False)[0]
        # return cosine similarity
        sim = np.dot(vec1, vec2) / \
            (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return sim


@dataclass
class PRDetectorResult:
    domain: str
    similar_domains: list
    similar_scores: list


@dataclass
class PRClusterResult:
    label_domain_dic: dict
    label_filter_dic: dict
    domain_label_dic: dict


class PRDetector:
    def __init__(self, label_domain_dic={}, domain_label_dic={}, label_filter_dic={}, cluster_vec_dic={}, args=None):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.logger = logging.getLogger(__name__)
        self.psl = PublicSuffixList()
        self.threshold = config.threshold
        self.domain_label_dic = domain_label_dic
        self.label_filter_dic = label_filter_dic
        self.domain_vec_dic = {}
        self.label_domain_dic = label_domain_dic
        self.converter = Converter()

        self.cluster_vec_dic = cluster_vec_dic
        if domain_label_dic is {}:
            self.get_label_domain_dic()
        self.similar_domains_threshold = config.similar_domains_threshold
        if args is not None:
            self.domain_vec_dir = args.new_domain_vec_dir
            self.target_date = args.target_date

    def get_label_domain_dic(self):
        """
        reverse the dictionary of domain_label_dic to label_domain_dic
        """
        for domain, label in self.domain_label_dic.items():
            if label == -1:
                continue
            self.label_domain_dic.setdefault(label, []).append(domain)
        return self.label_domain_dic

    @staticmethod
    def get_common_part(string1, string2, max=100):
        """
        Get the common part of two strings from the end of the strings.
        """
        common_part = ""
        for i in range(min(len(string1), len(string2))):
            if i > max:
                break
            if string1[-i-1] != string2[-i-1]:
                break
            common_part = string1[-i-1] + common_part
        return common_part

    def dissimilar_except_common(self, prdetector_result: PRDetectorResult, freq_tlds, freq_flds):
        """

        If TLD or FLD is frequent, measure the similarity of the string excluding TLD or FLD.

        Parameters:
        -----------
        prdetector_result: PRDetectorResult object
        freq_tlds: list
            list of frequent tlds
        freq_flds: list
            list of frequent flds

        Returns:
        --------
        bool

        """
        domain = prdetector_result.domain
        similar_domains = prdetector_result.similar_domains
        d_tld = get_tld("http://"+prdetector_result.domain)
        d_fld = get_fld("http://"+prdetector_result.domain)
        sim_tld = get_tld("http://"+similar_domains[0])
        sim_fld = get_fld("http://"+similar_domains[0])
        if d_tld in freq_tlds or sim_tld in freq_tlds:
            common_part = self.get_common_part(domain, similar_domains[0])
        elif d_fld in freq_flds or sim_fld in freq_flds:
            common_part = self.get_common_part(domain, similar_domains[0])
        else:
            return False
        if common_part == "":
            return True
        self.logger.debug(domain, similar_domains[0], common_part)
        sim = self.converter.calc_similarity(
            domain[:-len(common_part)], similar_domains[0][:-len(common_part)])
        if sim < self.threshold:
            return True
        else:
            return False

    @staticmethod
    def _filter_out_domain(domain, filters):
        """
        return True if the domain should be filtered out
        """
        d_tld = get_tld("http://"+domain)
        d_fld = get_fld("http://"+domain)
        for fil in filters:
            filter_type = fil["type"]
            filter_val = fil["val"]

            if filter_type == "length" and len(domain) != filter_val:
                return True
            elif filter_type == "tld" and d_tld != filter_val:
                return True
            elif filter_type == "fld" and d_fld != filter_val:
                return True
        return False

    def filter(self, matched_domains, domain_list, D, I):
        """
        Parameters:
        -----------
        result: list of int
            list of label
        matched_domains: list of str
            list of matched domains
        label_list: list of int
            list of label
        domain_list: list of str   
            list of domains of clusters

        Returns:
        --------
        list of PRDetectorResult object
        """

        self.logger.info("Filtering out domains")

        return_result = []
        domain_set = set(list(domain_list))

        # get frequent tlds and flds from matched domains
        flds = [get_fld("http://"+d) for d in matched_domains]
        tlds = [get_tld("http://"+d) for d in matched_domains]
        freq_tlds = []
        freq_flds = []
        for d, c in Counter(tlds).most_common(100):
            spl = d.split(".")
            if len(spl) > 1 and c > 5:
                freq_tlds.append(d)
        for d, c in Counter(flds).most_common(100):
            if c > 5:  # fld
                freq_flds.append(d)

        for distances, indices, domain in list(zip(D, I, matched_domains)):
            # skip if the domain is already in the Phishing TI
            if domain in domain_set:
                continue
            # skip if the domain is IDN
            if "xn--" in domain:
                continue

            # get up to 5 similar domains and scores that exceed the threshold
            similar_domains = np.array(domain_list)[
                indices[distances > self.threshold]][:5]
            similar_scores = distances[distances > self.threshold][:5]
            similar_scores = [float(score) for score in similar_scores]
            similar_domains = [
                domain for domain in similar_domains if "xn--" not in domain] 
            similar_scores = [score for score, domain in zip(
                similar_scores, similar_domains) if "xn--" not in domain]

            # skip if the number of similar domains is less than threshold
            if len(similar_domains) < self.similar_domains_threshold:
                continue
            label = self.domain_label_dic[similar_domains[0]]
            try:
                filters = self.label_filter_dic[label]
            except KeyError:
                filters = []
            # skip if the domain should be filtered out
            if self._filter_out_domain(domain, filters):
                continue
            prdetector_result = PRDetectorResult(
                domain, similar_domains, similar_scores)
            # measure the similarity of domains excluding the TLD or FLD
            if self.dissimilar_except_common(prdetector_result, freq_tlds, freq_flds):
                continue
            return_result.append(prdetector_result)
        self.logger.info(
            "Number of detected domains: {}".format(len(return_result)))
        return return_result

    def load_domain_vec(self, domain_vec_dir=None, target_date=None):
        """
        load pre-calculated vectors

        Parameters:
        -----------
        domain_vec_dir: str
            directory path of pre-calculated vectors
        target_date: str
            target date of pre-calculated vectors
        """
        if domain_vec_dir is None:
            domain_vec_dir = self.domain_vec_dir
        if target_date is None:
            target_date = self.target_date
        self.logger.info("Load pre-calculated vectors")
        # each pkl file should be contained in the target_date dictionary
        self.domain_vec_dic = {}
        for filename in tqdm(glob(os.path.join(domain_vec_dir, target_date, '*.pkl'))):
            with open(filename, "rb") as f:
                self.domain_vec_dic.update(pickle.load(f))
        self.logger.info("Number of domains: {}".format(
            len(self.domain_vec_dic.keys())))

    def load_domain_and_calculate_vec(self, input_domains_path_list):
        """
        load domains from input_domains_path_list and calculate vectors

        Parameters:
        -----------
        input_domains_path_list: list of str
            list of paths to input domains

        """

        self.domain_vec_dic = {}
        self.input_domains_list = []
        for input_domains_path in input_domains_path_list:
            for filepath in glob(input_domains_path):
                with open(filepath) as f:
                    domains = [line.strip() for line in f]
                self.input_domains_list += PRCluster.exclude_domains(domains)

        if len(self.input_domains_list) == 0:
            self.logger.error("No domains in input_domains")
            sys.exit(1)
        self.logger.info("Start converting domain names to vectors")
        converter = Converter()
        vecs = converter.encode(self.input_domains_list)
        self.logger.info("Finish converting domain names to vectors")
        self.domain_vec_dic = {domain: vec for domain,
                               vec in zip(self.input_domains_list, vecs)}
        self.logger.info("Number of input_domains: {}".format(
            len(self.input_domains_list)))

    def run(self):
        """

        Returns:
        --------
        List of PRDetectorResult object
        """

        self.logger.info("Detecting generated squatting domains")

        domain_vec_array = np.array(list(self.domain_vec_dic.values()))
        domain_vec_key = np.array(list(self.domain_vec_dic.keys()))
        try:
            dim = domain_vec_array[0].shape[0]
        except IndexError:
            self.logger.error("No domain vector found")
            return []
        # cosine similarity index
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(domain_vec_array)
        index.add(domain_vec_array)
        num = 100

        domain_list = []
        for label, domains in self.label_domain_dic.items():
            domain_list += domains
        search_vecs = np.array([self.cluster_vec_dic[l] for l in domain_list])

        # normalize vectors of clustered domains
        faiss.normalize_L2(search_vecs)
        D, I = index.search(search_vecs, num)
        matched_domains = domain_vec_key[np.unique(I[D > self.threshold])]
        self.logger.info(
            "Number of matched domains: {}".format(len(matched_domains)))

        # search again with matched domains
        index = faiss.IndexFlatIP(dim)
        index.add(search_vecs)
        matched_domains_vec = np.array(
            [self.domain_vec_dic[d] for d in matched_domains])
        if len(matched_domains_vec) == 0:
            return []
        faiss.normalize_L2(matched_domains_vec)
        D, I = index.search(matched_domains_vec, num)
        filtered_matched_domains = self.filter(
            matched_domains, domain_list, D, I)
        self.logger.debug(
            pformat(f"Detected domains: {filtered_matched_domains}"))
        return filtered_matched_domains


class PRCluster:
    def __init__(self, args=None):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.eps = config.eps
        self.min_samples = config.min_samples
        self.phishingti_vec_dic = {}
        if args is not None:
            self.args = args
            if args.target_date is not None:
                self.target_date = args.target_date  # YYYYMMDD
                self.target_date_yesterday = (datetime.strptime(
                    self.target_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')

        self.logger = logging.getLogger(__name__)
        # get log_level from config
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, config.log_level.upper()))
        self.logger.addHandler(handler)
        self.logger.propagate = False

    @staticmethod
    def load_vec_from_file(domain_vec_file_list: list) -> dict:
        """
        load pre-calculated vectors from files
        """
        domain_vec_dic = {}
        for path in domain_vec_file_list:
            for filename in glob(path):
                # load pickled file
                with open(filename, "rb") as f:
                    domain_vec_dic = {**domain_vec_dic, **pickle.load(f)}
        return domain_vec_dic

    @staticmethod
    def has_uuid(string):
        """
        check if string has uuid
        """
        exp=r'\b\w{8}-\w{4}-\w{4}-\w{4}-\w{12}\b'
        uuid_regex = re.compile(exp)
        match = uuid_regex.search(string)
        if match is not None:
            uuid_str = match.group()
            try:
                uuid_obj = UUID(uuid_str)
            except ValueError:
                return False
            if str(uuid_obj) == str(uuid_str):
                return True
        else:
            return False

    @staticmethod
    def _filter_digit_domain(domain):
        """
        filter domain name with digits
        """
        domain_tld = get_tld("http://"+domain)
        domain_str = domain[:-len(domain_tld)-1]
        digits = sum(c.isdigit() for c in domain_str)
        non_digits = len(domain_str) - digits
        if digits < non_digits:
            return domain

    @staticmethod
    def has_ipv4(domain):
        """
        check if domain has ipv4 address

        Parameters:
        -----------
        domain: str
            domain name

        Returns:
        --------
        bool
        """
        exp = r"((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
        ip_regex = re.compile(exp)
        if re.search(ip_regex, domain):
            return True
        exp = r"((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\-){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
        ip_regex = re.compile(exp)
        if re.search(ip_regex, domain):
            return True
        return False

    @staticmethod
    def has_32hexa(domain):
        """
        check if domain has 32 hexa characters

        Parameters
        ----------
        domain : str
            domain name
        Returns
        -------
        bool
            True if domain has 32 hexa characters
        """
        exp = r"[0-9a-fA-F]{32}"
        ip_regex = re.compile(exp)
        if re.search(ip_regex, domain):
            return True
        else:
            return False

    @staticmethod
    def exclude_domains(domains: list) -> list:
        """
        exclude domains which are not valid or in the exclude list before clustering

        Parameters
        ----------
        domains : list
            list of domains in PhishingTi
        Returns
        -------
        result_domains : list
            list of domains
        """

        domains = list(set(domains))
        result_domains = []
        for domain in domains:
            if not validators.domain(domain):
                continue
            try:
                domain_tld = get_tld("http://"+domain)
                domain_fld = get_fld("http://"+domain)
            except tld.exceptions.TldDomainNotFound:
                continue
            if PRCluster.has_uuid(domain):
                continue
            if PRCluster.has_ipv4(domain):
                continue
            if PRCluster.has_32hexa(domain):
                continue
            if len(domain) > 60:
                continue

            # exclude domains which are too short
            domain_fldstrip = domain.rstrip(domain_tld).rstrip(".")
            if len(domain_fldstrip) < 7:
                continue

            if PRCluster._filter_digit_domain(domain) is None:
                continue

            # exclude domains whose subdomains are only numbers
            sub = domain_fld.rstrip(domain_tld).rstrip(".")
            if sub.isdecimal():
                len_fld = len(domain_fld)
                len_sub = len(domain.rstrip(domain_fld).rstrip("."))
                if len_sub < len_fld:
                    continue
            result_domains.append(domain)
        result_domains = list(
            set([domain for domain in result_domains if domain is not None]))
        return result_domains

    def cluster(self, vecs):
        """
        cluster domains by DBSCAN

        Parameters
        ----------
        vecs : list
            list of vectors

        Returns
        -------
        labels : list
            list of labels
        """
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                    metric="cosine", n_jobs=-1).fit(vecs)
        labels = db.labels_
        return labels

    def get_filter(self, label_domain_dic):
        """
        get filter for each cluster

        Parameters
        ----------
        label_domain_dic : dict
            key: label, value: list of domains

        Returns
        -------
        filters : dict
            key: label, value: list of filters
        """

        filters = {}
        for label, domains in label_domain_dic.items():
            if label == -1:
                continue
            length = list(set([len(d) for d in domains]))
            flds = list(set([get_fld("http://" + d) for d in domains]))
            tlds = list(set([get_tld("http://" + d) for d in domains]))
            filters[label] = []
            if len(length) == 1:
                filters[label].append({"type": "length", "val": length[0]})
            if len(flds) == 1:
                filters[label].append({"type": "fld", "val": flds[0]})
            elif len(tlds) == 1:
                filters[label].append({"type": "tld", "val": tlds[0]})
        return filters

    def load_phishingti_vec(self, phishingti_vec_dir):
        """
        Load pre-calculated vectors from pickle file in phishingti_vec_dir

        Parameters
        ----------
        phishingti_vec_dir : str
            Path to directory where phishingti vectors are stored
        """

        # load pre-calculated vectors
        self.logger.info("Load pre-calculated vectors")

        target_date_list = []
        for i in range(60):  # get phishingti up to 60 days ago from target_date
            target_date_list.append((datetime.strptime(
                self.target_date, "%Y%m%d") - timedelta(days=i)).strftime("%Y%m%d"))
        domain_vec_file_list = [os.path.join(
            phishingti_vec_dir, date+'*.pkl') for date in target_date_list]
        self.phishingti_vec_dic = self.load_vec_from_file(domain_vec_file_list)
        self.phishingti_list = self.exclude_domains(
            list(self.phishingti_vec_dic.keys()))
        # filter domains which are not in phishingti_list
        self.phishingti_vec_dic = {domain: vec for domain, vec in self.phishingti_vec_dic.items(
        ) if domain in self.phishingti_list}
        self.logger.info("Number of phishingti: {}".format(
            len(self.phishingti_list)))

    def load_phishingti_and_calculate_vec(self, phishingti_path_list):
        """
        Load phishingti from files in phishing_dir and calculate vectors

        Parameters
        ----------
        phishing_dir : str : path to directory of phishing TI

        """

        self.logger.info("Load phishingti and calculate vectors")

        self.phishingti_list = []
        for phishingti_path in phishingti_path_list:
            for filepath in glob(phishingti_path):
                with open(filepath) as f:
                    domains = [line.strip() for line in f]
                self.phishingti_list += self.exclude_domains(domains)

        if len(self.phishingti_list) == 0:
            self.logger.error("No domains in phishing TI")
            sys.exit(1)
        self.logger.info("Start converting domain names to vectors")
        converter = Converter()
        vecs = converter.encode(self.phishingti_list)
        self.logger.info("Finish converting domain names to vectors")
        self.phishingti_vec_dic = {
            domain: vec for domain, vec in zip(self.phishingti_list, vecs)}
        self.phishingti_list = self.exclude_domains(
            list(self.phishingti_vec_dic.keys()))
        # filter domains which are not in phishingti_list
        self.phishingti_vec_dic = {domain: vec for domain, vec in self.phishingti_vec_dic.items(
        ) if domain in self.phishingti_list}
        self.logger.info("Number of phishingti: {}".format(
            len(self.phishingti_list)))

    def run(self, phishing_vec_dir=None, phishingti_dir=None) -> tuple:
        """
        Cluster phishing threat intelligence

        Parameters
        ----------
        phishing_vec_dir : str
            Path to directory where phishing vectors are stored
        phishingti_dir : str
            Path to directory where phishing threat intelligence are stored

        Returns
        -------
        cluster_result : PRClusterResult
            Clustering result
        """

        self.logger.info("Start clustering phishing TI")

        if phishingti_dir is not None:
            self.load_phishingti_and_calculate_vec(phishingti_dir)

        if phishing_vec_dir is not None:
            self.load_phishingti_vec(phishing_vec_dir)

        # clustering
        phishingti_labels = self.cluster(
            list(self.phishingti_vec_dic.values()))
        # save clustering result
        domain_label_dic = {domain: label for domain, label in zip(
            list(self.phishingti_vec_dic.keys()), phishingti_labels) if label != -1}
        if domain_label_dic == {}:
            self.logger.error("No domains are clustered")
            sys.exit(1)
        # reverse dictionary
        label_domain_dic = {}
        for d, label in domain_label_dic.items():
            label_domain_dic.setdefault(label, []).append(d)
        self.logger.info(f"Number of clusters: {len(label_domain_dic)}")
        self.logger.info(
            f"Number of extracted domains by clustering: {len(domain_label_dic)}")
        # self.logger.debug(pformat(label_domain_dic, compact=True))

        # create filters
        # label_filter_dic = self.get_filter(phishingti_labels,list(self.phishingti_list))
        label_filter_dic = self.get_filter(label_domain_dic)
        # self.logger.debug(pformat(label_filter_dic, compact=True))

        self.logger.info(
            f"Number of filters: {len([label for label in label_filter_dic if label_filter_dic[label] is not [] ])}")
        self.logger.info("Finish clustering phishing TI")
        self.logger.debug(f"label_domain_dic: {pformat(domain_label_dic, compact=True)}")

        self.label_domain_dic = label_domain_dic
        self.label_filter_dic = label_filter_dic
        self.domain_label_dic = domain_label_dic
        return PRClusterResult(label_domain_dic, label_filter_dic, domain_label_dic)


def main(args):

    # Step 1:
    # load pre-calculated vectors
    prcluster = PRCluster(args)
    prcluster.load_phishingti_and_calculate_vec(
        phishingti_path_list=config.phishingti_path_list)

    # clustering phishing TI
    prcluster_result = prcluster.run()

    prdetector = PRDetector(label_domain_dic=prcluster_result.label_domain_dic, domain_label_dic=prcluster_result.domain_label_dic,
                            label_filter_dic=prcluster_result.label_filter_dic, cluster_vec_dic=prcluster.phishingti_vec_dic, args=args)
    prdetector.load_domain_and_calculate_vec(
        input_domains_path_list=config.input_domains_path_list)
    del prcluster
    gc.collect()
    prdetector_results = prdetector.run()

    # save detected result
    output_path = os.path.join(args.output_dir, 'result.txt')
    i = 1
    while os.path.isfile(output_path):
        output_path = f"result_{i}.txt"
        i += 1
    prdetector.logger.info(f"Save result to {output_path}")
    with open(output_path, 'w') as f:
        for prdetector_result in prdetector_results:
            f.write(json.dumps(asdict(prdetector_result))+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PhishReplicant")
    parser.add_argument('--new_domain_dir',
                        help="Directory path of newly registered domains")
    parser.add_argument('--new_domain_vec_dir',
                        help="Directory path of pre-calculated new domains")
    parser.add_argument('--phishingti_dir',
                        help="Directory path of phishing TI")
    parser.add_argument('--phishingti_vec_dir',
                        help="Directory path of pre-calculated phishing TI")
    parser.add_argument('--output_dir', default='./',
                        help="Directory path of output")
    parser.add_argument(
        '--target_date', help="Target date (YYYYMMDD) for clustering phishing TI")

    args = parser.parse_args()
    main(args)
