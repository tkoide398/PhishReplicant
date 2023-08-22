# PhishReplicant

## Introduction
PhishReplicant is a system designed to identify phishing domain names created using domain squatting techniques to impersonate legitimate domain names.

## Installation
To install PhishReplicant, you will need Python 3.6 or later.
Install the required dependencies by running the following command:
1. Clone the PhishReplicant repository:
   ```
   git clone {this repository}
   cd phishreplicant/
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


## Quick Start
Try PhishReplicant with the sample data provided in the sample_data directory.
```
python main.py --output_dir ./
cat result.txt
```
PhishReplicant will analyze the sample data and provide a list of similar domain names along with their similarity scores.
```
{"domain": "www.amazoon.co.uk5.top", "similar_domains": ["www.amazoon.co.uk3.top", "www.amazoon.co.uk2.top", "www.amazoon.co.uk1.top"], "similar_scores": [0.992138683795929, 0.9893498420715332, 0.9890196323394775]}
{"domain": "www.amazoon.co.uk4.top", "similar_domains": ["www.amazoon.co.uk3.top", "www.amazoon.co.uk2.top", "www.amazoon.co.uk1.top"], "similar_scores": [0.9856407642364502, 0.9840406179428101, 0.9834299087524414]}
```

## Usage
To use PhishReplicant, you will need to provide input data in the form of phishing domain names and newly registered domain names. You can collect phishing domain names from sources such as PhishTank, and extract newly registered domain names from CT logs and other sources. Once you have input data, you will need to edit the config.py file to specify the file paths.

1. Prepare input data.
    - *Phishing domain names* collected from phishing threat intelligence such as PhishTank, etc. (e.g., sample_data/phishingti/sample.txt)
    - *Newly registered domain names* extracted from CT logs and other sources (e.g., sample_data/input_domains/sample.txt)
2. Edit config.py to add file paths of the above input data.
    - phishingti_path_list: list of file paths of phishing domain names
    - new_domain_path_list: list of file paths of newly registered domain names
    - model_path: file path of the fine-tuned SBERT model

    Here is an example of config.py:
    ```
    phishingti_path_list = ['/path/to/phishingti/sample.txt']
    new_domain_path_list = ['/path/to/input_domains/sample.txt']
    model_path = '/path/to/fine-tuned-SBERT-model'
    ```

3. Run the following command:
   ```
   python main.py --output_dir ./
   ```


## Fine-tuning (Optional)

 To improve the accuracy of PhishReplicant, you can fine-tune the SBERT model using triplet data. Triplet data is a set of examples that consist of an anchor domain name, a positive example (i.e., a similar domain name), and a negative example (i.e., a dissimilar domain name). You can also provide additional brand names and top-level domains (TLDs) to improve the model's performance. You will need to create files called brand_names.txt and tlds.txt containing the brand names and TLDs, respectively. Once you have your triplet data and (optionally) your brand names and TLDs, you can fine-tune the SBERT model by running the following command:

Fine-tune the SBERT model to improve the accuracy.
1. Prepare the triplet data for fine-tuning (e.g., sample_data/triplet.tsv).
2. Prepare additional brand names (e.g., sample_data/brands.txt) and tlds (e.g., sample_data/tlds.txt) if you want to use them for the fine-tuning.
    - brand_names.txt: list of brand names
    - tlds.txt: list of tlds
3. Run the following command.
    ```
    python train.py --triplet /path/to/triplet-data.tsv
    ```

## Reference
Please consider citing our paper:
```
@inproceedings{koide23acsac,
  author       = {Takashi Koide and
                  Naoki Fukushi and
                  Hiroki Nakano and
                  Daiki Chiba},
  title        = {PhishReplicant: A Language Model-based Approach to Detect Generated Squatting Domain Names},
  booktitle    = {Annual Computer Security Applications Conference,
                  {ACSAC} 2023, Austin, TX, USA, December 4-8, 2023},
  publisher    = {{ACM}},
  year         = {2023},
}
```
