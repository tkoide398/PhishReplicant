# list of files or directories of phishing ti
phishingti_path_list = [
    "./sample_data/phishingti/*", 
]

# list of files or directories of input domains
input_domains_path_list = [
    "./sample_data/input_domains/*",
]

# path to SBERT model
model_path = ""

# log level
log_level = "INFO"

# eps for clustering
eps = 0.04
# min_samples for clustering
min_samples = 3

# threshold for detector
threshold = 0.96
# number of similar domains is 2 or more
similar_domains_threshold = 2
