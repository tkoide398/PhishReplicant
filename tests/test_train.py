import argparse
import pytest
import os
from train import TrainTriplet

@pytest.fixture
def train_triplet():
    return TrainTriplet(args=argparse.Namespace(
        triplet="./sample_data/triplet.tsv",
        output_path="./sbert_stair",
        tlds="./sample_data/tlds.txt",
        brands="./sample_data/brands.txt"
    ))

def test_load_data(train_triplet):
    # show current directory
    print(os.getcwd())
    
    train_triplet.load_data("./sample_data/triplet.tsv")
    assert os.path.exists("./sample_data/triplet_train.tsv")
    assert os.path.exists("./sample_data/triplet_dev.tsv")
    assert os.path.exists("./sample_data/triplet_test.tsv")

def test_train_model(train_triplet):
    train_triplet.load_data("./sample_data/triplet.tsv")
    train_triplet.train_model(output_path="./test_sbert_stair")
    assert os.path.exists("./test_sbert_stair")
