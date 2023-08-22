# train sentence-BERT model with triplet data
# save the model to ./sbert_stair


import argparse
from sentence_transformers import SentenceTransformer, models, InputExample
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader
from sentence_transformers.datasets import SentencesDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging


class TrainTriplet:
    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.output_path = args.output_path
        self.brands = args.brands
        self.tlds = args.tlds

    def load_data(self, filepath):
        """
        load triplet data from a tsv file which is separated by tab e.g., "anchor_domain\tpositive_domain\tnegative_domain"
        data split into train, dev, test data are saved to ./sample_data/triplet_train.tsv, ./sample_data/triplet_dev.tsv, ./sample_data/triplet_test.tsv

        Parameters:
        ----------
        filepath: str
            path to a tsv file
        """
        self.logger.info("load data from %s" % filepath)

        triplets = []
        # read triplet data from a tsv file which is separated by tab e.g., "anchor_domain\tpositive_domain\tnegative_domain"
        with open(filepath) as f:
            for l in f:
                l = l.rstrip("\n")
                triplet = l.split("\t")
                triplets.append(triplet)

        self.train, dev_test = train_test_split(
            triplets, train_size=0.8, random_state=1)
        self.dev, self.test = train_test_split(
            dev_test, train_size=0.5, random_state=2)

        def _to_tsv(fname, triplet_data):
            with open(fname, "w") as f:
                lines = ["%s\t%s\t%s" % (data[0], data[1], data[2])
                         for data in triplet_data]
                f.write("\n".join(lines)+"\n")

        _to_tsv("./sample_data/triplet_train.tsv", self.train)
        _to_tsv("./sample_data/triplet_dev.tsv", self.dev)
        _to_tsv("./sample_data/triplet_test.tsv", self.test)

    def train_model(self, output_path=""):
        """
        train sentence-BERT model with triplet data

        Parameters:
        ----------
        output_path: str
            path to save the model

        """
        self.logger.info(
            "Start training sentence-BERT model with triplet data")

        BATCH_SIZE = 16
        NUM_EPOCHS = 15
        EVAL_STEPS = 100
        train_len = len(self.train)
        WARMUP_STEPS = int(train_len // BATCH_SIZE * 0.1)

        if output_path == "":
            output_path = self.output_path

        word_embedding_model = models.Transformer(
            'sentence-transformers/all-mpnet-base-v2')

        # load brand name to add tokens
        with open(self.brands) as f:
            brands = f.read().splitlines()
        with open(self.tlds) as f:
            tlds = f.read().splitlines()

        tokens = tlds+brands
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(
            len(word_embedding_model.tokenizer))

        pooling = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        model = SentenceTransformer(modules=[word_embedding_model, pooling])

        triplet_reader = TripletReader(".")
        train = []
        for i, (a, p, n) in enumerate(self.train):
            train.append(InputExample(guid=str(i), texts=[a, p, n]))
        train_data = SentencesDataset(train, model=model)
        train_dataloader = DataLoader(
            train_data, shuffle=True, batch_size=BATCH_SIZE)

        train_loss = TripletLoss(
            model=model, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=NUM_EPOCHS,
                  evaluation_steps=EVAL_STEPS,
                  warmup_steps=WARMUP_STEPS,
                  output_path=output_path
                  )

        self.logger.info(
            "Finish training sentence-BERT model with triplet data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train sentence-BERT model with triplet data')
    parser.add_argument('--triplet', type=str, required=True,
                        help='path to a tsv file which contains triplet data')
    parser.add_argument('--output_path', type=str,
                        default='./sbert_stair', help='path to save the model')
    parser.add_argument(
        '--tlds', type=str, default='./sample_data/tlds.txt', help='path to a tlds file')
    parser.add_argument(
        '--brands', type=str, default='./sample_data/brands.txt', help='path to a brands file')

    args = parser.parse_args()

    trainer = TrainTriplet(args)
    trainer.load_data(args.triplet)
    trainer.train_model()
