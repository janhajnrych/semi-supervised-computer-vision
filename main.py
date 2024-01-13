from dataset import ImageTripletDataset, DatasetSplit
import os
from training import Session
from net import EmbeddingNetwork
import torch
import argparse
import pandas as pd
from strategies import TripletTrainer, TripletValidator, TripletTester
import logging
import sys


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


def get_args():
    parser = argparse.ArgumentParser(
        prog="Image Embedding Neural Network Training",
        description="Performs self-supervised training of  neural network",
        epilog="Happy training!",
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        default=16,
        type=int,
        help="The number of epochs to training the model.",
    )
    parser.add_argument(
        "-i", "--input_size", default=96, type=int, help="The size of the input images"
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="data",
        type=str,
        help="The path to the dataset directory",
    )
    parser.add_argument(
        "-e",
        "--embedding_size",
        default=32,
        type=int,
        help="The size of the embedding vector.",
    )
    parser.add_argument(
        "-b", "--batch_size", default=16, type=int, help="The batch size for training"
    )
    parser.add_argument(
        "-c",
        "--use_cuda",
        default=True,
        action="store_true",
        help="A flag indicating whether to use CUDA for training",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="results",
        help="The directory to save the output files",
    )
    parser.add_argument(
        "-m", "--model_filename", default="model.pth", help="Final model save filename"
    )
    return parser.parse_args()


def get_device(args):
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def create_datasets(input_size, dataset_path, strategy: Session.SessionStrategy):
    full_dataset = ImageTripletDataset()
    full_dataset.load_from_dir(dataset_path)
    dataset_split = DatasetSplit.create_train_val_test_split(
        full_dataset, test_fraction=0.15, val_fraction=0.1
    )
    dataset_split.train.transform = strategy.trainer.get_transform(input_size)
    dataset_split.val.transform = strategy.validator.get_transform(input_size)
    dataset_split.test.transform = strategy.tester.get_transform(input_size)
    return dataset_split


def train(session, output_dir, n_epochs):
    csv_path = os.path.join(output_dir, "losses.csv")
    report_period = 20
    for epoch in session.train(n_epochs=n_epochs):
        if epoch % report_period == 0:
            df = pd.DataFrame(session.history)
            df.to_csv(csv_path, index=False)
    return session


if __name__ == "__main__":
    args = get_args()
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    set_logger(os.path.join(out_dir, "log.txt"))
    model = EmbeddingNetwork(emb_dim=args.embedding_size, input_size=args.input_size)
    device = get_device(args)
    logging.debug(f"device = {device}")
    model = model.to(device)
    strategy = Session.SessionStrategy(
        trainer=TripletTrainer(model, batch_size=args.batch_size),
        validator=TripletValidator(model, batch_size=args.batch_size),
        tester=TripletTester(model, batch_size=args.batch_size),
    )
    dataset_dir = os.path.join(out_dir, "dataset")
    datasets = create_datasets(args.input_size, args.dataset_path, strategy)
    session = Session(datasets=datasets, strategy=strategy)
    session = train(session, out_dir, args.n_epochs)
    test_result = session.test()
    logging.info(f" ---> test result:  ROC_AUC = {test_result['roc_auc']}")
    session.save_model(os.path.join(out_dir, args.model_filename))
