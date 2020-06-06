import numpy as np
import pandas as pd
import pickle
import os
import argparse

import surprise
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--lr', '-l', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--latent', '-v', default=10, type=int,
                    help='Latent vector size')
parser.add_argument('--seed', '-s', default=2017, type=int,
                    help='Random seed for numpy')

# Set hyperparameters and parameters
args = parser.parse_args()
lr = args.lr
latent_size = parser.latent
random_seed = args.seed


# Prepare training/validation sets
def prepare_train_test_data(TPS_DIR, filename, reader):
    """
    Prepare data for matrix factorization (Surprise lib)

    Args:
    -------
    TPS_DIR: directory of the data folder
    filename: file name of the dataset
    reader: the Reader object used to parse a file containing ratings.

    Outputs:
    -------
    : dataset
    : dataset size
    """

    pkl_file = open(os.path.join(TPS_DIR, filename), 'rb')
    data = pickle.load(pkl_file)
    data = np.array(data)
    pkl_file.close()

    uid, iid, reuid, reiid, yrating = zip(*data)
    uid = np.array(uid)
    iid = np.array(iid)
    yrating = np.array(yrating)
    df = pd.DataFrame({'userID': uid[:, 0], 'itemID': iid[:, 0], 'rating': yrating[:, 0]})
    dataset = Dataset.load_from_df(df_train[['userID', 'itemID', 'rating']], reader)
    dataset = dataset.build_full_trainset()

    return dataset, len(uid)


print('Loading files...')

TPS_DIR = 'data/toyandgame'
reader = Reader(rating_scale=(1., 5.))

data_tr, num_tr = prepare_train_test_data(TPS_DIR, 'toyandgame.train', reader)
data_va, num_va = prepare_train_test_data(TPS_DIR, 'toyandgame.valid', reader)
data_va = data_va.build_testset()

print('Training set: {} samples and validation set: {} samples prepared'.format(num_tr, num_va))


# Model training
algo = surprise.prediction_algorithms.matrix_factorization.NMF(n_factors=latent_size, lr_bu=lr, lr_bi=lr,
                                                               random_state=random_seed)
algo.fit(data_tr)
pred_train = algo.test(data_tr.build_testset())
pred_valid = algo.test(data_va)
print('Train RMSE:', accuracy.rmse(pred_train))
print('Valid RMSE:', accuracy.rmse(pred_valid))
