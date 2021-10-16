#!/usr/bin/env python
import argparse
from functools import partial
import os
import pickle
from pathlib import Path
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from naive_bayes import NaiveBayes, NaiveBayesLaplace

def load_dataset(filename):
    with open(Path(".", "data", filename), "rb") as f:
        return pickle.load(f)


def main():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


def main_laplace():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    model = NaiveBayesLaplace(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error with Laplace: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error with Laplace: {err_valid:.3f}")

if __name__ == "__main__":
    main()
    main_laplace()
