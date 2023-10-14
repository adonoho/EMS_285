#!/usr/bin/env python3

import time
import numpy as np
import pandas as pd
from pandas import DataFrame

from EMS.manager import do_on_cluster
from dask.distributed import Client, LocalCluster
import logging

logging.basicConfig(level=logging.INFO)


# Function that generates data with noise; will use again in later homeworks
def generate_data(nrow: int, ncol: int, seed: int = 0) -> tuple:

    # Set seed
    np.random.seed(seed)

    # Create length-n vector u with element equal to (-1)^i/sqrt(n)
    u = np.array([(-1)**i/np.sqrt(nrow) for i in range(nrow)])
    v = np.array([(-1)**(i+1)/np.sqrt(ncol) for i in range(ncol)])

    # Generate signal
    signal = 3 * np.outer(u,v)

    # noise matrix of normal(0,1)
    noise = np.random.normal(0,1,(nrow,ncol))/np.sqrt(nrow*ncol)

    # observations matrix
    X = signal + noise

    return X, u, v, signal  # return data


def experiment(*, nrow: int, ncol: int, seed: int) -> DataFrame:
    # Begin saving runtime
    start_time = time.time()

    # Generate data
    # seed = 285
    # nrow = 1000
    # ncol = 1000
    X, u_true, v_true, signal_true = generate_data(nrow, ncol, seed=seed)

    print('Data Generated')

    # Analyze the data using SVD
    U, S, Vh = np.linalg.svd(X)

    # Using first singular vector of U and V to estimate signal
    u_est = U[:,0]
    v_est = Vh[0,:]

    # Calculate estimate of signal
    signal_est = S[0] * np.outer(u_est,v_est)

    # Calculate alignment between u_est and u_true
    u_align = np.inner(u_est,u_true)

    # Calculate alignment between v_est and v_true
    v_align = np.inner(v_est,v_true)

    # Calculate distance between signal_est and signal_true
    signal_error = np.linalg.norm(signal_est-signal_true)/np.sqrt(nrow*ncol)

    # Print results to text file
    print("nrow = ", nrow)
    print("ncol = ", ncol)
    print("u_alignment = ", u_align)
    print("v_alignment = ", v_align)
    print("signal_error = ", signal_error)

    # Save u_est, v_est, u_true, v_true in a CSV file with an index column
    df = pd.DataFrame({'nrow': nrow, 'ncol': ncol, 'seed': seed,  # P, Parameters
                       "u_est": u_est, "v_est": v_est, "u_true": u_true, "v_true": v_true})  # W, Observables
    # df.to_csv("hw2data.csv", index_label="index")

    # Print runtime
    print("--- %s seconds ---" % (time.time() - start_time))
    return df



def test_experiment() -> dict:
    exp = dict(table_name='ems_285_hw2',
               params=[{
                   'nrow': [1000],
                   'ncol': [1000],
                   'seed': [285]
               }])
    return exp


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, experiment, client)


def do_test():
    exp = test_experiment()
    print(exp)
    pass
    df = experiment(nrow=1000, ncol=1000, seed=285)
    print(df)


if __name__ == "__main__":
    # do_local_experiment()
    do_test()
