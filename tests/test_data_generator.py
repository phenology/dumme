import numpy as np
import pandas as pd
from dumme.utils import DummeDataGenerator


def test_create_cluster_sizes():
    clusters = DummeDataGenerator.create_cluster_sizes_array([1, 2, 3], 1)
    assert clusters == [1, 2, 3]

    clusters = DummeDataGenerator.create_cluster_sizes_array([30, 20, 7], 3)
    assert clusters == [30, 30, 30, 20, 20, 20, 7, 7, 7]


def test_generate_samples():
    dg = DummeDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
    df, _, _ = dg.generate_samples([1, 2, 3])

    # check columns
    assert df.columns.tolist() == ["y", "X_0", "X_1", "X_2", "Z", "cluster"]

    # check length
    assert len(df) == 6

    # check cluster sizes
    assert len(df[df["cluster"] == 0]) == 1
    assert len(df[df["cluster"] == 1]) == 2
    assert len(df[df["cluster"] == 2]) == 3


def test_generate_split_samples():
    dg = DummeDataGenerator(m=0.7, sigma_b=2.7, sigma_e=1)
    (
        train,
        test_known,
        test_new,
        training_ids,
        _,
        _,
    ) = dg.generate_split_samples([1, 3], [3, 2], [1, 1])

    # check all have same columns
    assert train.columns.tolist() == ["y", "X_0", "X_1", "X_2", "Z", "cluster"]
    assert test_known.columns.tolist() == ["y", "X_0", "X_1", "X_2", "Z", "cluster"]
    assert test_new.columns.tolist() == ["y", "X_0", "X_1", "X_2", "Z", "cluster"]

    # check length
    assert len(train) == 4
    assert len(test_known) == 5
    assert len(test_new) == 2

    # check cluster sizes
    assert len(train[train["cluster"] == 0]) == 1
    assert len(train[train["cluster"] == 1]) == 3
    assert len(test_known[test_known["cluster"] == 0]) == 3
    assert len(test_known[test_known["cluster"] == 1]) == 2
    assert len(test_new[test_new["cluster"] == 2]) == 1
    assert len(test_new[test_new["cluster"] == 3]) == 1

    # Check training ids
    assert training_ids.tolist() == [0, 1]


def test_ohe_clusters():
    training_cluster_ids = np.array([0, 1, 2, 3])

    # Training like encoding -- all categories in matrix
    X_ohe = DummeDataGenerator.ohe_clusters(
        pd.Series([0, 0, 1, 2, 2, 2, 3]), training_cluster_ids=training_cluster_ids
    )

    # check columns and sums
    assert X_ohe.columns.tolist() == [
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
    ]
    assert X_ohe.sum().tolist() == [2, 1, 3, 1]

    # New encoding -- no categories in matrix
    X_ohe = DummeDataGenerator.ohe_clusters(
        pd.Series([4, 4, 5, 6, 6, 7]), training_cluster_ids=training_cluster_ids
    )

    # check columns and sums
    assert X_ohe.columns.tolist() == [
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
    ]
    assert X_ohe.sum().tolist() == [0, 0, 0, 0]

    # Mixed encoding -- some categories in matrix
    X_ohe = DummeDataGenerator.ohe_clusters(
        pd.Series([1, 1, 3, 0, 0, 4, 5, 6, 6, 7]),
        training_cluster_ids=training_cluster_ids,
    )
    # check columns and sums
    assert X_ohe.columns.tolist() == [
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
    ]
    assert X_ohe.sum().tolist() == [2, 2, 0, 1]
