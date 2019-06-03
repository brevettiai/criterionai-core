import numpy as np
from sklearn.model_selection import train_test_split

def flatten_dataset(datasets):
    """
    Generator flattening all datasets to list of samples
    :param datasets: dictionary of datasets
    :return:
    """
    for ds, dsv in datasets.items():
        for cat, samples in dsv.items():
            yield from samples

def merge_samples(datasets, selector):
    """
    Merge samples within a category.
    ex: merge samples stored as multiple files (channels) to one.
    :param datasets: dict of datasets
    :param selector: Selector function returning a key and a channel
    :return:
    """
    for ds, dsv in datasets.items():
        for cat, samples in dsv.items():
            merged = {}
            for sample in samples:
                key, channel = selector(sample)
                merged.setdefault(key, {})[channel] = sample
            dsv[cat] = list(merged.values())
    return datasets


def split_datasets(datasets, strategy="by_class", split=0.8, *args, **kwargs):
    """
    Utility function for performing different types of dataset splits
    :param datasets:
    :param strategy: Type of split, string as in SPLIT_STRATEGY, or split function itself
    :param split:
    :param args: args for splitting function
    :param kwargs: kwargs for splitting function
    :return:
    """
    if isinstance(strategy, str):
        strategy = SPLIT_STRATEGY[strategy]

    return strategy(datasets, split=split, *args, **kwargs)


def split_datasets_by_class(datasets, split):
    """
    Split data by class in every dataset
    :param datasets:
    :param split:
    :return:
    """
    s1, s2 = [], []
    for ds, dsv in datasets.items():
        for cat, samples in dsv.items():
            train, test = train_test_split(np.array(samples), test_size=int(len(samples) * split), random_state=42)
            s1.append(train)
            s2.append(test)

    return np.hstack(s1), np.hstack(s2)


SPLIT_STRATEGY = dict(
    by_class=split_datasets_by_class
)
