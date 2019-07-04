import copy
import logging

import numpy as np
import pandas as pd

from criterion_core.utils import path
from criterion_core.utils import tag_utils

log = logging.getLogger(__name__)


def _sample_predictor(model, data_generator, output_index=None):
    for idx in range(len(data_generator)):
        samples, X = data_generator.get_batch(idx)
        outputs = model.predict(X)
        outputs = outputs if len(model.outputs) == 1 else outputs[0]

        class_names = data_generator.classes
        if output_index is not None:
            outputs = outputs[:, output_index, None]
            class_names = [data_generator.classes[output_index]]

        dfs = pd.DataFrame.from_records(list(samples))
        dfp = pd.DataFrame(outputs, columns=class_names)
        dff = dfs.apply(lambda x: pd.Series(path.get_folders(x["path"], x["bucket"])[-1]), axis=1)
        dff.columns = ["folder"]
        df = pd.concat(
            (dfs,
             dfp.add_prefix('prob_'),
             dff,
             dfp.idxmax(axis=1).rename("prediction")), axis=1)
        df.dataset_id = df.dataset_id.astype('category')
        yield df


def sample_predictions(model, data_generator, *args, **kwargs):
    return pd.concat(_sample_predictor(model, data_generator, *args, **kwargs), axis=0)


def pivot_dataset_tags(datasets, tags):
    for ds in datasets:
        paths = [next(tag_utils.find_path(tags, "id", t["id"])) for t in ds["tags"]]
        df = pd.DataFrame.from_records([{"tag_" + p[0]["id"]: p[1]["name"]for p in paths}])
        df["dataset_id"] = pd.Series(ds["id"], index=df.index, dtype="category")
        yield df


def pivot_category_splitter(records):
    for r in records:
        cats = r["category"]
        for c in (cats if isinstance(cats, tuple) else (cats,)):
            yield {**r, "category": c}


def pivot_summarizer(df_samples, datasets, tags, accept_class="Accept"):
    df_samples = df_samples.groupby(['category', 'dataset', 'dataset_id', 'folder', 'prediction']).size().reset_index(
        name="count")
    df_samples["id"] = df_samples[["dataset_id", "folder", "prediction"]].apply(lambda x: "-".join(x), axis=1)
    df_samples["dataset_url"] = df_samples.dataset_id.apply(lambda x: "https://app.criterion.ai/data/" + x)

    df_tags = pd.concat(pivot_dataset_tags(datasets, tags), axis=0, sort=False).set_index("dataset_id", drop=True)

    rec = df_samples.join(df_tags, on="dataset_id").to_dict("records")
    rec = [{k: v for k, v in x.items() if not isinstance(v, float) or not np.isnan(v)} for x in rec]
    rec = list(pivot_category_splitter(rec))
    add_outlier_classification_summary(rec, accept_class)

    fields = {
        "rowFields": [
            dict(key="category", label="Category"),
            dict(key="dataset", label="Dataset")
        ],
        "fields": [dict(key=c, label=next(tag_utils.find(tags, "id", c.split("_")[-1]))["name"]) for c in
                   df_tags.columns] + \
                  [dict(key=c, label=c) for c in
                   filter(lambda x: x not in ("id", "count", "category", "prediction", "dataset", "dataset_id"),
                          df_samples.columns)] + \
                  [
                      dict(key="sample_quality", label="Sample Quality"),
                      dict(key="decision", label="Decision")
                  ],
        "colFields": [
            dict(key="prediction", label="Prediction"),
        ]
    }

    return rec, fields


def add_outlier_classification_summary(classification_summary, accept_name):
    for cs in classification_summary:
        accepted = cs["prediction"] == accept_name
        accept_class = accept_name == cs["category"]
        cs["sample_quality"] = ["reject", "accept"][accept_class]
        cs["decision"] = ["reject", "accept"][accepted]
    return classification_summary




