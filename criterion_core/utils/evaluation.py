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

        df = pd.concat((dfs, dfp.add_prefix('prob_'), dff, dfp.idxmax(axis=1).rename("prediction")), axis=1)
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


def compare_thresholds(df, class_names, thresholds=[50.0, 90.0, 95.0, 98.0, 99.0]):
    df_security = []
    for threshold in thresholds:
        dfp = df[["prob_" + cc for cc in class_names]]
        dfp.columns = class_names
        predictions = dfp.gt(threshold / 100.0)
        predictions["gray"] = predictions.sum(axis=1) == 0
        df_security.append(pd.concat((df, predictions), axis=1))
        df_security[-1] = df_security[-1].melt(
            id_vars=[cn for cn in df_security[-1].columns if cn not in class_names + ["gray"]], var_name="decision",
            value_name="detection")
        df_security[-1]["security_threshold"] = threshold
    df_security = pd.concat(df_security, axis=0)
    return df_security


def pivot_summarizer(df_samples, datasets, tags, class_names, accept_class="Accept"):
    df_samples = compare_thresholds(df_samples, class_names)
    df_samples = df_samples.groupby(['category', 'dataset', 'dataset_id', 'folder', 'decision', "security_threshold"])["detection"].sum().reset_index(
        name="count")
    df_samples["id"] = df_samples[["dataset_id", "folder", "decision"]].apply(lambda x: "-".join(x), axis=1)
    df_samples["dataset_url"] = df_samples.dataset_id.apply(lambda x: "https://app.criterion.ai/data/" + x)

    df_tags = pd.concat(pivot_dataset_tags(datasets, tags), axis=0, sort=False).set_index("dataset_id", drop=True)
    rec = df_samples.join(df_tags, on="dataset_id").to_dict("records")
    rec = [{k: v for k, v in x.items() if not isinstance(v, float) or not np.isnan(v)} for x in rec]
    rec = list(pivot_category_splitter(rec))
    # add_outlier_classification_summary(rec, accept_class)

    fields = {
        "rowFields": [
            dict(key="category", label="Category"),
            dict(key="dataset", label="Dataset")
        ],
        "fields": [dict(key=c, label=next(tag_utils.find(tags, "id", c.split("_")[-1]))["name"]) for c in
                   df_tags.columns] + \
                  [dict(key=c, label=c) for c in
                   filter(lambda x: x not in ("id", "count", "category", "decision", "prediction", "dataset", "dataset_id", "security_threshold"),
                          df_samples.columns)] + \
                  [
                      dict(key="security_threshold", label="Security threshold")
                  ],
        "colFields": [
            dict(key="decision", label="Decision"),
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
