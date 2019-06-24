from criterion_core.utils import path
import numpy as np
import copy
import pandas as pd
from criterion_core.utils import tag_utils


def _sample_predictor(model, data_generator, output_index=None):
    for idx in range(len(data_generator)):
        samples, X = data_generator.get_batch(idx)
        outputs = model.predict(X)

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
             dfp.add_prefix('prob/'),
             dff,
             dfp.idxmax(axis=1).rename("prediction")), axis=1)
        df.dataset_id = df.dataset_id.astype('category')
        yield df

def sample_predictions(model, data_generator, output_index=None):
    return pd.concat(_sample_predictor(model, data_generator), axis=0)

def pivot_dataset_tags(datasets, tags):
    for ds in datasets:
        paths = [next(tag_utils.find_path(tags, "id", t["id"])) for t in ds["tags"]]
        df = pd.DataFrame.from_records([{p[0]["id"]: p[1]["name"]} for p in paths])
        df["dataset_id"] = pd.Series(ds["id"], index=df.index, dtype="category")
        yield df

def pivot_category_splitter(records):
    for r in records:
        cats = r["category"]
        for c in (cats if isinstance(cats, tuple) else (cats, )):
            yield {**r, "category":c}

def pivot_summarizer(df_samples, datasets, tags, output_index=None, accept_class="Accept"):
    df_samples = df_samples.groupby(['category', 'dataset', 'dataset_id', 'folder', 'prediction']).size().reset_index(name="count")
    df_samples["id"] = df_samples[["dataset_id", "folder", "prediction"]].apply(lambda x: "-".join(x), axis=1)
    df_samples["dataset_url"] = df_samples.dataset_id.apply(lambda x: "https://app.criterion.ai/data/" + x)

    df_tags = pd.concat(pivot_dataset_tags(datasets, tags), axis=0).set_index("dataset_id", drop=True)

    rec = df_samples.join(df_tags, on="dataset_id").to_dict("records")
    rec = [{k: v for k, v in x.items() if not isinstance(v, float) or not np.isnan(v)} for x in rec]
    rec = list(pivot_category_splitter(rec))
    add_outlier_classification_summary(rec, accept_class)

    fields = {
        "rowFields": [
            dict(key="category", label="Category"),
            dict(key="dataset", label="Dataset")
        ],
        "fields": [dict(key=c, label=next(tag_utils.find(tags, "id", c))["name"]) for c in df_tags.columns] +\
             [dict(key=c, label=c) for c in
              filter(lambda x: x not in ("id", "count", "category", "prediction", "dataset"), df_samples.columns)] +\
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

def get_classification_predictions(model, data_set, data_gen, class_names, output_index=-1):
    # saving target_mode to be able to restore

    prediction_output = []

    for ii in range(len(data_gen)):
        samples, X = data_gen.get_batch(ii)
        outputs = model.predict(X)

        predictions_class = outputs if output_index < 0 else outputs[output_index]
        prediction_output.append([dict(**s, **{"prob_{}".format(cn): float(pp) for cn, pp in zip(data_gen.classes, p)},
                                       **{'folder_{}'.format(ii): name for ii, name in enumerate(path.get_folders(s['path'], s['bucket']))},
                                      prediction=class_names[np.argmax(p)]) for s, p in zip(samples, predictions_class)])

    prediction_output = np.concatenate(prediction_output).tolist()
    return prediction_output


def get_classification_summary(class_names, prediction_output, class_weights, class_mapping):
    classification_cnt = {}
    for po in prediction_output:
        folder = path.get_folders(po["path"], po["bucket"])[-1]
        accept_class = class_weights in po["category"]
        prediction = po["prediction"]
        accepted = prediction == class_weights
        key = "-".join((po["id"], folder, prediction))
        classification_cnt.setdefault(key, {'count': 0, "categories": po["category"], "prediction": prediction,
                                            "sample_quality": ["reject", "accept"][accept_class], "id": po["id"],
                                            "Folder": folder,
                                            "decision": ["reject", "accept"][accepted]})["count"] += 1
    classification_summary = []
    for kk, vv in classification_cnt.items():
        for category in vv["categories"]:
            dd = copy.deepcopy(vv)
            del dd["categories"]
            dd.update({"class": category, "unique_id": kk})
            classification_summary.append(dd)
    return classification_summary

def tagged_pivot_classification_summary(classification_summary, ds_tag_records):
    # adding tags
    # adding dataset url
    tagged_classification_summary = []
    for cs in classification_summary:
        cs["dataset_url"] = "https://app.criterion.ai/data/" + cs["id"]
        tags = ds_tag_records[cs["id"]]
        for tt in tags:
            tag_cs = copy.deepcopy(cs)
            tag_cs.update(tt)
            tagged_classification_summary.append(tag_cs)
    return tagged_classification_summary
