from criterion_core.utils import path
import copy


def create_classification_summary(devel_pred_output, class_list):
    classification_summary = get_classification_summary(class_list, devel_pred_output)
    classification_summary = add_outlier_classification_summary(classification_summary, class_list[0])
    ds_tag_records, tag_fields = tag_utils.get_dataset_tag_records(args, config.datasets)
    tag_classification_summary = tagged_pivot_classification_summary(classification_summary, ds_tag_records)
    all_fields = tag_fields + [(kk, kk) for kk in classification_summary[0].keys() if kk not in ["id"]]

    views = dict(rowFields=["Batches", "Dataset", "class"], colFields=["prediction"])
    views["fields"] = [ff[1] for ff in all_fields if ff[1] not in views['rowFields']+views['colFields']]
    for kk in views.keys():
        views[kk] = [ff for vkk in views[kk] for ff in all_fields if ff[1] == vkk]

    return tag_classification_summary, views


def get_classification_summary(class_names, prediction_output):
    classification_cnt = {}
    for po in prediction_output:
        folder = path.get_folders(po["path"], po["bucket"])[-1]
        prediction = po["prediction"]
        key = "-".join((po["id"], folder, prediction))
        classification_cnt.setdefault(key, {'count': 0, "categories": po["category"], "prediction": prediction,
                                            "Folder": folder,  "id": po["id"]})["count"] += 1
    classification_summary = []
    for kk, vv in classification_cnt.items():
        for category in vv["categories"]:
            dd = copy.deepcopy(vv)
            del dd["categories"]
            dd.update({"class": category, "unique_id": kk})
            classification_summary.append(dd)
    return classification_summary


def add_outlier_classification_summary(classification_summary, accept_name):
    for cs in classification_summary:
        accepted = cs["prediction"] == accept_name
        accept_class = accept_name == cs["class"]
        cs["sample_quality"] = ["reject", "accept"][accept_class]
        cs["decision"] = ["reject", "accept"][accepted]
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
            tag_cs["id"] = tag_cs["unique_id"]
            del tag_cs["unique_id"]
            tag_cs.update(tt)
            tagged_classification_summary.append(tag_cs)
    return tagged_classification_summary
