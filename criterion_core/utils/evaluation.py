from criterion_core.utils import path
import numpy as np
import copy


def get_classification_predictions(model, data_set, data_gen, class_names, output_index=-1):
    # saving target_mode to be able to restore
    target_mode_save = data_gen.target_mode
    data_gen.target_mode = "samples"

    prediction_output = []

    for ii in range(len(data_gen)):
        X, samples = data_gen[ii]
        outputs = model.predict(X)

        predictions_class = outputs if output_index < 0 else outputs[output_index]
        prediction_output.append([dict(**s, **{"prob_{}".format(cn): float(pp) for cn, pp in zip(class_names, p)},
                                       **{'folder_{}'.format(ii): name for ii, name in enumerate(path.get_folders(s['path'], s['bucket']))},
                                      prediction=class_names[np.argmax(p)]) for s, p in zip(samples, predictions_class)])

    # restoring data generator target mode
    data_gen.target_mode = target_mode_save

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
