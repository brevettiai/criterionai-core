from criterion_core.utils import path
import numpy as np


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
        categories = po["category"]
        accept_class = class_weights in categories
        accepted = po["prediction"] == class_weights
        for cat in categories:
            key = (po["id"], folder, cat)
            classification_cnt.setdefault(key, dict(accept=0, reject=0))["accept"] += int(accepted) # counting accept
            classification_cnt[key]["reject"] += int(not accepted)  # counting reject
    classification_summary = []
    for kk, vv in classification_cnt.items():
        for pred in ["accept", "reject"]:
            classification_summary.append(
                {"id":kk[0], "Folder":kk[1], "class": kk[2], "prediction": pred, 'count': vv[pred]})
    return classification_summary
