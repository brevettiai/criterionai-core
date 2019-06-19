from criterion_core.utils import path
import numpy as np


def get_classification_predictions(model, data_set, data_gen, class_names, output_index=-1):
    # saving target_mode to be able to restore
    target_mode_restore_tmp = data_gen.target_mode
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
    data_gen.target_mode = target_mode_restore_tmp

    prediction_output = np.concatenate(prediction_output).tolist()
    return prediction_output
