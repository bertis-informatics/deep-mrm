import numpy as np



def convert_to_score_matrix(predicted_score_ser):

    if type(predicted_score_ser.values[0]) == str:
        y_proba = predicted_score_ser.apply(lambda x : np.fromstring(x[1:-1], sep=','))
        y_proba = np.stack(y_proba, axis=0)
    else:
        y_proba = np.array(predicted_score_ser.values.tolist())

    return y_proba