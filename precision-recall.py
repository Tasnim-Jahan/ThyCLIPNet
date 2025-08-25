
def precision_metric(y_true, y_pred, smooth=1e-6):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)

    pp = K.sum(y_pred_f)

    return (tp + smooth) / (pp + smooth)




def recall_metric(y_true, y_pred, smooth=1e-6):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)

    ap = K.sum(y_true_f)

    return (tp + smooth) / (ap + smooth)
