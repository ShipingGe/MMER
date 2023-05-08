import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def calc_metrics(y_true, y_pred, to_print=False, dataset="MOSEI"):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """

    if dataset == "ur_funny":
        test_preds = np.argmax(y_pred, 1)
        test_truth = y_true

        if to_print:
            # print("Confusion Matrix (pos/neg) :")
            # print(confusion_matrix(test_truth, test_preds))
            print("Classification Report (pos/neg) :")
            # print(classification_report(test_truth, test_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))

        return accuracy_score(test_truth, test_preds)

    else:
        test_preds = y_pred
        test_truth = y_true

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

        # pos - neg
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)

        if to_print:
            print("mae: ", mae)
            print("corr: ", corr)
            print("mult_acc: ", mult_a7)
            # print("Classification Report (pos/neg) :")
            # print(classification_report(binary_truth, binary_preds, digits=5))
            print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
            print("F1 score: ", f_score)

        # non-neg - neg
        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)

        # if to_print:
            # print("Classification Report (non-neg/neg) :")
            # print(classification_report(binary_truth, binary_preds, digits=5))
            # print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

        return accuracy_score(binary_truth, binary_preds)