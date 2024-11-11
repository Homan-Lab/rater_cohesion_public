from typing import Any
import numpy as np
import numpy.typing as npt
import krippendorff
from scipy import stats as st
from scipy.special import xlogy
from nltk.metrics.agreement import AnnotationTask


def calculate_metrics() -> None:
    """Runs metrics"""
    Y = np.array([[2, 0, 0], [0, 0, 2], [1, 1, 1], [0, 1, 1], [0, 0, 0]])
    Z = np.array([[2, 0, 0], [2, 0, 2], [1, 1, 1], [1, 2, 0], [0, 0, 0]])
    print(Y)
    print(Z)

    # in-group cohesion metrics
    k_alpha = krippendorffs_alpha(Y)
    print(f"Krippendorff's alpha: {k_alpha}")

    plurality_score = plurality_size(Y)
    print(f"Plurality size: {plurality_score}")

    neg_entropy = negentropy(Y, 3)
    print(f"Negentropy: {neg_entropy}")

    # cross-group divergence metrics
    voting_agreement_score = voting_agreement(Y, Z)
    print(f"Voting Agreement: {voting_agreement_score}")

    cross_neg_entropy = cross_negentropy(Y, Z, 3)
    print(f"Cross-negentropy: {cross_neg_entropy}")

    return


def krippendorffs_alpha(Y: npt.NDArray[Any], level_of_measurement="nominal") -> float:
    """This function computes Inter Annotator Agreement using Krippendorff's alpha

    Args:
        Y (npt.NDArray[Any]): Input data
        level_of_measurement (str, optional):
            Takes "nominal", "ordinal", "interval", "ratio", or a callable.
            Defaults to "nominal".

    Returns:
        float: Value for Krippendorff's alpha
    """

    reliability_data = np.transpose(Y)
    # print(reliability_data.shape)

    k_alpha = krippendorff.alpha(
        reliability_data=reliability_data, level_of_measurement=level_of_measurement
    )

    return k_alpha


def nltk_irr(df, rater_id_column, item_id_column, label_column):
    """This function computes Krippendorff's alpha using NLTK

    Args:
        df (_type_): Data for the task
        rater_id_column (_type_): Rater column
        item_id_column (_type_): Item column
        label_column (_type_): Label column

    Returns:
        _type_: Value for Krippendorff's alpha
    """
    t = AnnotationTask(data=df[[rater_id_column, item_id_column, label_column]].values)
    return t.alpha()


def plurality_size(Y: npt.NDArray[Any], nan_policy="omit") -> float:
    """This function computes Plurality Size

    Args:
        Y (npt.NDArray[Any]): Input data
        nan_policy (str, optional): How to handle nan. Defaults to "omit".

    Returns:
        float: Value of Plurality Size
    """
    # measure plurality size

    num_raters = np.sum(~np.isnan(Y), axis=-1)
    Y = Y[num_raters != 0]
    num_raters = num_raters[num_raters != 0]

    majority_vote = st.mode(Y, axis=-1, nan_policy=nan_policy, keepdims=False)
    plurality_score = np.mean(majority_vote[1] / num_raters)

    return plurality_score


def negentropy(Y: npt.NDArray[Any], k: int) -> float:
    """This function computes negative entropy

    Args:
        Y (npt.NDArray[Any]): Input data
        k (int): Number of responses

    Returns:
        float: Value of negative entropy
    """

    num_raters = np.sum(~np.isnan(Y), axis=-1)
    Y = Y[num_raters != 0]

    probs = np.asarray(
        [np.sum(y == x) / np.sum(~np.isnan(y)) for y in Y for x in range(k)]
    ).reshape((-1, k))

    # print(probs)

    # entropy = st.entropy(probs, axis=-1)
    entropy = -np.sum(xlogy(probs, probs), axis=-1)
    neg_entropy = np.log(k) - entropy
    neg_entropy = np.mean(neg_entropy)

    return neg_entropy


def voting_agreement(
    Y: npt.NDArray[Any], Z: npt.NDArray[Any], nan_policy="omit"
) -> float:
    """This function computes voting agreement between two groups

    Args:
        Y (npt.NDArray[Any]): Input data for group 1
        Z (npt.NDArray[Any]): Input data for group 2
        nan_policy (str, optional): How to handle nan. Defaults to "omit".

    Returns:
        float: Value for voting agreement
    """

    num_raters_y = np.sum(~np.isnan(Y), axis=-1)
    num_raters_z = np.sum(~np.isnan(Z), axis=-1)
    idxs_with_raters = np.logical_and(num_raters_y != 0, num_raters_z != 0)
    Y = Y[idxs_with_raters]
    Z = Z[idxs_with_raters]

    majority_vote_y = st.mode(Y, axis=-1, nan_policy=nan_policy, keepdims=True)
    majority_vote_z = st.mode(Z, axis=-1, nan_policy=nan_policy, keepdims=True)
    # print(majority_vote_y[0], majority_vote_z[0])

    majority_votes = np.asarray(
        [np.append(y, z) for y, z in zip(majority_vote_y[0], majority_vote_z[0])]
    )
    # print(majority_votes)

    voting_agreement_score = krippendorffs_alpha(majority_votes)

    return voting_agreement_score


def cross_negentropy(Y: npt.NDArray[Any], Z: npt.NDArray[Any], k: int) -> float:
    """This function computes cross negentropy

    Args:
        Y (npt.NDArray[Any]): Input data for group 1
        Z (npt.NDArray[Any]): Input data for group 2
        k (int): Number of responses

    Returns:
        float: Value for cross negentropy
    """

    num_raters_y = np.sum(~np.isnan(Y), axis=-1)
    num_raters_z = np.sum(~np.isnan(Z), axis=-1)
    idxs_with_raters = np.logical_and(num_raters_y != 0, num_raters_z != 0)
    Y = Y[idxs_with_raters]
    Z = Z[idxs_with_raters]

    probs_y = np.asarray(
        [np.sum(y == x) / np.sum(~np.isnan(y)) for y in Y for x in range(k)]
    ).reshape((-1, k))

    probs_z = np.asarray(
        [np.sum(z == x) / np.sum(~np.isnan(z)) for z in Z for x in range(k)]
    ).reshape((-1, k))

    # print(probs_y.shape, probs_z.shape)

    prod = np.where(probs_z == 0, 0, xlogy(probs_y, probs_z))
    cross_entropy = -np.sum(prod, axis=-1)
    cross_neg_entropy = np.log(k) - cross_entropy
    cross_neg_entropy = np.mean(cross_neg_entropy)

    return cross_neg_entropy


if __name__ == "__main__":
    calculate_metrics()
