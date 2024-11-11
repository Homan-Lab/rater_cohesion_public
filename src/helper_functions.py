import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocess as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import metrics.metrics as cohm
from metrics.xrr.xrr_class import xRR
import wandb

logging.basicConfig(level=logging.WARN)


def filter_df(df, group_dict, is_same_group=False):
    """This function filters the dataframe based on groups

    Args:
        df (_type_): Input data
        group_dict (_type_): Dictionaries with groups
        is_same_group (bool, optional): If groups being compared are the same. Defaults to False.

    Returns:
        _type_: Returns two dataframes after filtering
    """

    sub_df = df.copy(deep=True)
    comp_sub_df = df.copy(deep=True)

    if "annotator_political" in group_dict:
        sub_df = sub_df[
            (sub_df.annotator_political == group_dict["annotator_political"])
        ]
        # print(sub_df.shape)
    if "rater_age" in group_dict:
        sub_df = sub_df[(sub_df.rater_age == group_dict["rater_age"])]
        # print(sub_df.shape)
    if "rater_gender" in group_dict:
        sub_df = sub_df[(sub_df.rater_gender == group_dict["rater_gender"])]
        # print(sub_df.shape)
    if "rater_race" in group_dict:
        sub_df = sub_df[(sub_df.rater_race == group_dict["rater_race"])]
        # print(sub_df.shape)

    if (
        "user_specified_cross_group" in group_dict
        and group_dict["user_specified_cross_group"]
    ):
        if "cross_group_pol" in group_dict:
            comp_sub_df = comp_sub_df[
                (comp_sub_df.annotator_political == group_dict["cross_group_pol"])
            ]
        if "cross_group_age" in group_dict:
            comp_sub_df = comp_sub_df[
                (comp_sub_df.rater_age == group_dict["cross_group_age"])
            ]
        if "cross_group_gender" in group_dict:
            comp_sub_df = comp_sub_df[
                (comp_sub_df.rater_gender == group_dict["cross_group_gender"])
            ]
        if "cross_group_race" in group_dict:
            comp_sub_df = comp_sub_df[
                (comp_sub_df.rater_race == group_dict["cross_group_race"])
            ]

    if not is_same_group:
        comp_sub_df = comp_sub_df[~comp_sub_df.index.isin(sub_df.index)]

    return sub_df, comp_sub_df


def filter_df_modular(df, group_dict, demographic_column):
    sub_df = df.copy(deep=True)
    comp_sub_df = df.copy(deep=True)
    sub_df = sub_df[(sub_df[demographic_column] == group_dict[demographic_column])]
    if (
        "user_specified_cross_group" in group_dict
        and group_dict["user_specified_cross_group"]
    ):
        comp_sub_df = comp_sub_df[
            (comp_sub_df[demographic_column] != group_dict[demographic_column])
        ]
    comp_sub_df = comp_sub_df[~comp_sub_df.index.isin(sub_df.index)]
    return sub_df, comp_sub_df


def encode_df(
    sub_df,
    comp_sub_df,
    df_voiced,
    group_label_column="Q_overall",
    cross_group_label_column="Q_overall",
    label_choices=None,
):
    """This function encodes the labels

    Args:
        sub_df (_type_): Input data for groups
        comp_sub_df (_type_): Input data for cross groups
        df_voiced (_type_): Original data
        group_label_column (str, optional): Label column for groups. Defaults to "Q_overall".
        cross_group_label_column (str, optional): Label column for cross groups. Defaults to "Q_overall".
        label_choices (_type_, optional): Labels array. Defaults to None.

    Returns:
        _type_: Returns two dataframes with encoded labels
    """
    if label_choices is None:
        label_choices = pd.unique(df_voiced[group_label_column])
    le = LabelEncoder()
    le.fit(label_choices)

    sub_df["label"] = le.transform(sub_df[group_label_column])
    comp_sub_df["label"] = le.transform(comp_sub_df[cross_group_label_column])

    return sub_df, comp_sub_df


def group_df(sub_df, comp_sub_df):
    """This function groups the labels

    Args:
        sub_df (_type_): Input data for groups
        comp_sub_df (_type_): Input data for cross groups

    Returns:
        _type_: Returns two dataframes after grouping
    """

    grouped_df = sub_df.groupby("item_id").apply(
        lambda s: pd.Series(
            {
                "id": s["id"].to_list(),
                "labels_dict": dict(zip(s["rater_id"], s["label"] + 1)),
            }
        )
    )

    comp_grouped_df = comp_sub_df.groupby("item_id").apply(
        lambda s: pd.Series(
            {
                "id": s["id"].to_list(),
                "labels_dict": dict(zip(s["rater_id"], s["label"] + 1)),
            }
        )
    )

    return grouped_df, comp_grouped_df


def vectorize_df(df, labels_dict_column):
    """This function vectorizes the labels in a dataframe

    Args:
        df (_type_): Input data
        labels_dict_column (_type_): Column name

    Returns:
        _type_: Returns a dataframe with vectorized labels
    """

    dictvectorizer = DictVectorizer(sparse=False)

    # Convert dictionary into feature matrix
    features = dictvectorizer.fit_transform(df[labels_dict_column])
    df["labels_list"] = np.where(features == 0, np.nan, features - 1).tolist()
    # print(features, len(dictvectorizer.get_feature_names_out()), features.shape)

    return df


def make_num_items_equal(source_df, target_df, column_name):
    """Make the number of items same for comparison

    Args:
        source_df (_type_): Input data
        target_df (_type_): Input data
        column_name (_type_): Column name

    Returns:
        _type_: List of labels
    """
    target_labels_list = target_df[column_name].to_list()
    idx_list = list(set(source_df.item_id.unique()) - set(target_df.index))
    if idx_list:
        idx_list.sort()
        # idx_list
        # print(len(target_labels_list))
        num_elements = 0
        if target_labels_list:
            num_elements = len(target_labels_list[0])
        for idx in idx_list:
            # print(target_labels_list[idx])
            target_labels_list.insert(idx, [np.nan] * num_elements)
            # print(target_labels_list[idx])
            # break
        # print(len(target_labels_list))
    return target_labels_list


def calculate_cohesion_metrics(
    grouped_labels_list, comp_grouped_labels_list, k, run_cross_group_metrics=True
):
    """This function computes the cohesion metrics

    Args:
        grouped_labels_list (_type_): Input data for groups
        comp_grouped_labels_list (_type_): Input data for cross groups
        k (_type_): Number of responses
        run_cross_group_metrics (bool, optional): Whether to run cross group metrics. Defaults to True.

    Returns:
        _type_: Returns calculated metrics
    """

    Y = np.array(grouped_labels_list)
    Z = np.array(comp_grouped_labels_list)
    # print(Y.shape)
    # print(Z.shape)

    k_alpha = 0.0
    try:
        k_alpha = cohm.krippendorffs_alpha(Y)
        # print(f"Krippendorff's alpha: {k_alpha}")
    except Exception as err:
        print(Y.shape, Z.shape)
        print("Exception: ", err)

    neg_entropy = 0.0
    try:
        neg_entropy = cohm.negentropy(Y, k)
        # print(f"Negentropy: {neg_entropy}")
    except Exception as err:
        print(Y.shape, Z.shape)
        print("Exception: ", err)

    plurality_score = 0.0
    try:
        plurality_score = cohm.plurality_size(Y)
        # print(f"Plurality size: {plurality_score}")
    except Exception as err:
        print(Y.shape, Z.shape)
        print("Exception: ", err)

    if run_cross_group_metrics:
        cross_neg_entropy = 0.0
        try:
            cross_neg_entropy = cohm.cross_negentropy(Y, Z, k)
            # print(f"Cross-negentropy: {cross_neg_entropy}")
        except Exception as err:
            print(Y.shape, Z.shape)
            print("Exception: ", err)

        voting_agreement_score = 0.0
        try:
            voting_agreement_score = cohm.voting_agreement(Y, Z)
            # print(f"Voting Agreement: {voting_agreement_score}")
        except Exception as err:
            print(Y.shape, Z.shape)
            print("Exception: ", err)

        return (
            k_alpha,
            neg_entropy,
            cross_neg_entropy,
            plurality_score,
            voting_agreement_score,
        )

    return k_alpha, neg_entropy, plurality_score


def save_metrics(base_path, metrics_df):
    """Saves metrics at the given location

    Args:
        base_path (_type_): Directory location
        metrics_df (_type_): Metrics dataframe
    """
    path_exists = os.path.exists(base_path)
    if not path_exists:
        os.makedirs(base_path)
    path_metrics_csv = os.path.join(base_path, "metrics.csv")
    path_metrics_json = os.path.join(base_path, "metrics.json")
    path_metrics_tex = os.path.join(base_path, "metrics.tex")

    metrics_df.to_csv(path_metrics_csv, index=False)
    metrics_df.to_json(path_metrics_json, orient="records", lines=True)
    metrics_df.style.to_latex(path_metrics_tex)


def compute_xrr(sub_df, comp_sub_df, workers=1):
    """This function computes XRR between two groups

    Args:
        sub_df (_type_): Input data for group
        comp_sub_df (_type_): Input data for cross group
        workers (int, optional): Number of workers. Defaults to 1.

    Returns:
        _type_: Value of XRR
    """

    xrr_score = 0.0
    try:
        xrr_obj = xRR(sub_df, comp_sub_df, "label", "rater_id", "item_id")
        xrr_score = xrr_obj.kappa_x()
    except KeyError as err:
        print(
            sub_df.shape,
            comp_sub_df.shape,
            np.intersect1d(
                sub_df["item_id"].unique(), comp_sub_df["item_id"].unique()
            ).shape,
        )
        print("KeyError: ", err)
    except Exception as err:
        print(
            sub_df.shape,
            comp_sub_df.shape,
            np.intersect1d(
                sub_df["item_id"].unique(), comp_sub_df["item_id"].unique()
            ).shape,
        )
        print("Exception: ", err)
    return xrr_score


def save_perm_test(base_path, num_trials, metrics_arr, columns=None):
    """Save permutation test results

    Args:
        base_path (_type_): Directory
        num_trials (_type_): Number of trials
        metrics_arr (_type_): Metrics array
        columns (_type_, optional): List of columns. Defaults to None.
    """
    path = os.path.join(base_path, "perm_test")
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)

    path_perm_metrics = os.path.join(path, f"perm_metrics_arr_{num_trials}.npz")
    path_perm_metrics_csv = os.path.join(path, "perm_mean_metrics.csv")
    path_perm_metrics_json = os.path.join(path, "perm_mean_metrics.json")
    path_perm_metrics_tex = os.path.join(path, "perm_mean_metrics.tex")

    np.savez(path_perm_metrics, metrics_arr=metrics_arr, label="metrics")

    if columns is None:
        columns = [
            "IRR",
            "XRR",
            "Negentropy",
            "Cross Negentropy",
            "Plurality Size",
            "Voting Agreement",
            "GAI",
        ]

    perm_metrics_df = pd.DataFrame(
        np.mean(metrics_arr, axis=0),
        columns=columns,
    )
    # perm_metrics_df = metrics_df[["Group"]].join(perm_metrics_df)
    # perm_metrics_df

    perm_metrics_df.to_csv(path_perm_metrics_csv, index=False)
    perm_metrics_df.to_json(path_perm_metrics_json, orient="records", lines=True)
    perm_metrics_df.to_latex(path_perm_metrics_tex, index=False)


def permutation_test(
    num_trials,
    cols_to_permutate,
    cols_to_sample,
    group_dicts,
    label_column,
    labels_dict_column,
    labels_list_column,
    original_df,
    base_path,
    label_choices,
    run_xrr=False,
    metrics_cols=None,
    run_cross_group_metrics=True,
    vic_in_gp=False,
    is_same_group=False,
):
    """This function runs the permutation test

    Args:
        num_trials (_type_): Number of trials
        cols_to_permutate (_type_): Columns used in permutation
        cols_to_sample (_type_): Sampling columns
        group_dicts (_type_): Group dictionaries
        label_column (_type_): Label column
        labels_dict_column (_type_): Labels dictionary
        labels_list_column (_type_): Labels list
        original_df (_type_): Original data
        base_path (_type_): Directory location
        label_choices (_type_): List of labels
        run_xrr (bool, optional): Flag to run xrr. Defaults to False.
        metrics_cols (_type_, optional): Columns for metrics. Defaults to None.
        run_cross_group_metrics (bool, optional): Run cross group metrics. Defaults to True.
        vic_in_gp (bool, optional): Vicarious in group experiments. Defaults to False.
        is_same_group (bool, optional): Same comparison groups. Defaults to False.

    Returns:
        _type_: Array with metrics
    """
    metrics_from_samples = []
    num_responses = len(label_choices)

    path = os.path.join(base_path, f"perm_test/num_trials_{num_trials}")
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)

    # for i in tqdm(range(num_trials)):
    def permutation_trial(i):
        sampled_df = original_df[cols_to_sample].copy(deep=True)
        sampled_df = sampled_df.drop_duplicates(ignore_index=True)
        sampled_df[cols_to_permutate] = sampled_df[cols_to_permutate].sample(
            frac=1, random_state=i, ignore_index=True
        )
        # print(sampled_df.shape)
        merged_df = pd.merge(
            sampled_df,
            original_df.loc[:, ~original_df.columns.isin(cols_to_permutate)],
            on="rater_id",
            how="inner",
        )
        # print(merged_df.shape)

        path_df_json = os.path.join(path, f"permuted_df_{i}.json")
        # merged_df.to_csv(path_df_csv, index=False)
        merged_df.to_json(path_df_json, orient="records", lines=True)

        metrics_list = metric_calculation(
            group_dicts,
            merged_df,
            label_column,
            num_responses,
            metric_run=False,
            run_xrr=run_xrr,
            workers=1,
            run_cross_group_metrics=run_cross_group_metrics,
            vic_in_gp=vic_in_gp,
            is_same_group=is_same_group,
        )
        # metrics_from_samples.append(metrics_list)

        if (i + 1) % 100 == 0:
            print(f"{datetime.now()} Permutation Test Trial: {i+1}")

        return metrics_list

    with mp.Pool(mp.cpu_count()) as pool:
        metrics_from_samples = pool.map(permutation_trial, range(num_trials))

    print(f"***** Permutation Tests Completed with {num_trials} Trials")
    print(
        len(metrics_from_samples),
        len(metrics_from_samples[0]),
        len(metrics_from_samples[0][0]),
    )
    metrics_arr = np.array(metrics_from_samples)
    print(metrics_arr.shape)

    save_perm_test(base_path, num_trials, metrics_arr, columns=metrics_cols)
    return metrics_arr


def calculate_pvalues(metrics_arr, metrics_df, num_trials, sampling_column=2):
    """Calculate p-values

    Args:
        metrics_arr (_type_): Array with all metrics from permutation trials
        metrics_df (_type_): Dataframe with metrics
        num_trials (_type_): Number of trials
        sampling_column (int, optional): Column. Defaults to 2.

    Returns:
        _type_: P-values arrays
    """
    p_vals_gt = np.array(
        [
            (
                metrics_arr[:, idx, :]
                > metrics_df.iloc[idx][sampling_column:].to_numpy()
            ).sum(axis=0)
            / num_trials
            for idx in range(metrics_arr.shape[1])
        ]
    )
    p_vals_lt = np.array(
        [
            (
                metrics_arr[:, idx, :]
                < metrics_df.iloc[idx][sampling_column:].to_numpy()
            ).sum(axis=0)
            / num_trials
            for idx in range(metrics_arr.shape[1])
        ]
    )
    print(p_vals_gt.shape, p_vals_lt.shape)

    return p_vals_gt, p_vals_lt


def combine_metrics_p_values(
    metrics_df, p_vals_gt, p_vals_lt, dataset_name, metrics_cols
):
    """Combine p values for wandb

    Args:
        metrics_df (_type_): Metrics data
        p_vals_gt (_type_): P-values
        p_vals_lt (_type_): P-values
        dataset_name (_type_): Name of the dataset
        metrics_cols (_type_): Metrics columns

    Returns:
        _type_: Combined metrics list
    """
    metrics_dict = metrics_df.to_dict("records")
    combined_metrics = []
    for metrics_row, p_gt, p_lt in zip(metrics_dict, p_vals_gt, p_vals_lt):
        for metric, metric_id in zip(metrics_cols, range(len(metrics_cols))):
            metrics_row[f"{metric}_p_gt"] = p_gt[metric_id]
            metrics_row[f"{metric}_p_lt"] = p_lt[metric_id]
        metrics_row["dataset"] = dataset_name
        combined_metrics.append(metrics_row)
    return combined_metrics


def wandb_log(
    metrics, project_name, base_path=False, num_trials=1000, wandb_entity="rit_pl"
):
    """Log metrics to weights and biases

    Args:
        metrics (_type_): Metrics to log
        project_name (_type_): Name of the project
        base_path (bool, optional): Directory. Defaults to False.
        num_trials (int, optional): Number of trials. Defaults to 1000.
        wandb_entity (str, optional): Entity for wandb. Defaults to "rit_pl".
    """
    for row in metrics:
        run = wandb.init(entity=wandb_entity, project=project_name)
        run.summary.update(row)
        if base_path:
            artifact = wandb.Artifact(name="text_files", type="dataset")
            # artifact.add_dir(base_path)
            artifact.add_file(local_path=os.path.join(base_path, "metrics.tex"))
            artifact.add_file(local_path=os.path.join(base_path, "metrics.json"))
            artifact.add_file(
                local_path=os.path.join(base_path, "perm_test", "perm_mean_metrics.tex")
            )
            artifact.add_file(
                local_path=os.path.join(
                    base_path, "perm_test", "perm_mean_metrics.json"
                )
            )
            artifact.add_file(
                local_path=os.path.join(
                    base_path,
                    "perm_test",
                    f"perm_metrics_arr_{num_trials}.npz",
                )
            )
            run.log_artifact(artifact)
        run.finish()


def metric_calculation(
    group_dicts,
    original_df,
    group_label_column,
    num_responses,
    labels_dict_column="labels_dict",
    metric_run=True,
    run_xrr=False,
    workers=mp.cpu_count(),
    run_cross_group_metrics=True,
    vic_in_gp=False,
    is_same_group=False,
):
    """Calculate the metrics

    Args:
        group_dicts (_type_): Dictionaries containg the groups
        original_df (_type_): Original data
        group_label_column (_type_): Group label column
        num_responses (_type_): Number of responses
        labels_dict_column (str, optional): Labels dictionary column. Defaults to "labels_dict".
        metric_run (bool, optional): Defaults to True.
        run_xrr (bool, optional): Flag to run XRR. Defaults to False.
        workers (_type_, optional): Number of workers. Defaults to mp.cpu_count().
        run_cross_group_metrics (bool, optional): Flag to run cross group metrics. Defaults to True.
        vic_in_gp (bool, optional): Vicarious in group experiments. Defaults to False.
        is_same_group (bool, optional): Same comparison groups. Defaults to False.

    Returns:
        _type_: List of metrics
    """

    metrics_list = []
    cross_group_label_column = group_label_column
    label_choices = pd.unique(original_df[group_label_column])
    for group_dict in tqdm(group_dicts):
        row = []
        if "dimension" in group_dict and metric_run:
            row.append(group_dict["dimension"])
        if "group" in group_dict and metric_run:
            row.append(group_dict["group"])

        sub_df, comp_sub_df = filter_df(original_df, group_dict, is_same_group)

        if "group_label_column" in group_dict:
            group_label_column = group_dict["group_label_column"]
            sub_df = sub_df[sub_df[group_label_column] != -1]
        if "cross_group_label_column" in group_dict:
            cross_group_label_column = group_dict["cross_group_label_column"]
            comp_sub_df = comp_sub_df[comp_sub_df[cross_group_label_column] != -1]

        sub_df, comp_sub_df = encode_df(
            sub_df,
            comp_sub_df,
            original_df,
            group_label_column,
            cross_group_label_column,
            label_choices,
        )
        grouped_df, comp_grouped_df = group_df(sub_df, comp_sub_df)
        grouped_df = vectorize_df(grouped_df, labels_dict_column)
        comp_grouped_df = vectorize_df(comp_grouped_df, labels_dict_column)
        grouped_labels_list = make_num_items_equal(
            original_df, grouped_df, "labels_list"
        )
        comp_grouped_labels_list = make_num_items_equal(
            original_df, comp_grouped_df, "labels_list"
        )
        if vic_in_gp:
            coh_metrics = calculate_cohesion_metrics(
                comp_grouped_labels_list,
                [],
                num_responses,
                run_cross_group_metrics=run_cross_group_metrics,
            )
        else:
            coh_metrics = calculate_cohesion_metrics(
                grouped_labels_list,
                comp_grouped_labels_list,
                num_responses,
                run_cross_group_metrics=run_cross_group_metrics,
            )

        coh_metrics = list(coh_metrics)
        if run_cross_group_metrics and run_xrr:
            xrr = compute_xrr(sub_df, comp_sub_df, workers=workers)
            coh_metrics.insert(1, xrr)
            if xrr != 0:
                coh_metrics.append(coh_metrics[0] / xrr)
            else:
                coh_metrics.append(0.0)
        row.extend(coh_metrics)
        metrics_list.append(row)
    return metrics_list
