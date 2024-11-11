import pandas as pd
from helper_functions import (
    permutation_test,
    save_metrics,
    calculate_pvalues,
    combine_metrics_p_values,
    wandb_log,
    metric_calculation,
)

NUM_TRIALS = 1000
NUM_RESPONSES = 2

CT_FILTERS = [False, True]
# CT_FILTERS = [False]
# CT_FILTERS = [True]

OFFENSE_TYPES = ["personal", "vicarious_ing_v_personal"]
# OFFENSE_TYPES = ["vicarious", "vicarious_in_group"]
# OFFENSE_TYPES = [
#     "vicarious_cross_group",
#     "vicarious_cross_group_same_pol",
#     "vicarious_ing_v_same_pol_personal",
# ]

# SPLITS = ["gun","abortion","all"]
SPLITS = ["all"]


def run_experiments(
    ct_filter,
    split_name,
    dataset_name,
    data_load_path,
    group_dicts,
    offense_type,
    run_xrr_flag=True,
    run_cross_group_metrics=True,
    vic_in_gp=False,
    is_same_group=False,
):
    """This function runs the permutation test experiments

    Args:
        ct_filter (_type_): Flag for CrowdTruth
        split_name (_type_): Dataset split
        dataset_name (_type_): Name of the dataset
        data_load_path (_type_): Path to the dataset
        group_dicts (_type_): Dictionaries specifying the groups for experiments
        offense_type (_type_): Offense type
        run_xrr_flag (bool, optional): Flag for running xrr. Defaults to True.
        run_cross_group_metrics (bool, optional): Flag for running cross group metrics.
            Defaults to True.
        vic_in_gp (bool, optional): Vicarious in group experiments. Defaults to False.
        is_same_group (bool, optional): Groups being compared are the same. Defaults to False.
    """

    print("Running permutation for data split: ", split_name)
    dataset_name = f"{dataset_name}_{split_name}"

    if ct_filter:
        dataset_name = f"ct_{dataset_name}"
    label_choices = [1, 0]
    base_path = f"../output/{dataset_name}"

    df_voiced = pd.read_excel(data_load_path)
    df_voiced.head()
    if split_name == "gun" or split_name == "abortion":
        df_voiced[["news", "split", "rating"]] = df_voiced.dataset_bin.str.split(
            "_", expand=True
        )
        df_voiced = df_voiced.loc[df_voiced["split"] == split_name]

    df_voiced = df_voiced.rename(
        columns={
            "annotator_id": "rater_id",
            "comment_id": "item_id",
            "age": "rater_age",
            "race": "rater_race",
            "gender": "rater_gender",
        }
    )
    df_voiced["id"] = df_voiced.index

    new_cols = [
        "id",
        "rater_id",
        "item_id",
        "dataset",
        "duration",
        "dataset_bin",
        "comment_text",
        "PERSON_TOXIC",
        "PERSON_TOXIC_raw",
        "DEM_TOXIC",
        "DEM_TOXIC_raw",
        "REP_TOXIC",
        "REP_TOXIC_raw",
        "IND_TOXIC",
        "IND_TOXIC_raw",
        "online",
        "social",
        "2016_election",
        "2020_election",
        "rater_age",
        "rater_race",
        "rater_gender",
        "education",
        "annotator_political",
        "published_at",
    ]
    df_voiced = df_voiced[new_cols]

    label_column = "PERSON_TOXIC"
    df_voiced = df_voiced[df_voiced[label_column] != -1]

    df_columns = ["Dimension", "Group", "IRR", "Negentropy", "Plurality Size"]
    metrics_cols = ["IRR", "Negentropy", "Plurality Size"]
    if run_cross_group_metrics:
        df_columns = [
            "Dimension",
            "Group",
            "IRR",
            "Negentropy",
            "Cross Negentropy",
            "Plurality Size",
            "Voting Agreement",
        ]
        metrics_cols = [
            "IRR",
            "Negentropy",
            "Cross Negentropy",
            "Plurality Size",
            "Voting Agreement",
        ]
        if run_xrr_flag:
            df_columns = [
                "Dimension",
                "Group",
                "IRR",
                "XRR",
                "Negentropy",
                "Cross Negentropy",
                "Plurality Size",
                "Voting Agreement",
                "GAI",
            ]
            metrics_cols = [
                "IRR",
                "XRR",
                "Negentropy",
                "Cross Negentropy",
                "Plurality Size",
                "Voting Agreement",
                "GAI",
            ]

    metrics_list = metric_calculation(
        group_dicts,
        df_voiced,
        label_column,
        len(label_choices),
        run_xrr=run_xrr_flag,
        run_cross_group_metrics=run_cross_group_metrics,
        vic_in_gp=vic_in_gp,
        is_same_group=is_same_group,
    )
    metrics_df = pd.DataFrame(metrics_list, columns=df_columns)
    save_metrics(base_path, metrics_df)

    cols_to_sample = [
        "rater_id",
        "annotator_political",
        "rater_gender",
        "rater_race",
        "rater_age",
    ]
    cols_to_permutate = [
        "annotator_political",
        "rater_gender",
        "rater_race",
        "rater_age",
    ]
    labels_dict_column = "labels_dict"
    labels_list_column = "labels_list"
    print(f"**** Running permutation test for data split:{split_name} ****")
    metrics_arr = permutation_test(
        NUM_TRIALS,
        cols_to_permutate,
        cols_to_sample,
        group_dicts,
        label_column,
        labels_dict_column,
        labels_list_column,
        df_voiced,
        base_path,
        label_choices,
        metrics_cols=metrics_cols,
        run_xrr=run_xrr_flag,
        run_cross_group_metrics=run_cross_group_metrics,
        vic_in_gp=vic_in_gp,
        is_same_group=is_same_group,
    )

    p_vals_gt, p_vals_lt = calculate_pvalues(
        metrics_arr, metrics_df, NUM_TRIALS, sampling_column=2
    )
    combined_metrics = combine_metrics_p_values(
        metrics_df, p_vals_gt, p_vals_lt, dataset_name, metrics_cols
    )
    combined_metrics = [
        {**row, "perspective": offense_type} for row in combined_metrics
    ]

    wandb_log(
        combined_metrics,
        project_name="crowdalign_metrics",
        base_path=base_path,
        num_trials=NUM_TRIALS,
    )


for ct_filter in CT_FILTERS:
    if ct_filter:
        print("Running permutation test for crowdtruth filtered data")
        DATA_LOAD_PATH = (
            "../datasets/voiced/crowdtruth/voiced/voiced_public_20240219_ct.xlsx"
        )
    else:
        print("Running permutation test for original data")
        DATA_LOAD_PATH = "../datasets/voiced/voiced_public_20240129.xlsx"

    for offense_type in OFFENSE_TYPES:
        RUN_CROSS_GROUP_METRICS = True
        VIC_IN_GP = False
        RUN_XRR_FLAG = True
        DATASET_NAME = "voiced"
        IS_SAME_GROUP = False

        if offense_type == "vicarious":
            DATASET_NAME = "voiced_vicarious"

            # Vicarious dicts
            group_dicts = [
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat v Everyone else -> Democrat",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat v Republican -> Democrat",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat v Independent -> Democrat",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican v Everyone else -> Republican",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican v Democrat -> Republican",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican v Independent -> Republican",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent v Everyone else -> Independent",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent v Democrat -> Independent",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent v Republican -> Independent",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "IND_TOXIC",
                },
            ]

        if offense_type == "personal":

            group_dicts = [
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning",
                    "group": "Democrat",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning",
                    "group": "Republican",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning",
                    "group": "Independent",
                },
                {"rater_gender": "Male", "dimension": "Gender", "group": "Man"},
                {"rater_gender": "Female", "dimension": "Gender", "group": "Woman"},
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Man",
                },
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Woman",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Man",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Woman",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Man",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Woman",
                },
            ]

            cross_group_dicts = [
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Man v Democrat, Woman",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_gender": "Female",
                },
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Man v All other men",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Man v Republican, Woman",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_gender": "Female",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Man v All other men",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Man v Independent, Woman",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_gender": "Female",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Male",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Man v All other men",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Woman v Democrat, Man",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Democrat",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Democrat, Woman v All other women",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Female",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Woman v Republican, Man",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Republican",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Republican, Woman v All other women",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Female",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Woman v Independent, Man",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_gender": "Male",
                },
                {
                    "annotator_political": "Independent",
                    "rater_gender": "Female",
                    "dimension": "Political Leaning, Gender",
                    "group": "Independent, Woman v All other women",
                    "user_specified_cross_group": True,
                    "cross_group_gender": "Female",
                },
            ]
            # group_dicts.extend(cross_group_dicts)

        if offense_type == "vicarious_in_group":
            RUN_CROSS_GROUP_METRICS = False
            VIC_IN_GP = True
            RUN_XRR_FLAG = False

            # Vicarious in group dicts
            group_dicts = [
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Everyone else -> Democrat",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Democrat",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Democrat",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Everyone else -> Republican",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Republican",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Republican",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Everyone else -> Independent",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Independent",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Independent",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "IND_TOXIC",
                },
            ]

        if offense_type == "vicarious_cross_group":
            DATASET_NAME = "voiced_5mar_vicarious_cg"

            # Vicarious cross group dicts
            group_dicts = [
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Democrat v Independent -> Democrat",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Democrat v Republican -> Democrat",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Republican v Independent -> Republican",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Republican v Democrat -> Republican",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Independent v Republican -> Independent",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Independent v Democrat -> Independent",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "IND_TOXIC",
                },
            ]

        if offense_type == "vicarious_ing_v_personal":
            DATASET_NAME = "voiced_5mar_ing_per"

            # Vicarious in group vs personal dicts
            group_dicts = [
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Democrat (v Democrat)",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Democrat (v Democrat)",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Republican (v Republican)",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Republican (v Republican)",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Independent (v Independent)",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Independent (v Independent)",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
            ]

        if offense_type == "vicarious_cross_group_same_pol":
            IS_SAME_GROUP = True
            DATASET_NAME = "voiced_5mar_cg_same_pol"

            # Vicarious cross group with same predicting political leaning dicts
            group_dicts = [
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Democrat v Republican -> Independent",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Independent v Republican -> Democrat",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "DEM_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Republican v Democrat -> Independent",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "IND_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Independent v Democrat -> Republican",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Democrat v Independent -> Republican",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "REP_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Republican v Independent -> Democrat",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "DEM_TOXIC",
                },
            ]

        if offense_type == "vicarious_ing_v_same_pol_personal":
            IS_SAME_GROUP = True
            DATASET_NAME = "voiced_5mar_ing_per_pol"

            # Vicarious in group vs personal with same predicting political leaning dicts
            group_dicts = [
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Democrat (v Republican)",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Republican",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Republican -> Independent (v Republican)",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Republican",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Republican (v Democrat)",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Democrat",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Democrat -> Independent (v Democrat)",
                    "group_label_column": "IND_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Democrat",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Democrat (v Independent)",
                    "group_label_column": "DEM_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
                {
                    "annotator_political": "Independent",
                    "dimension": "Political Leaning - Vicarious",
                    "group": "Independent -> Republican (v Independent)",
                    "group_label_column": "REP_TOXIC",
                    "user_specified_cross_group": True,
                    "cross_group_pol": "Independent",
                    "cross_group_label_column": "PERSON_TOXIC",
                },
            ]

        for split_name in SPLITS:
            run_experiments(
                ct_filter,
                split_name,
                DATASET_NAME,
                DATA_LOAD_PATH,
                group_dicts,
                offense_type,
                run_xrr_flag=RUN_XRR_FLAG,
                run_cross_group_metrics=RUN_CROSS_GROUP_METRICS,
                vic_in_gp=VIC_IN_GP,
                is_same_group=IS_SAME_GROUP,
            )
