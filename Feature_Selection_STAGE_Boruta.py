"""
Feature Selection - Boruta
LEAKAGE-SAFE / PATIENT-LEVEL
FedBCa Centers 1-4

IMPORTANT
---------
The radiomics CSV does NOT need to contain a label column.

Labels are loaded from:
    Center1/Center1_label.xlsx
    Center2/Center2_label.xlsx
    Center3/Center3_label.xlsx
    Center4/Center4_label.xlsx

Each label Excel file has:
    label
    image_name
    mask_name

Patient ID is constructed from:
    Center + image number

Example:
    Center1 + 021.nii.gz
    -> Center1_021

If a patient has multiple annotations, e.g.
    021_1.nii.gz
    021_2.nii.gz

both are mapped to the SAME patient:
    Center1_021

and therefore MUST remain in the same train/test split.

Boruta is fitted ONLY on the training data.
"""

from pathlib import Path
import os
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from boruta import BorutaPy

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

# CHANGE THIS to the radiomics CSV you are currently processing.
INPUT_CSV = "radiomics_features_FedBCa_Centers1-4_final.csv"

# Root dataset containing Center1 ... Center4
DATASET_DIR = r"D:\Research\DATASET\Revisi"

OUTPUT_DIR = Path(
    r"D:\Research\DATASET\Revisi\boruta_output"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

MISSING_THRESHOLD = 0.05

BORUTA_MAX_ITER = 100
BORUTA_PERC = 90

RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = 7
RF_RANDOM_STATE = 42

TOP_N_FEATURES = 30


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUTPUT_TRAIN = (
    OUTPUT_DIR / "train_selected_top30.csv"
)

OUTPUT_TEST = (
    OUTPUT_DIR / "test_selected_top30.csv"
)

# --------------------------------------------------------------
# PRE-BORUTA (full feature set, before feature selection)
# --------------------------------------------------------------

OUTPUT_TRAIN_PRE_BORUTA = (
    OUTPUT_DIR / "train_pre_boruta_all_features.csv"
)

OUTPUT_TEST_PRE_BORUTA = (
    OUTPUT_DIR / "test_pre_boruta_all_features.csv"
)

OUTPUT_FEATURES = (
    OUTPUT_DIR / "boruta_feature_ranking.csv"
)

OUTPUT_SPLIT = (
    OUTPUT_DIR / "patient_train_test_split.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR / "boruta_report.txt"
)


# ============================================================
# 3. PRINT
# ============================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 4. EXTRACT PATIENT NUMBER
# ============================================================

def extract_patient_number(filename):
    """
    Extract the patient number from image_name.

    Examples:
        001.nii.gz       -> 001
        021.nii.gz       -> 021
        021_1.nii.gz     -> 021
        021_2.nii.gz     -> 021

    The important rule is:
        annotation suffix (_1, _2, ...) is ignored.

    Only the FIRST numeric component is used.
    """

    name = os.path.basename(
        str(filename)
    )

    # Remove .nii.gz
    name = re.sub(
        r"\.nii\.gz$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove .nii if present
    name = re.sub(
        r"\.nii$",
        "",
        name,
        flags=re.IGNORECASE
    )

    match = re.match(
        r"^(\d+)",
        name
    )

    if not match:
        raise ValueError(
            f"Cannot extract patient number from "
            f"image_name: {filename}"
        )

    return match.group(1)


# ============================================================
# 5. LOAD LABEL FILES
# ============================================================

def load_all_labels():

    section("1. LOAD CENTER LABEL FILES")

    all_labels = []

    for center_number in range(1, 5):

        center_name = (
            f"Center{center_number}"
        )

        label_file = (
            Path(DATASET_DIR)
            / f"{center_name}_label.xlsx"
        )

        print(
            f"\nReading: {label_file}"
        )

        if not label_file.exists():
            raise FileNotFoundError(
                f"Label file not found:\n"
                f"{label_file}"
            )

        label_df = pd.read_excel(
            label_file
        )

        required = {
            "label",
            "image_name"
        }

        missing = (
            required -
            set(label_df.columns)
        )

        if missing:
            raise ValueError(
                f"{label_file.name} is missing "
                f"columns: {sorted(missing)}"
            )

        # ----------------------------------------------------
        # Create patient number from image_name
        # ----------------------------------------------------

        label_df["PatientNumber"] = (
            label_df["image_name"]
            .apply(
                extract_patient_number
            )
        )

        # ----------------------------------------------------
        # PatientID
        # ----------------------------------------------------

        label_df["PatientID"] = (
            center_name
            + "_"
            + label_df["PatientNumber"]
        )

        label_df["Center"] = (
            center_name
        )

        # Keep only what is required
        label_df = label_df[
            [
                "PatientID",
                "PatientNumber",
                "Center",
                "label",
                "image_name",
                "mask_name"
            ]
        ].copy()

        print(
            f"Rows loaded: {len(label_df)}"
        )

        print(
            f"Unique patient numbers: "
            f"{label_df['PatientNumber'].nunique()}"
        )

        all_labels.append(
            label_df
        )

    labels = pd.concat(
        all_labels,
        ignore_index=True
    )

    print(
        f"\nTotal label rows: {len(labels)}"
    )

    print(
        f"Total unique PatientID: "
        f"{labels['PatientID'].nunique()}"
    )

    return labels


# ============================================================
# 6. BUILD PATIENT-LEVEL LABEL TABLE
# ============================================================

def build_patient_label_table(labels):

    section("2. BUILD PATIENT-LEVEL LABEL TABLE")

    # A label file can contain multiple annotation rows.
    # We require all rows belonging to one PatientID to have
    # the same clinical label.

    patient_rows = []

    for patient_id, group in labels.groupby(
        "PatientID",
        sort=True
    ):

        label_values = (
            pd.to_numeric(
                group["label"],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        if len(label_values) == 0:
            continue

        if len(label_values) > 1:
            print("\n[WARNING] Multiple labels found for patient:")
            print(group.to_string(index=False))

            # Follow the original handling:
            # use the label from the first annotation
            selected_label = int(
                pd.to_numeric(
                    group["label"].iloc[0],
                    errors="coerce"
                )
            )

            print(
                f"[WARNING] PatientID {patient_id}: "
                f"using label={selected_label} "
                f"from first annotation."
            )

        else:
            selected_label = int(label_values[0])

        patient_rows.append(
            {
                "PatientID": patient_id,
                "Center": group["Center"].iloc[0],
                "label": selected_label
            }
        )

    patient_table = pd.DataFrame(
        patient_rows
    )

    print(
        f"Unique patients: "
        f"{len(patient_table)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        patient_table[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPatients by center:"
    )

    print(
        patient_table
        .groupby("Center")
        .size()
        .sort_index()
        .to_string()
    )

    return patient_table


# ============================================================
# 7. NORMALIZE RADIOMICS PATIENT IDs
# ============================================================

def normalize_radiomics_patient_id(
    value,
    center=None
):
    """
    Converts common PatientID representations to:

        Center1_001
        Center1_021

    The extraction CSV may already contain:
        Center1_021

    so those are preserved.

    If the CSV contains only:
        021

    the Center column is used.
    """

    value = str(value).strip()

    # Already contains Center information
    match = re.search(
        r"(Center[1-4])[_-]?(\d+)",
        value,
        flags=re.IGNORECASE
    )

    if match:

        center_name = (
            match.group(1)
            .replace("center", "Center")
        )

        number = match.group(2)

        return (
            f"{center_name}_{number}"
        )

    # Only numeric ID
    match = re.match(
        r"^(\d+)",
        value
    )

    if match and center is not None:

        center_name = str(
            center
        ).strip()

        if not center_name.lower().startswith(
            "center"
        ):
            center_name = (
                f"Center{center_name}"
            )

        number = match.group(1)

        return (
            f"{center_name}_{number}"
        )

    return value


# ============================================================
# 8. LOAD RADIOMICS + LABEL
# ============================================================

def prepare_radiomics(
    radiomics_path,
    labels
):

    section("3. LOAD RADIOMICS AND MERGE LABELS")

    if not os.path.exists(
        radiomics_path
    ):
        raise FileNotFoundError(
            f"Radiomics CSV not found:\n"
            f"{radiomics_path}"
        )

    df = pd.read_csv(
        radiomics_path
    )

    print(
        f"Radiomics rows: {len(df)}"
    )

    print(
        f"Radiomics columns: "
        f"{len(df.columns)}"
    )

    if "PatientID" not in df.columns:
        raise ValueError(
            "Radiomics CSV must contain "
            "'PatientID'."
        )

    # --------------------------------------------------------
    # Normalize Center
    # --------------------------------------------------------

    if "Center" not in df.columns:
        raise ValueError(
            "Radiomics CSV must contain "
            "'Center' so that PatientID can "
            "be mapped correctly."
        )

    df["Center"] = (
        df["Center"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Normalize PatientID
    # --------------------------------------------------------

    df["PatientID_original"] = (
        df["PatientID"]
        .astype(str)
        .str.strip()
    )

    df["PatientID"] = [
        normalize_radiomics_patient_id(
            pid,
            center
        )
        for pid, center in zip(
            df["PatientID_original"],
            df["Center"]
        )
    ]

    # --------------------------------------------------------
    # Detect duplicate annotations
    # --------------------------------------------------------

    duplicate_counts = (
        df["PatientID"]
        .value_counts()
    )

    duplicate_patients = (
        duplicate_counts[
            duplicate_counts > 1
        ]
    )

    print(
        f"\nPatients with multiple "
        f"radiomics rows: "
        f"{len(duplicate_patients)}"
    )

    if len(duplicate_patients) > 0:

        print(
            "These rows are RETAINED."
        )

        print(
            "They will be grouped by PatientID "
            "during train/test splitting."
        )

        print(
            duplicate_patients
            .head(10)
            .to_string()
        )

    # --------------------------------------------------------
    # Merge label using PatientID
    # --------------------------------------------------------

    label_mapping = labels[
        [
            "PatientID",
            "label"
        ]
    ].drop_duplicates(
        "PatientID"
    )

    df = df.merge(
        label_mapping,
        on="PatientID",
        how="left",
        validate="many_to_one"
    )

    missing_labels = df[
        "label"
    ].isna()

    if missing_labels.any():

        print(
            "\nRadiomics PatientIDs without labels:"
        )

        print(
            df.loc[
                missing_labels,
                [
                    "PatientID",
                    "Center"
                ]
            ]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )

        raise ValueError(
            f"{missing_labels.sum()} radiomics rows "
            "could not be matched to a label."
        )

    print(
        f"\n✓ All radiomics rows successfully "
        f"matched to labels."
    )

    print(
        f"Final radiomics rows: {len(df)}"
    )

    print(
        f"Unique patients: "
        f"{df['PatientID'].nunique()}"
    )

    return df


# ============================================================
# 9. PATIENT LEVEL SPLIT
# ============================================================

def patient_level_split(
    df,
    patient_table
):

    section("4. PATIENT-LEVEL 80:20 SPLIT")

    patient_ids = (
        patient_table[
            "PatientID"
        ]
        .to_numpy()
    )

    patient_labels = (
        patient_table[
            "label"
        ]
        .to_numpy()
    )

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_labels
    )

    train_ids = set(
        train_ids
    )

    test_ids = set(
        test_ids
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    if train_ids.intersection(
        test_ids
    ):

        raise RuntimeError(
            "Patient overlap detected "
            "between train and test."
        )

    # --------------------------------------------------------
    # Assign split to ALL annotations
    # --------------------------------------------------------

    df["Split"] = np.where(
        df["PatientID"].isin(
            train_ids
        ),
        "train",
        "test"
    )

    # --------------------------------------------------------
    # Final patient overlap check
    # --------------------------------------------------------

    train_patients = set(
        df.loc[
            df["Split"] == "train",
            "PatientID"
        ]
    )

    test_patients = set(
        df.loc[
            df["Split"] == "test",
            "PatientID"
        ]
    )

    overlap = (
        train_patients &
        test_patients
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE: "
            f"{list(overlap)[:20]}"
        )

    print(
        f"Training patients: "
        f"{len(train_patients)}"
    )

    print(
        f"Testing patients: "
        f"{len(test_patients)}"
    )

    print(
        "\n✓ Patient-level split successful."
    )

    print(
        "✓ Multiple annotations remain in "
        "the same split."
    )

    # --------------------------------------------------------
    # Center distribution
    # --------------------------------------------------------

    split_patient_table = (
        patient_table.copy()
    )

    split_patient_table["Split"] = np.where(
        split_patient_table["PatientID"].isin(
            train_ids
        ),
        "train",
        "test"
    )

    print(
        "\nTraining patients by center:"
    )

    print(
        split_patient_table[
            split_patient_table["Split"] == "train"
        ]
        .groupby("Center")
        .size()
        .sort_index()
        .to_string()
    )

    print(
        "\nTesting patients by center:"
    )

    print(
        split_patient_table[
            split_patient_table["Split"] == "test"
        ]
        .groupby("Center")
        .size()
        .sort_index()
        .to_string()
    )

    return (
        df,
        train_ids,
        test_ids,
        split_patient_table
    )


# ============================================================
# 10. RADIOMICS FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    df
):

    section("5. BUILD FEATURE MATRIX")

    metadata = {
        "PatientID",
        "PatientID_original",
        "Center",
        "label",
        "Split",
        "MaskIndex",
        "image_file",
        "mask_file",
        "image_name",
        "mask_name",
        "Condition"
    }

    feature_columns = [
        c
        for c in df.columns
        if c not in metadata
    ]

    X = (
        df[
            feature_columns
        ]
        .select_dtypes(
            include=[np.number]
        )
        .copy()
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric radiomics features found."
        )

    print(
        f"Numeric features: {X.shape[1]}"
    )

    return X


# ============================================================
# 11. TRAINING-ONLY PREPROCESSING
# ============================================================

def preprocess(
    df,
    X
):

    section("6. TRAINING-ONLY PREPROCESSING")

    train_mask = (
        df["Split"] == "train"
    )

    test_mask = (
        df["Split"] == "test"
    )

    X_train = X.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = (
        df.loc[
            train_mask,
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        df.loc[
            test_mask,
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Missingness filter FIT on TRAIN ONLY
    # --------------------------------------------------------

    missing_ratio = (
        X_train.isna()
        .mean()
    )

    keep_columns = (
        missing_ratio[
            missing_ratio <=
            MISSING_THRESHOLD
        ]
        .index
    )

    removed = (
        len(X_train.columns)
        - len(keep_columns)
    )

    X_train = X_train[
        keep_columns
    ]

    X_test = X_test[
        keep_columns
    ]

    print(
        f"Removed high-missing features: "
        f"{removed}"
    )

    print(
        f"Remaining features: "
        f"{len(keep_columns)}"
    )

    # --------------------------------------------------------
    # Median imputation FIT TRAIN ONLY
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )

    print(
        "✓ Imputer fitted on TRAIN only."
    )

    return (
        X_train_imp,
        X_test_imp,
        y_train,
        y_test,
        list(keep_columns)
    )


# ============================================================
# 11b. EXPORT TRAIN / TEST BEFORE BORUTA SELECTION
# ============================================================

def save_pre_boruta_train_test(
    df,
    X_train,
    X_test,
    feature_names
):
    """
    Save TRAIN and TEST sets with the FULL (pre-Boruta) feature
    set, i.e. after missingness filtering + imputation but
    BEFORE Boruta selects the final feature subset.

    This lets you inspect / audit exactly what goes INTO Boruta.
    """

    section(
        "6b. EXPORT TRAIN / TEST (PRE-BORUTA, ALL FEATURES)"
    )

    train_mask = (
        df["Split"] == "train"
    )

    test_mask = (
        df["Split"] == "test"
    )

    metadata_columns = [
        "PatientID",
        "label",
        "Center",
        "Split",
        "Condition",
        "MaskIndex"
    ]

    metadata_columns = [
        col
        for col in metadata_columns
        if col in df.columns
    ]

    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------

    train_metadata = (
        df.loc[
            train_mask,
            metadata_columns
        ]
        .reset_index(drop=True)
    )

    train_features_df = pd.DataFrame(
        X_train,
        columns=feature_names
    ).reset_index(drop=True)

    train_pre_boruta_df = pd.concat(
        [
            train_metadata,
            train_features_df
        ],
        axis=1
    )

    train_pre_boruta_df.to_csv(
        OUTPUT_TRAIN_PRE_BORUTA,
        index=False
    )

    # ----------------------------------------------------------
    # TEST
    # ----------------------------------------------------------

    test_metadata = (
        df.loc[
            test_mask,
            metadata_columns
        ]
        .reset_index(drop=True)
    )

    test_features_df = pd.DataFrame(
        X_test,
        columns=feature_names
    ).reset_index(drop=True)

    test_pre_boruta_df = pd.concat(
        [
            test_metadata,
            test_features_df
        ],
        axis=1
    )

    test_pre_boruta_df.to_csv(
        OUTPUT_TEST_PRE_BORUTA,
        index=False
    )

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print(
        f"TRAIN (pre-Boruta) shape : {train_pre_boruta_df.shape}"
    )

    print(
        f"TEST  (pre-Boruta) shape : {test_pre_boruta_df.shape}"
    )

    print(
        f"\n✓ Saved: {OUTPUT_TRAIN_PRE_BORUTA}"
    )

    print(
        f"✓ Saved: {OUTPUT_TEST_PRE_BORUTA}"
    )

    return (
        train_pre_boruta_df,
        test_pre_boruta_df
    )


# ============================================================
# 12. BORUTA
# ============================================================

def run_boruta(
    X_train,
    y_train,
    feature_names
):

    section("7. BORUTA - TRAIN ONLY")

    print(
        "Boruta receives TRAINING data only."
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=BORUTA_MAX_ITER,
        perc=BORUTA_PERC,
        random_state=RF_RANDOM_STATE,
        verbose=2
    )

    boruta.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    confirmed = [
        feature_names[i]
        for i in range(
            len(feature_names)
        )
        if boruta.support_[i]
    ]

    tentative = [
        feature_names[i]
        for i in range(
            len(feature_names)
        )
        if boruta.support_weak_[i]
    ]

    rejected = [
        feature_names[i]
        for i in range(
            len(feature_names)
        )
        if (
            not boruta.support_[i]
            and
            not boruta.support_weak_[i]
        )
    ]

    print(
        "\n" + "-" * 60
    )

    print(
        f"Confirmed BEFORE top 30: "
        f"{len(confirmed)}"
    )

    print(
        f"Tentative: {len(tentative)}"
    )

    print(
        f"Rejected: {len(rejected)}"
    )

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    ranking = pd.DataFrame({
        "Feature": feature_names,
        "Boruta_Rank": boruta.ranking_,
        "Confirmed": boruta.support_,
        "Tentative": boruta.support_weak_
    })

    ranking["Status"] = np.where(
        ranking["Confirmed"],
        "Confirmed",
        np.where(
            ranking["Tentative"],
            "Tentative",
            "Rejected"
        )
    )

    # Boruta rank 1 = strongest confirmed group.
    confirmed_ranking = (
        ranking[
            ranking["Confirmed"]
        ]
        .sort_values(
            ["Boruta_Rank", "Feature"]
        )
        .reset_index(drop=True)
    )

    top_n = min(
        TOP_N_FEATURES,
        len(confirmed_ranking)
    )

    top_features = (
        confirmed_ranking
        .head(top_n)
        ["Feature"]
        .tolist()
    )

    ranking["Top30"] = (
        ranking["Feature"]
        .isin(top_features)
    )

    return (
        boruta,
        ranking,
        confirmed,
        tentative,
        rejected,
        top_features
    )


# ============================================================
# 13. SAVE OUTPUT
# ============================================================

def save_outputs(
    df,
    X_train,
    X_test,
    feature_names,
    top_features,
    ranking,
    split_patient_table
):

    section("8. SAVE OUTPUT")

    train_mask = (
        df["Split"] == "train"
    )

    test_mask = (
        df["Split"] == "test"
    )

    train_features = pd.DataFrame(
        X_train,
        columns=feature_names,
        index=df.loc[
            train_mask
        ].index
    )

    test_features = pd.DataFrame(
        X_test,
        columns=feature_names,
        index=df.loc[
            test_mask
        ].index
    )

    base_columns = [
        "PatientID",
        "Center",
        "label"
    ]

    if "MaskIndex" in df.columns:
        base_columns.append(
            "MaskIndex"
        )

    train_output = pd.concat(
        [
            df.loc[
                train_mask,
                base_columns
            ],
            train_features[
                top_features
            ]
        ],
        axis=1
    )

    test_output = pd.concat(
        [
            df.loc[
                test_mask,
                base_columns
            ],
            test_features[
                top_features
            ]
        ],
        axis=1
    )

    train_output.to_csv(
        OUTPUT_TRAIN,
        index=False
    )

    test_output.to_csv(
        OUTPUT_TEST,
        index=False
    )

    ranking.to_csv(
        OUTPUT_FEATURES,
        index=False
    )

    split_patient_table.to_csv(
        OUTPUT_SPLIT,
        index=False
    )

    print(
        f"Train output : {OUTPUT_TRAIN}"
    )

    print(
        f"Test output  : {OUTPUT_TEST}"
    )

    print(
        f"Feature rank : {OUTPUT_FEATURES}"
    )

    print(
        f"Patient split: {OUTPUT_SPLIT}"
    )

    return (
        train_output,
        test_output
    )


# ============================================================
# 14. REPORT
# ============================================================

def save_report(
    patient_table,
    split_patient_table,
    df,
    confirmed,
    tentative,
    rejected,
    top_features
):

    report = []

    report.append(
        "BORUTA LEAKAGE-SAFE REPORT"
    )

    report.append(
        "=" * 60
    )

    report.append(
        f"Unique patients: "
        f"{len(patient_table)}"
    )

    report.append(
        f"Radiomics rows: "
        f"{len(df)}"
    )

    report.append(
        f"Training patients: "
        f"{sum(split_patient_table['Split'] == 'train')}"
    )

    report.append(
        f"Testing patients: "
        f"{sum(split_patient_table['Split'] == 'test')}"
    )

    report.append("")

    report.append(
        "CENTER DISTRIBUTION"
    )

    report.append(
        split_patient_table
        .groupby(
            ["Split", "Center"]
        )
        .size()
        .to_string()
    )

    report.append("")

    report.append(
        f"Confirmed Boruta BEFORE top 30: "
        f"{len(confirmed)}"
    )

    report.append(
        f"Tentative: {len(tentative)}"
    )

    report.append(
        f"Rejected: {len(rejected)}"
    )

    report.append(
        f"Final selected features: "
        f"{len(top_features)}"
    )

    report.append("")

    report.append(
        "LEAKAGE CONTROLS"
    )

    report.append(
        "- Patient-level train/test split."
    )

    report.append(
        "- Multiple annotations for one patient "
        "remain in the same split."
    )

    report.append(
        "- Missingness filtering uses TRAIN only."
    )

    report.append(
        "- Median imputation is fitted on TRAIN only."
    )

    report.append(
        "- Boruta is fitted on TRAIN only."
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    print(
        f"Report: {OUTPUT_REPORT}"
    )


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # A. Labels from Center1-4
    # --------------------------------------------------------

    labels = load_all_labels()

    # --------------------------------------------------------
    # B. Patient-level labels
    # --------------------------------------------------------

    patient_table = (
        build_patient_label_table(
            labels
        )
    )

    # --------------------------------------------------------
    # C. Radiomics + labels
    # --------------------------------------------------------

    df = prepare_radiomics(
        INPUT_CSV,
        labels
    )

    # --------------------------------------------------------
    # D. Patient-level split
    # --------------------------------------------------------

    (
        df,
        train_ids,
        test_ids,
        split_patient_table
    ) = patient_level_split(
        df,
        patient_table
    )

    # --------------------------------------------------------
    # E. Numeric radiomics
    # --------------------------------------------------------

    X = build_feature_matrix(
        df
    )

    # --------------------------------------------------------
    # F. TRAIN-only preprocessing
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) = preprocess(
        df,
        X
    )

    # --------------------------------------------------------
    # F2. Export TRAIN / TEST BEFORE Boruta selection
    # --------------------------------------------------------

    (
        train_pre_boruta_df,
        test_pre_boruta_df
    ) = save_pre_boruta_train_test(
        df,
        X_train,
        X_test,
        feature_names
    )

    # --------------------------------------------------------
    # G. TRAIN-only Boruta
    # --------------------------------------------------------

    (
        boruta,
        ranking,
        confirmed,
        tentative,
        rejected,
        top_features
    ) = run_boruta(
        X_train,
        y_train,
        feature_names
    )

    # --------------------------------------------------------
    # H. Save
    # --------------------------------------------------------

    save_outputs(
        df,
        X_train,
        X_test,
        feature_names,
        top_features,
        ranking,
        split_patient_table
    )

    # --------------------------------------------------------
    # I. Report
    # --------------------------------------------------------

    save_report(
        patient_table,
        split_patient_table,
        df,
        confirmed,
        tentative,
        rejected,
        top_features
    )

    section(
        "BORUTA PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "\nFinal selected features:"
    )

    for i, feature in enumerate(
        top_features,
        start=1
    ):

        print(
            f"{i:02d}. {feature}"
        )

    print(
        "\nIMPORTANT FOR THE SECOND CONDITION:"
    )

    print(
        "Run this script again using the "
        "NON-HARMONIZED CSV."
    )

    print(
        "For a fair comparison, reuse the SAME "
        "patient_train_test_split.csv generated "
        "from the first run rather than creating "
        "a new random split."
    )

# ============================================================
# 16. BORUTA FEATURE SELECTION STABILITY ANALYSIS
# ============================================================
#
# PURPOSE:
#   Evaluate whether Boruta-selected features are stable across
#   different training folds.
#
# IMPORTANT:
#   - The original code ABOVE is NOT modified.
#   - The original patient-level train/test split is preserved.
#   - TEST data is NEVER used here.
#   - Boruta is repeatedly fitted only on subsets of TRAIN.
#   - Imputation is fitted separately inside each fold.
#
# OUTPUT:
#   1. boruta_stability_frequency.csv
#   2. boruta_stability_jaccard.csv
#   3. boruta_stability_top30.csv
#   4. boruta_stability_report.txt
#
# ============================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from itertools import combinations


# ============================================================
# 16.1 CONFIGURATION
# ============================================================

STABILITY_N_SPLITS = 5

STABILITY_RANDOM_STATE = 42

# Number of Boruta iterations for each fold.
#
# The original Boruta uses BORUTA_MAX_ITER = 100.
# We keep the same setting for methodological consistency.
STABILITY_BORUTA_MAX_ITER = BORUTA_MAX_ITER

# Same Boruta threshold as the original analysis.
STABILITY_BORUTA_PERC = BORUTA_PERC

# Same RF configuration as the original Boruta.
STABILITY_RF_ESTIMATORS = RF_N_ESTIMATORS
STABILITY_RF_MAX_DEPTH = RF_MAX_DEPTH

# Number of final stable features to report.
STABILITY_TOP_N = TOP_N_FEATURES


# ============================================================
# 16.2 OUTPUT FILES
# ============================================================

OUTPUT_STABILITY_FREQUENCY = (
    OUTPUT_DIR / "boruta_stability_frequency.csv"
)

OUTPUT_STABILITY_JACCARD = (
    OUTPUT_DIR / "boruta_stability_jaccard.csv"
)

OUTPUT_STABILITY_TOP30 = (
    OUTPUT_DIR / "boruta_stability_top30.csv"
)

OUTPUT_STABILITY_REPORT = (
    OUTPUT_DIR / "boruta_stability_report.txt"
)


# ============================================================
# 16.3 FUNCTION:
#       RUN BORUTA ON ONE FOLD
# ============================================================

def run_boruta_stability_fold(
    X_fold_train,
    y_fold_train,
    feature_names,
    fold_number
):

    print("\n" + "-" * 70)
    print(
        f"BORUTA STABILITY - FOLD {fold_number}"
    )
    print("-" * 70)

    print(
        f"Fold training samples: "
        f"{len(X_fold_train)}"
    )

    # --------------------------------------------------------
    # Imputation is fitted ONLY on this fold's training data.
    # --------------------------------------------------------

    fold_imputer = SimpleImputer(
        strategy="median"
    )

    X_fold_train_imp = fold_imputer.fit_transform(
        X_fold_train
    )

    print(
        "✓ Median imputation fitted on fold training data."
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=STABILITY_RF_ESTIMATORS,
        max_depth=STABILITY_RF_MAX_DEPTH,
        random_state=(
            STABILITY_RANDOM_STATE + fold_number
        ),
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Boruta
    # --------------------------------------------------------

    boruta_fold = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=STABILITY_BORUTA_MAX_ITER,
        perc=STABILITY_BORUTA_PERC,
        random_state=(
            STABILITY_RANDOM_STATE + fold_number
        ),
        verbose=0
    )

    boruta_fold.fit(
        X_fold_train_imp,
        np.asarray(y_fold_train)
    )

    # --------------------------------------------------------
    # Extract confirmed features
    # --------------------------------------------------------

    confirmed_fold = [
        feature_names[i]
        for i in range(len(feature_names))
        if boruta_fold.support_[i]
    ]

    tentative_fold = [
        feature_names[i]
        for i in range(len(feature_names))
        if boruta_fold.support_weak_[i]
    ]

    print(
        f"Confirmed features : "
        f"{len(confirmed_fold)}"
    )

    print(
        f"Tentative features : "
        f"{len(tentative_fold)}"
    )

    print(
        f"✓ Fold {fold_number} completed."
    )

    return (
        confirmed_fold,
        tentative_fold
    )


# ============================================================
# 16.4 MAIN STABILITY ANALYSIS
# ============================================================

section(
    "9. BORUTA FEATURE SELECTION STABILITY ANALYSIS"
)

print(
    "\nThis analysis evaluates whether Boruta-selected "
    "features remain stable across training folds."
)

print(
    "\nIMPORTANT:"
)

print(
    "✓ Only TRAINING patients are used."
)

print(
    "✓ HELD-OUT TEST patients are NOT used."
)

print(
    "✓ Patient-level split from the original analysis is preserved."
)

print(
    "✓ Imputation is fitted separately inside each fold."
)

print(
    f"✓ Number of folds: {STABILITY_N_SPLITS}"
)

print(
    f"✓ Boruta iterations per fold: "
    f"{STABILITY_BORUTA_MAX_ITER}"
)


# ============================================================
# 16.5 PREPARE TRAINING DATA
# ============================================================

# X_train from the original pipeline is already TRAIN-only.
#
# Convert to DataFrame so that indexing by fold positions
# is unambiguous.

X_train_stability = pd.DataFrame(
    X_train,
    columns=feature_names
)

y_train_stability = pd.Series(
    np.asarray(y_train)
).reset_index(drop=True)

X_train_stability = (
    X_train_stability.reset_index(drop=True)
)


print("\nTraining data for stability analysis:")
print(
    f"  Samples  : {len(X_train_stability)}"
)

print(
    f"  Features : {len(feature_names)}"
)


# ============================================================
# 16.6 STRATIFIED K-FOLD
# ============================================================

stability_cv = StratifiedKFold(
    n_splits=STABILITY_N_SPLITS,
    shuffle=True,
    random_state=STABILITY_RANDOM_STATE
)


fold_selected_features = {}

fold_tentative_features = {}


# ============================================================
# 16.7 RUN BORUTA FOR EACH FOLD
# ============================================================

for fold_number, (
    fold_train_idx,
    fold_validation_idx
) in enumerate(
    stability_cv.split(
        X_train_stability,
        y_train_stability
    ),
    start=1
):

    X_fold_train = (
        X_train_stability
        .iloc[fold_train_idx]
        .copy()
    )

    y_fold_train = (
        y_train_stability
        .iloc[fold_train_idx]
        .copy()
    )

    (
        confirmed_fold,
        tentative_fold
    ) = run_boruta_stability_fold(
        X_fold_train,
        y_fold_train,
        feature_names,
        fold_number
    )

    fold_selected_features[
        fold_number
    ] = set(
        confirmed_fold
    )

    fold_tentative_features[
        fold_number
    ] = set(
        tentative_fold
    )


# ============================================================
# 16.8 FEATURE SELECTION FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print(
    "FEATURE SELECTION FREQUENCY"
)
print("=" * 70)


feature_frequency_rows = []

for feature in feature_names:

    selected_folds = [
        fold_number
        for fold_number, selected_set
        in fold_selected_features.items()
        if feature in selected_set
    ]

    frequency = len(
        selected_folds
    )

    frequency_percent = (
        frequency
        / STABILITY_N_SPLITS
        * 100
    )

    feature_frequency_rows.append(
        {
            "Feature": feature,
            "Selected_Folds": frequency,
            "Selection_Frequency": frequency_percent,
            "Fold_1": int(
                feature
                in fold_selected_features[1]
            ),
            "Fold_2": int(
                feature
                in fold_selected_features[2]
            ),
            "Fold_3": int(
                feature
                in fold_selected_features[3]
            ),
            "Fold_4": int(
                feature
                in fold_selected_features[4]
            ),
            "Fold_5": int(
                feature
                in fold_selected_features[5]
            )
        }
    )


stability_frequency_df = pd.DataFrame(
    feature_frequency_rows
)


# ------------------------------------------------------------
# Sort:
#   1. Frequency descending
#   2. Feature name
# ------------------------------------------------------------

stability_frequency_df = (
    stability_frequency_df
    .sort_values(
        [
            "Selected_Folds",
            "Feature"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Add stability category
# ------------------------------------------------------------

def stability_category(freq):

    if freq == 100:
        return "Very Stable"

    elif freq >= 80:
        return "Stable"

    elif freq >= 60:
        return "Moderately Stable"

    elif freq >= 40:
        return "Low Stability"

    else:
        return "Unstable"


stability_frequency_df[
    "Stability_Category"
] = (
    stability_frequency_df[
        "Selection_Frequency"
    ]
    .apply(
        stability_category
    )
)


# ============================================================
# 16.9 SAVE FREQUENCY TABLE
# ============================================================

stability_frequency_df.to_csv(
    OUTPUT_STABILITY_FREQUENCY,
    index=False
)


print(
    f"\n✓ Saved feature frequency:"
)

print(
    f"  {OUTPUT_STABILITY_FREQUENCY}"
)


# ============================================================
# 16.10 JACCARD SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print(
    "JACCARD SIMILARITY BETWEEN FOLDS"
)
print("=" * 70)


def calculate_jaccard(
    set_a,
    set_b
):

    union = set_a.union(
        set_b
    )

    if len(union) == 0:
        return 1.0

    intersection = set_a.intersection(
        set_b
    )

    return (
        len(intersection)
        / len(union)
    )


jaccard_rows = []

for fold_a, fold_b in combinations(
    range(
        1,
        STABILITY_N_SPLITS + 1
    ),
    2
):

    selected_a = (
        fold_selected_features[
            fold_a
        ]
    )

    selected_b = (
        fold_selected_features[
            fold_b
        ]
    )

    jaccard_score = calculate_jaccard(
        selected_a,
        selected_b
    )

    jaccard_rows.append(
        {
            "Fold_A": fold_a,
            "Fold_B": fold_b,
            "Features_Fold_A": len(
                selected_a
            ),
            "Features_Fold_B": len(
                selected_b
            ),
            "Intersection": len(
                selected_a.intersection(
                    selected_b
                )
            ),
            "Union": len(
                selected_a.union(
                    selected_b
                )
            ),
            "Jaccard_Similarity": (
                jaccard_score
            )
        }
    )


jaccard_df = pd.DataFrame(
    jaccard_rows
)


# ============================================================
# 16.11 MEAN JACCARD
# ============================================================

mean_jaccard = (
    jaccard_df[
        "Jaccard_Similarity"
    ]
    .mean()
)

std_jaccard = (
    jaccard_df[
        "Jaccard_Similarity"
    ]
    .std()
)


print(
    "\nPairwise Jaccard similarity:"
)

print(
    jaccard_df.to_string(
        index=False
    )
)

print(
    f"\nMean Jaccard similarity : "
    f"{mean_jaccard:.4f}"
)

print(
    f"Std Jaccard similarity  : "
    f"{std_jaccard:.4f}"
)


# ============================================================
# 16.12 SAVE JACCARD TABLE
# ============================================================

jaccard_df.to_csv(
    OUTPUT_STABILITY_JACCARD,
    index=False
)

print(
    f"\n✓ Saved Jaccard results:"
)

print(
    f"  {OUTPUT_STABILITY_JACCARD}"
)


# ============================================================
# 16.13 STABILITY-BASED TOP 30
# ============================================================
#
# IMPORTANT:
# This does NOT replace the original Top 30.
#
# It creates a separate stability-based ranking that can be
# compared against the original Boruta Top 30.
#
# Ranking priority:
#
#   1. Selection frequency
#   2. Original Boruta rank
#   3. Feature name
#
# ============================================================

print("\n" + "=" * 70)
print(
    "STABILITY-BASED TOP 30 FEATURES"
)
print("=" * 70)


# Merge original Boruta ranking if available.

stability_rank_df = (
    stability_frequency_df[
        [
            "Feature",
            "Selected_Folds",
            "Selection_Frequency",
            "Stability_Category"
        ]
    ]
    .copy()
)


# Add original Boruta rank.

original_rank_lookup = (
    ranking[
        [
            "Feature",
            "Boruta_Rank",
            "Confirmed",
            "Top30"
        ]
    ]
    .copy()
)


stability_rank_df = stability_rank_df.merge(
    original_rank_lookup,
    on="Feature",
    how="left"
)


# Sort by:
#   1. Frequency
#   2. Original Boruta rank
#   3. Feature name

stability_rank_df = (
    stability_rank_df
    .sort_values(
        [
            "Selection_Frequency",
            "Boruta_Rank",
            "Feature"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )
    .reset_index(drop=True)
)


stable_top30 = (
    stability_rank_df
    .head(STABILITY_TOP_N)
    .copy()
)


stable_top30[
    "Stability_Top30"
] = True


stable_top30[
    "Stability_Rank"
] = np.arange(
    1,
    len(stable_top30) + 1
)


print(
    "\nTop stable features:"
)

print(
    stable_top30[
        [
            "Stability_Rank",
            "Feature",
            "Selected_Folds",
            "Selection_Frequency",
            "Boruta_Rank",
            "Confirmed",
            "Top30"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 16.14 SAVE STABILITY TOP 30
# ============================================================

stable_top30.to_csv(
    OUTPUT_STABILITY_TOP30,
    index=False
)

print(
    f"\n✓ Saved stability-based Top 30:"
)

print(
    f"  {OUTPUT_STABILITY_TOP30}"
)


# ============================================================
# 16.15 OVERLAP WITH ORIGINAL TOP 30
# ============================================================

original_top30_set = set(
    top_features
)

stable_top30_set = set(
    stable_top30[
        "Feature"
    ]
)

overlap_top30 = (
    original_top30_set
    .intersection(
        stable_top30_set
    )
)

jaccard_top30 = calculate_jaccard(
    original_top30_set,
    stable_top30_set
)


print("\n" + "=" * 70)
print(
    "ORIGINAL TOP 30 vs STABILITY TOP 30"
)
print("=" * 70)

print(
    f"Original Top 30             : "
    f"{len(original_top30_set)}"
)

print(
    f"Stability-based Top 30      : "
    f"{len(stable_top30_set)}"
)

print(
    f"Overlapping features        : "
    f"{len(overlap_top30)}"
)

print(
    f"Top-30 Jaccard similarity   : "
    f"{jaccard_top30:.4f}"
)


# ============================================================
# 16.16 VERY STABLE FEATURES
# ============================================================

very_stable = (
    stability_frequency_df[
        stability_frequency_df[
            "Selection_Frequency"
        ] >= 80
    ]
    .copy()
)


print("\n" + "-" * 70)

print(
    "FEATURES SELECTED IN >=80% OF FOLDS"
)

print("-" * 70)

print(
    f"Number of stable features: "
    f"{len(very_stable)}"
)

if len(very_stable) > 0:

    print(
        very_stable[
            [
                "Feature",
                "Selected_Folds",
                "Selection_Frequency",
                "Stability_Category"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No feature reached the >=80% stability threshold."
    )


# ============================================================
# 16.17 FOLD SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    "BORUTA FOLD SUMMARY"
)
print("=" * 70)


fold_summary_rows = []

for fold_number in range(
    1,
    STABILITY_N_SPLITS + 1
):

    selected = (
        fold_selected_features[
            fold_number
        ]
    )

    tentative = (
        fold_tentative_features[
            fold_number
        ]
    )

    fold_summary_rows.append(
        {
            "Fold": fold_number,
            "Confirmed": len(
                selected
            ),
            "Tentative": len(
                tentative
            )
        }
    )


fold_summary_df = pd.DataFrame(
    fold_summary_rows
)

print(
    fold_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 16.18 STABILITY REPORT
# ============================================================

stability_report = []

stability_report.append(
    "BORUTA FEATURE SELECTION STABILITY REPORT"
)

stability_report.append(
    "=" * 70
)

stability_report.append(
    ""
)

stability_report.append(
    "PURPOSE"
)

stability_report.append(
    "Evaluate the stability of Boruta-selected radiomics "
    "features across stratified training folds."
)

stability_report.append(
    ""
)

stability_report.append(
    "DATA / LEAKAGE CONTROL"
)

stability_report.append(
    f"Training patients: {len(y_train_stability)}"
)

stability_report.append(
    f"Held-out test patients: {len(y_test)}"
)

stability_report.append(
    "Held-out test data was NOT used in this stability analysis."
)

stability_report.append(
    "The original patient-level train/test split was preserved."
)

stability_report.append(
    "Median imputation was fitted separately inside each fold."
)

stability_report.append(
    ""
)

stability_report.append(
    "BORUTA CONFIGURATION"
)

stability_report.append(
    f"Number of folds: {STABILITY_N_SPLITS}"
)

stability_report.append(
    f"Boruta max iterations: "
    f"{STABILITY_BORUTA_MAX_ITER}"
)

stability_report.append(
    f"Boruta percentile threshold: "
    f"{STABILITY_BORUTA_PERC}"
)

stability_report.append(
    f"Random Forest estimators: "
    f"{STABILITY_RF_ESTIMATORS}"
)

stability_report.append(
    f"Random Forest max depth: "
    f"{STABILITY_RF_MAX_DEPTH}"
)

stability_report.append(
    ""
)

stability_report.append(
    "FOLD SUMMARY"
)

stability_report.append(
    fold_summary_df.to_string(
        index=False
    )
)

stability_report.append(
    ""
)

stability_report.append(
    "JACCARD SIMILARITY"
)

stability_report.append(
    f"Mean pairwise Jaccard: "
    f"{mean_jaccard:.4f}"
)

stability_report.append(
    f"Std pairwise Jaccard: "
    f"{std_jaccard:.4f}"
)

stability_report.append(
    ""
)

stability_report.append(
    "TOP 30 OVERLAP"
)

stability_report.append(
    f"Original Top 30: "
    f"{len(original_top30_set)}"
)

stability_report.append(
    f"Stability Top 30: "
    f"{len(stable_top30_set)}"
)

stability_report.append(
    f"Overlap: "
    f"{len(overlap_top30)}"
)

stability_report.append(
    f"Top-30 Jaccard: "
    f"{jaccard_top30:.4f}"
)

stability_report.append(
    ""
)

stability_report.append(
    "STABLE FEATURES"
)

stability_report.append(
    f"Features selected in >=80% of folds: "
    f"{len(very_stable)}"
)


with open(
    OUTPUT_STABILITY_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            stability_report
        )
    )


# ============================================================
# 16.19 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    "✓ BORUTA STABILITY ANALYSIS COMPLETED"
)
print("=" * 70)

print(
    f"\nMean Jaccard similarity : "
    f"{mean_jaccard:.4f}"
)

print(
    f"Stable features >=80%   : "
    f"{len(very_stable)}"
)

print(
    f"Original/Stable Top30 overlap : "
    f"{len(overlap_top30)}/30"
)

print(
    f"Top30 Jaccard similarity : "
    f"{jaccard_top30:.4f}"
)

print("\nFiles saved:")

print(
    f"  1. {OUTPUT_STABILITY_FREQUENCY}"
)

print(
    f"  2. {OUTPUT_STABILITY_JACCARD}"
)

print(
    f"  3. {OUTPUT_STABILITY_TOP30}"
)

print(
    f"  4. {OUTPUT_STABILITY_REPORT}"
)

print("\n" + "=" * 70)

# ======================================================================
# STABLE FEATURE SET ANALYSIS
# TOP-20 / TOP-30 / TOP-40
# ======================================================================
#
# IMPORTANT:
# - Uses ONLY training-fold Boruta stability results
# - Does NOT use TEST data
# - Does NOT rerun Boruta
# - Does NOT modify the original Boruta Top-30
# - Uses the variables that actually exist in this script:
#       fold_selected_features
#       top_features
#       ranking
#
# Ranking priority:
#   1. Selection frequency across folds
#   2. Original Boruta rank
#   3. Feature name
#
# ======================================================================

print("\n" + "=" * 70)
print("STABLE FEATURE SET ANALYSIS")
print("TOP-20 / TOP-30 / TOP-40")
print("=" * 70)


# ----------------------------------------------------------------------
# 1. CHECK VARIABLES
# ----------------------------------------------------------------------

required_variables = [
    "fold_selected_features",
    "top_features",
    "ranking"
]

missing_variables = [
    var
    for var in required_variables
    if var not in globals()
]

if missing_variables:
    raise NameError(
        "Required variables not found:\n"
        + "\n".join(missing_variables)
        + "\n\n"
        "Make sure the original Boruta and "
        "Boruta stability analysis have completed first."
    )


# ----------------------------------------------------------------------
# 2. CHECK NUMBER OF FOLDS
# ----------------------------------------------------------------------

n_folds = len(fold_selected_features)

print(
    f"\nNumber of stability folds : {n_folds}"
)

if n_folds == 0:
    raise ValueError(
        "fold_selected_features is empty."
    )


# ----------------------------------------------------------------------
# 3. CALCULATE FEATURE SELECTION FREQUENCY
# ----------------------------------------------------------------------

feature_frequency = {}

for fold_number, selected_set in fold_selected_features.items():

    for feature in selected_set:

        if feature not in feature_frequency:
            feature_frequency[feature] = 0

        feature_frequency[feature] += 1


# ----------------------------------------------------------------------
# 4. CREATE STABILITY DATAFRAME
# ----------------------------------------------------------------------

stability_records = []

for feature, frequency in feature_frequency.items():

    frequency_percent = (
        frequency / n_folds
    ) * 100

    stability_records.append(
        {
            "Feature": feature,
            "Selected_Folds": frequency,
            "Selection_Frequency_%": frequency_percent
        }
    )


stable_df = pd.DataFrame(
    stability_records
)


# ----------------------------------------------------------------------
# 5. ADD ORIGINAL BORUTA RANK
# ----------------------------------------------------------------------

boruta_rank_columns = [
    "Feature",
    "Boruta_Rank"
]

boruta_rank_df = (
    ranking[
        boruta_rank_columns
    ]
    .drop_duplicates(
        "Feature"
    )
)


stable_df = stable_df.merge(
    boruta_rank_df,
    on="Feature",
    how="left"
)


# ----------------------------------------------------------------------
# 6. SORT STABILITY RANKING
# ----------------------------------------------------------------------
#
# Priority:
#   1. Higher selection frequency
#   2. Better original Boruta rank
#   3. Feature name
#
# ----------------------------------------------------------------------

stable_df = (
    stable_df
    .sort_values(
        by=[
            "Selection_Frequency_%",
            "Boruta_Rank",
            "Feature"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )
    .reset_index(drop=True)
)


stable_df.insert(
    0,
    "Stability_Rank",
    np.arange(
        1,
        len(stable_df) + 1
    )
)


# ----------------------------------------------------------------------
# 7. DISPLAY COMPLETE STABILITY RANKING
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("STABILITY RANKING")
print("-" * 70)

print(
    stable_df[
        [
            "Stability_Rank",
            "Feature",
            "Selected_Folds",
            "Selection_Frequency_%",
            "Boruta_Rank"
        ]
    ].to_string(index=False)
)


# ----------------------------------------------------------------------
# 8. CREATE TOP-20 / TOP-30 / TOP-40
# ----------------------------------------------------------------------

stable_top20 = (
    stable_df
    .head(20)
    .copy()
)

stable_top30 = (
    stable_df
    .head(30)
    .copy()
)

stable_top40 = (
    stable_df
    .head(40)
    .copy()
)


# ----------------------------------------------------------------------
# 9. DISPLAY TOP-20
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STABILITY-BASED TOP-20")
print("=" * 70)

print(
    stable_top20[
        [
            "Stability_Rank",
            "Feature",
            "Selected_Folds",
            "Selection_Frequency_%",
            "Boruta_Rank"
        ]
    ].to_string(index=False)
)


# ----------------------------------------------------------------------
# 10. DISPLAY TOP-30
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STABILITY-BASED TOP-30")
print("=" * 70)

print(
    stable_top30[
        [
            "Stability_Rank",
            "Feature",
            "Selected_Folds",
            "Selection_Frequency_%",
            "Boruta_Rank"
        ]
    ].to_string(index=False)
)


# ----------------------------------------------------------------------
# 11. DISPLAY TOP-40
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STABILITY-BASED TOP-40")
print("=" * 70)

print(
    stable_top40[
        [
            "Stability_Rank",
            "Feature",
            "Selected_Folds",
            "Selection_Frequency_%",
            "Boruta_Rank"
        ]
    ].to_string(index=False)
)


# ----------------------------------------------------------------------
# 12. COMPARE WITH ORIGINAL BORUTA TOP-30
# ----------------------------------------------------------------------

original_top30 = set(
    top_features[:30]
)

stable_top20_set = set(
    stable_top20["Feature"]
)

stable_top30_set = set(
    stable_top30["Feature"]
)

stable_top40_set = set(
    stable_top40["Feature"]
)


overlap_top20 = (
    original_top30 &
    stable_top20_set
)

overlap_top30 = (
    original_top30 &
    stable_top30_set
)

overlap_top40 = (
    original_top30 &
    stable_top40_set
)


print("\n" + "=" * 70)
print("OVERLAP WITH ORIGINAL BORUTA TOP-30")
print("=" * 70)

print(
    f"\nOriginal Top-30 vs Stable Top-20 : "
    f"{len(overlap_top20)}/20"
)

print(
    f"Original Top-30 vs Stable Top-30 : "
    f"{len(overlap_top30)}/30"
)

print(
    f"Original Top-30 vs Stable Top-40 : "
    f"{len(overlap_top40)}/30"
)


# ----------------------------------------------------------------------
# 13. JACCARD BETWEEN ORIGINAL TOP-30 AND STABLE SETS
# ----------------------------------------------------------------------

def jaccard_similarity(
    set_a,
    set_b
):

    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    intersection = set_a & set_b

    return (
        len(intersection) /
        len(union)
    )


jaccard_top20 = jaccard_similarity(
    original_top30,
    stable_top20_set
)

jaccard_top30 = jaccard_similarity(
    original_top30,
    stable_top30_set
)

jaccard_top40 = jaccard_similarity(
    original_top30,
    stable_top40_set
)


print("\n" + "-" * 70)
print("JACCARD SIMILARITY")
print("-" * 70)

print(
    f"Original Top-30 vs Stable Top-20 : "
    f"{jaccard_top20:.4f}"
)

print(
    f"Original Top-30 vs Stable Top-30 : "
    f"{jaccard_top30:.4f}"
)

print(
    f"Original Top-30 vs Stable Top-40 : "
    f"{jaccard_top40:.4f}"
)


# ----------------------------------------------------------------------
# 14. SAVE TOP FEATURE SETS
# ----------------------------------------------------------------------

stable_top20.to_csv(
    "stable_top20_features.csv",
    index=False
)

stable_top30.to_csv(
    "stable_top30_features.csv",
    index=False
)

stable_top40.to_csv(
    "stable_top40_features.csv",
    index=False
)

stable_df.to_csv(
    "all_stable_feature_ranking.csv",
    index=False
)


# ----------------------------------------------------------------------
# 15. SAVE FEATURE NAME LISTS
# ----------------------------------------------------------------------

pd.DataFrame({
    "Feature": stable_top20["Feature"]
}).to_csv(
    "stable_top20_feature_names.csv",
    index=False
)

pd.DataFrame({
    "Feature": stable_top30["Feature"]
}).to_csv(
    "stable_top30_feature_names.csv",
    index=False
)

pd.DataFrame({
    "Feature": stable_top40["Feature"]
}).to_csv(
    "stable_top40_feature_names.csv",
    index=False
)


# ----------------------------------------------------------------------
# 16. FINAL SUMMARY
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("✓ STABLE FEATURE SET ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"\nOriginal Boruta Top-30 : "
    f"{len(original_top30)} features"
)

print(
    f"Stable Top-20          : "
    f"{len(stable_top20)} features"
)

print(
    f"Stable Top-30          : "
    f"{len(stable_top30)} features"
)

print(
    f"Stable Top-40          : "
    f"{len(stable_top40)} features"
)

print("\nFiles saved:")
print("  1. all_stable_feature_ranking.csv")
print("  2. stable_top20_features.csv")
print("  3. stable_top30_features.csv")
print("  4. stable_top40_features.csv")
print("  5. stable_top20_feature_names.csv")
print("  6. stable_top30_feature_names.csv")
print("  7. stable_top40_feature_names.csv")

print("\n" + "=" * 70)

# ============================================================
# 10. VERIFY TRAIN-ONLY STABILITY
#     + EXPORT STABLE TOP-20 / TOP-30 / TOP-40
# ============================================================

print("\n" + "=" * 70)
print("10. VERIFY TRAIN-ONLY STABILITY + EXPORT STABLE FEATURE SETS")
print("=" * 70)


# ============================================================
# A. VERIFY REQUIRED VARIABLES
# ============================================================

print("\n" + "-" * 70)
print("A. VERIFY REQUIRED VARIABLES")
print("-" * 70)

required_variables = [
    "df",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "feature_names",
    "fold_selected_features",
    "stable_df",
    "train_ids",
    "test_ids"
]

missing_variables = [
    var
    for var in required_variables
    if var not in globals()
]

if missing_variables:

    raise NameError(
        "Required variables tidak ditemukan:\n"
        + "\n".join(missing_variables)
        + "\n\n"
        "Pastikan original Boruta pipeline dan "
        "stability analysis sudah selesai."
    )

print("✓ Semua variabel yang diperlukan tersedia.")


# ============================================================
# B. RECONSTRUCT TRAIN / TEST DATAFRAME
#    FROM ORIGINAL PATIENT-LEVEL SPLIT
# ============================================================

print("\n" + "-" * 70)
print("B. RECONSTRUCT TRAIN / TEST DATA")
print("-" * 70)

train_mask = (
    df["Split"] == "train"
)

test_mask = (
    df["Split"] == "test"
)


# ------------------------------------------------------------
# Metadata columns
# ------------------------------------------------------------

metadata_columns = [
    "PatientID",
    "label",
    "Center",
    "Split",
    "Condition",
    "MaskIndex"
]

train_metadata_columns = [
    col
    for col in metadata_columns
    if col in df.columns
]

test_metadata_columns = [
    col
    for col in metadata_columns
    if col in df.columns
]


# ------------------------------------------------------------
# TRAIN DATAFRAME
# ------------------------------------------------------------

train_metadata = (
    df.loc[
        train_mask,
        train_metadata_columns
    ]
    .reset_index(drop=True)
)

train_features_df = pd.DataFrame(
    X_train,
    columns=feature_names
).reset_index(drop=True)

train_df = pd.concat(
    [
        train_metadata,
        train_features_df
    ],
    axis=1
)


# ------------------------------------------------------------
# TEST DATAFRAME
# ------------------------------------------------------------

test_metadata = (
    df.loc[
        test_mask,
        test_metadata_columns
    ]
    .reset_index(drop=True)
)

test_features_df = pd.DataFrame(
    X_test,
    columns=feature_names
).reset_index(drop=True)

test_df = pd.concat(
    [
        test_metadata,
        test_features_df
    ],
    axis=1
)


print(
    f"TRAIN shape : {train_df.shape}"
)

print(
    f"TEST shape  : {test_df.shape}"
)


# ============================================================
# C. PATIENT-LEVEL VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("C. PATIENT-LEVEL VERIFICATION")
print("-" * 70)

train_ids_verify = set(
    train_df["PatientID"].astype(str)
)

test_ids_verify = set(
    test_df["PatientID"].astype(str)
)


print(
    f"TRAIN patients : "
    f"{len(train_ids_verify)}"
)

print(
    f"TEST patients  : "
    f"{len(test_ids_verify)}"
)


# ------------------------------------------------------------
# Verify expected patient counts
# ------------------------------------------------------------

EXPECTED_TRAIN_PATIENTS = 171
EXPECTED_TEST_PATIENTS = 44

if len(train_ids_verify) != EXPECTED_TRAIN_PATIENTS:

    raise ValueError(
        "❌ TRAIN patient count salah.\n"
        f"Expected : {EXPECTED_TRAIN_PATIENTS}\n"
        f"Found    : {len(train_ids_verify)}"
    )


if len(test_ids_verify) != EXPECTED_TEST_PATIENTS:

    raise ValueError(
        "❌ TEST patient count salah.\n"
        f"Expected : {EXPECTED_TEST_PATIENTS}\n"
        f"Found    : {len(test_ids_verify)}"
    )


print(
    "✓ TRAIN patient count = 171"
)

print(
    "✓ TEST patient count = 44"
)


# ------------------------------------------------------------
# Patient overlap
# ------------------------------------------------------------

patient_overlap = (
    train_ids_verify
    .intersection(
        test_ids_verify
    )
)

if patient_overlap:

    raise ValueError(
        "❌ PATIENT LEAKAGE DETECTED!\n"
        f"Overlapping PatientID: "
        f"{sorted(patient_overlap)}"
    )


print(
    "✓ No PatientID overlap"
)

print(
    "✓ Patient-level independence verified"
)


# ============================================================
# D. VERIFY STABILITY ANALYSIS IS TRAIN-ONLY
# ============================================================

print("\n" + "-" * 70)
print("D. TRAIN-ONLY STABILITY VERIFICATION")
print("-" * 70)

print(
    "Stability analysis source:"
)

print(
    "  ✓ X_train only"
)

print(
    "  ✓ Stratified 5-fold performed within TRAIN"
)

print(
    "  ✓ TEST data was NOT used for Boruta stability"
)

print(
    f"  ✓ TRAIN patients : "
    f"{len(train_ids_verify)}"
)

print(
    f"  ✓ TEST patients  : "
    f"{len(test_ids_verify)}"
)


# ------------------------------------------------------------
# Verify number of stability folds
# ------------------------------------------------------------

n_stability_folds = (
    len(fold_selected_features)
)

print(
    f"  ✓ Stability folds : "
    f"{n_stability_folds}"
)

if n_stability_folds != STABILITY_N_SPLITS:

    raise ValueError(
        "Jumlah stability folds tidak sesuai "
        f"dengan STABILITY_N_SPLITS = "
        f"{STABILITY_N_SPLITS}"
    )


# ============================================================
# E. VERIFY STABILITY RANKING
# ============================================================

print("\n" + "-" * 70)
print("E. VERIFY STABILITY RANKING")
print("-" * 70)

# IMPORTANT:
# The actual stability ranking generated earlier
# is called stable_df.
#
# Do NOT use stability_ranking_df because that
# variable does not exist in this script.

stable_ranking = (
    stable_df.copy()
)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_stability_columns = [
    "Feature",
    "Selected_Folds",
    "Selection_Frequency_%",
    "Boruta_Rank"
]

missing_stability_columns = [
    col
    for col in required_stability_columns
    if col not in stable_ranking.columns
]

if missing_stability_columns:

    raise ValueError(
        "Kolom stability ranking tidak lengkap:\n"
        + "\n".join(
            missing_stability_columns
        )
    )


print(
    f"✓ Stability ranking contains "
    f"{len(stable_ranking)} features"
)


# ------------------------------------------------------------
# Verify features are actual radiomics features
# ------------------------------------------------------------

ranking_features = set(
    stable_ranking[
        "Feature"
    ].astype(str)
)

metadata_in_ranking = (
    ranking_features
    .intersection(
        metadata_columns
    )
)

if metadata_in_ranking:

    raise ValueError(
        "❌ Metadata columns ditemukan "
        "di stability ranking:\n"
        + "\n".join(
            sorted(metadata_in_ranking)
        )
    )


print(
    "✓ Stability ranking contains "
    "feature columns only"
)


# ============================================================
# F. CREATE STABLE TOP-20 / TOP-30 / TOP-40
# ============================================================

print("\n" + "-" * 70)
print("F. CREATE STABLE TOP-20 / TOP-30 / TOP-40")
print("-" * 70)


# stable_df sudah diurutkan sebelumnya berdasarkan:
#
# 1. Selection frequency DESC
# 2. Original Boruta rank ASC
# 3. Feature name ASC
#
# Oleh karena itu kita cukup mengambil head(N).

stable_top20 = (
    stable_ranking
    .head(20)
    .copy()
)

stable_top30 = (
    stable_ranking
    .head(30)
    .copy()
)

stable_top40 = (
    stable_ranking
    .head(40)
    .copy()
)


stable_feature_sets = {

    20: stable_top20[
        "Feature"
    ].tolist(),

    30: stable_top30[
        "Feature"
    ].tolist(),

    40: stable_top40[
        "Feature"
    ].tolist()

}


print(
    f"Stable Top-20 : "
    f"{len(stable_feature_sets[20])}"
)

print(
    f"Stable Top-30 : "
    f"{len(stable_feature_sets[30])}"
)

print(
    f"Stable Top-40 : "
    f"{len(stable_feature_sets[40])}"
)


# ============================================================
# G. VERIFY FEATURES EXIST IN TRAIN AND TEST
# ============================================================

print("\n" + "-" * 70)
print("G. FEATURE AVAILABILITY CHECK")
print("-" * 70)


for top_n, feature_list in (
    stable_feature_sets.items()
):

    missing_train = [
        feature
        for feature in feature_list
        if feature not in train_df.columns
    ]

    missing_test = [
        feature
        for feature in feature_list
        if feature not in test_df.columns
    ]


    if missing_train:

        raise ValueError(
            f"❌ Stable Top-{top_n}: "
            "feature missing from TRAIN:\n"
            + "\n".join(
                missing_train
            )
        )


    if missing_test:

        raise ValueError(
            f"❌ Stable Top-{top_n}: "
            "feature missing from TEST:\n"
            + "\n".join(
                missing_test
            )
        )


    print(
        f"✓ Stable Top-{top_n}: "
        f"{len(feature_list)} features "
        "available in TRAIN and TEST"
    )


# ============================================================
# H. EXPORT STABLE TRAIN / TEST DATASETS
# ============================================================

print("\n" + "-" * 70)
print("H. EXPORTING STABLE TRAIN / TEST DATASETS")
print("-" * 70)


exported_files = []


for top_n, feature_list in (
    stable_feature_sets.items()
):


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_output_columns = (
        train_metadata_columns
        + feature_list
    )


    train_stable_df = (
        train_df[
            train_output_columns
        ]
        .copy()
    )


    train_output_file = (
        OUTPUT_DIR
        / f"train_selected_stable_top"
        f"{top_n}.csv"
    )


    train_stable_df.to_csv(
        train_output_file,
        index=False
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_output_columns = (
        test_metadata_columns
        + feature_list
    )


    test_stable_df = (
        test_df[
            test_output_columns
        ]
        .copy()
    )


    test_output_file = (
        OUTPUT_DIR
        / f"test_selected_stable_top"
        f"{top_n}.csv"
    )


    test_stable_df.to_csv(
        test_output_file,
        index=False
    )


    exported_files.append(
        train_output_file
    )

    exported_files.append(
        test_output_file
    )


    print(
        f"\n✓ STABLE TOP-{top_n}"
    )

    print(
        f"  TRAIN : "
        f"{train_output_file}"
    )

    print(
        f"    Shape = "
        f"{train_stable_df.shape}"
    )

    print(
        f"  TEST  : "
        f"{test_output_file}"
    )

    print(
        f"    Shape = "
        f"{test_stable_df.shape}"
    )


# ============================================================
# I. FINAL PATIENT COUNT VERIFICATION
# ============================================================

print("\n" + "-" * 70)
print("I. FINAL PATIENT COUNT VERIFICATION")
print("-" * 70)


for top_n in [20, 30, 40]:

    train_file_check = (
        OUTPUT_DIR
        / f"train_selected_stable_top"
        f"{top_n}.csv"
    )

    test_file_check = (
        OUTPUT_DIR
        / f"test_selected_stable_top"
        f"{top_n}.csv"
    )


    train_check = pd.read_csv(
        train_file_check
    )

    test_check = pd.read_csv(
        test_file_check
    )


    train_unique = (
        train_check[
            "PatientID"
        ]
        .astype(str)
        .nunique()
    )

    test_unique = (
        test_check[
            "PatientID"
        ]
        .astype(str)
        .nunique()
    )


    overlap_check = (
        set(
            train_check[
                "PatientID"
            ].astype(str)
        )
        .intersection(
            set(
                test_check[
                    "PatientID"
                ].astype(str)
            )
        )
    )


    print(
        f"\nStable Top-{top_n}:"
    )

    print(
        f"  TRAIN patients = "
        f"{train_unique}"
    )

    print(
        f"  TEST patients  = "
        f"{test_unique}"
    )

    print(
        f"  Patient overlap = "
        f"{len(overlap_check)}"
    )


    if train_unique != len(
        train_ids_verify
    ):

        raise ValueError(
            f"❌ TRAIN patient count "
            f"berubah pada Stable Top-{top_n}"
        )


    if test_unique != len(
        test_ids_verify
    ):

        raise ValueError(
            f"❌ TEST patient count "
            f"berubah pada Stable Top-{top_n}"
        )


    if overlap_check:

        raise ValueError(
            f"❌ Patient leakage "
            f"pada Stable Top-{top_n}"
        )


    print(
        "  ✓ Patient counts preserved"
    )

    print(
        "  ✓ No TRAIN/TEST overlap"
    )


# ============================================================
# J. SAVE STABILITY-BASED FEATURE LISTS
# ============================================================

print("\n" + "-" * 70)
print("J. SAVING STABLE FEATURE LISTS")
print("-" * 70)


stable_ranking.to_csv(
    OUTPUT_DIR
    / "all_stable_feature_ranking.csv",
    index=False
)


stable_top20.to_csv(
    OUTPUT_DIR
    / "stable_top20_features.csv",
    index=False
)


stable_top30.to_csv(
    OUTPUT_DIR
    / "stable_top30_features.csv",
    index=False
)


stable_top40.to_csv(
    OUTPUT_DIR
    / "stable_top40_features.csv",
    index=False
)


# Feature-name-only files

pd.DataFrame({
    "Feature":
        stable_top20[
            "Feature"
        ]
}).to_csv(
    OUTPUT_DIR
    / "stable_top20_feature_names.csv",
    index=False
)


pd.DataFrame({
    "Feature":
        stable_top30[
            "Feature"
        ]
}).to_csv(
    OUTPUT_DIR
    / "stable_top30_feature_names.csv",
    index=False
)


pd.DataFrame({
    "Feature":
        stable_top40[
            "Feature"
        ]
}).to_csv(
    OUTPUT_DIR
    / "stable_top40_feature_names.csv",
    index=False
)


print(
    "✓ Stable feature ranking files saved."
)


# ============================================================
# K. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ STABLE FEATURE EXPORT COMPLETED")
print("=" * 70)


print("\nTRAIN-ONLY STABILITY:")

print(
    f"  ✓ Stability ranking generated "
    f"from {len(train_ids_verify)} TRAIN patients"
)

print(
    f"  ✓ Held-out TEST patients "
    f"({len(test_ids_verify)}) were NOT used "
    "for feature selection"
)

print(
    f"  ✓ Stability folds = "
    f"{n_stability_folds}"
)


print("\nSTABLE FEATURE SETS:")

print(
    "  ✓ Top-20"
)

print(
    "  ✓ Top-30"
)

print(
    "  ✓ Top-40"
)


print("\nOUTPUT FILES:")

for file_path in exported_files:

    print(
        f"  ✓ {file_path}"
    )


print("\nFINAL VERIFICATION:")

print(
    f"  ✓ TRAIN = "
    f"{len(train_ids_verify)} patients"
)

print(
    f"  ✓ TEST  = "
    f"{len(test_ids_verify)} patients"
)

print(
    "  ✓ No patient overlap"
)

print(
    "  ✓ Same stable feature definitions "
    "applied to TRAIN and TEST"
)

print(
    "  ✓ Feature stability calculated "
    "from TRAIN only"
)

print(
    "\n" + "=" * 70
)