# ------------------------------------------------------------
# IMPORTS
# Import the libraries used for file handling, counting values,
# numerical operations and dataset processing.
# ------------------------------------------------------------

import os
from collections import Counter

import numpy as np
import pandas as pd


#------------------------------------------------------------
# Reference used for locating the programme folder:
# Python Software Foundation, "os.path - Common pathname manipulations"
# URL: https://docs.python.org/3/library/os.path.html
#------------------------------------------------------------


# ------------------------------------------------------------
# PROGRAMME FOLDER LOCATION
# Store the folder containing this Python file. This allows the
# programme to locate dataset files and save output files reliably.
# ------------------------------------------------------------

PROGRAM_FOLDER = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# DATASET SOURCE
# The Adult Dataset files were supplied through the CPU5006-20
# Artificial Intelligence assessment area on the university's
# Ultra learning page.

# The supplied Index and old.adult.names files were examined using a
# separate documentation-inspection script. This confirmed the original
# record totals, attribute descriptions, missing-value rate and class
# distribution.
#
# These files contain supporting documentation and are not required
# during normal execution of the rule-based classifier.
#
# Files used directly for classification:
# - adult.data
# - adult.test
#
# Documentation file checked by the programme:
# - adult.names
#
# Additional supplied documentation retained in the project:
# - Index
# - old.adult.names
#
# The adult.names file was consulted as the dataset documentation
# and is checked by the programme to confirm that it is present.
# The classification task is to predict whether income is
# less than or equal to 50K or greater than 50K.
# ------------------------------------------------------------


# ------------------------------------------------------------
# ADULT DATASET COLUMN STRUCTURE
# Assign names to the 15 columns in the same order that their
# values appear in adult.data and adult.test.
# ------------------------------------------------------------

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


# Identify columns containing text values
TEXT_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]


# Identify columns containing numerical values
NUMERICAL_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# ------------------------------------------------------------
# Reference used for recursive file searching:
# Python Software Foundation, "os - Miscellaneous operating
# system interfaces"
# URL: https://docs.python.org/3/library/os.html
#------------------------------------------------------------


# ------------------------------------------------------------
# FILE SEARCH
# Search the project folder and its subdirectories for a required file.
# Return the complete file path when found, or None when not found.
# ------------------------------------------------------------

def find_file(filename):
    """Search for a required file inside the project folder."""

    for root, _, files in os.walk(PROGRAM_FOLDER):
        if filename in files:
            return os.path.join(root, filename)

    return None


# ------------------------------------------------------------
# REFERENCE FOR DATASET LOADING
# Source: pandas Development Team
# Title: pandas.read_csv
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# This documentation supported understanding of column-name
# assignment, skipped rows, initial-space handling and missing-value
# markers when loading the Adult Dataset files.
# ------------------------------------------------------------

# ------------------------------------------------------------
# DATASET LOADING AND INITIAL VALIDATION
# Locate the required Adult Dataset files, confirm that they exist,
# load the training and testing data and perform initial checks.
# ------------------------------------------------------------

def load_datasets():
    """Load and return the Adult training and testing datasets."""

    # Search for the required dataset files
    training_file = find_file("adult.data")
    testing_file = find_file("adult.test")
    information_file = find_file("adult.names")

    # Stop the programme if the training file cannot be found
    if training_file is None:
        raise FileNotFoundError(
            "adult.data was not found in the project folder "
            "or its subfolders."
        )

    # Stop the programme if the testing file cannot be found
    if testing_file is None:
        raise FileNotFoundError(
            "adult.test was not found in the project folder "
            "or its subfolders."
        )

    # Stop the programme if the documentation file cannot be found
    if information_file is None:
        raise FileNotFoundError(
            "adult.names was not found in the project folder "
            "or its subfolders."
        )

    # Display the location of each dataset file
    print(f"Training file found at: {training_file}")
    print(f"Testing file found at: {testing_file}")
    print(f"Information file found at: {information_file}")

    # Load the training data and assign the predefined column names
    training_data = pd.read_csv(
        training_file,
        names=COLUMN_NAMES,
        header=None,
        skipinitialspace=True,
        na_values="?"
    )

    # Load the testing data and skip its first documentation line
    testing_data = pd.read_csv(
        testing_file,
        names=COLUMN_NAMES,
        header=None,
        skiprows=1,
        skipinitialspace=True,
        na_values="?"
    )

    # Remove the full stop from the testing income labels
    testing_data["income"] = testing_data["income"].str.replace(
        ".",
        "",
        regex=False
    )

    # Confirm that both loaded datasets contain records
    if training_data.empty:
        raise ValueError(
            "The training dataset contains no records."
        )

    if testing_data.empty:
        raise ValueError(
            "The testing dataset contains no records."
        )

    # Display the dimensions and first five records of each dataset
    print("\nDataset loading completed successfully.")
    print(f"Training dataset shape: {training_data.shape}")
    print(f"Testing dataset shape: {testing_data.shape}")

    print("\nFirst five training records:")
    print(training_data.head())

    print("\nFirst five testing records:")
    print(testing_data.head())

    # Return both datasets to the main programme
    return training_data, testing_data


# ------------------------------------------------------------
# REFERENCES FOR DATA CLEANING AND MISSING VALUES
# Source: pandas Development Team
# Title: Working with missing data
# URL: https://pandas.pydata.org/docs/user_guide/missing_data.html
#
# Source: Codecademy
# Title: Cleaning and Transforming Columns
# URL: https://www.codecademy.com/learn/paths/getting-started-with-python-for-data-science/tracks/getting-started-with-python-for-data-science/modules/cleaning-and-transforming-columns/cheatsheet
#
# These resources supported understanding of missing-value detection,
# numerical conversion and incomplete-record removal.
# ------------------------------------------------------------

# ------------------------------------------------------------
# DATA CLEANING AND VALIDATION
# Standardise text and numerical values, handle missing data,
# inspect possible duplicates and validate the income labels.
# ------------------------------------------------------------

def clean_datasets(training_data, testing_data):
    """Clean, validate and return both datasets."""

    # Remove surrounding spaces from every text value
    for column in TEXT_COLUMNS:
        training_data[column] = training_data[column].str.strip()
        testing_data[column] = testing_data[column].str.strip()

    # Convert the expected numerical columns into numerical data types
    # and replace values that cannot be converted with missing markers
    for column in NUMERICAL_COLUMNS:
        training_data[column] = pd.to_numeric(
            training_data[column],
            errors="coerce"
        )

        testing_data[column] = pd.to_numeric(
            testing_data[column],
            errors="coerce"
        )

    # Replace invalid infinite values with missing-value markers
    training_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    testing_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Display the number of missing values in each column
    print("\nMissing values in the training dataset:")
    print(training_data.isna().sum())

    print("\nMissing values in the testing dataset:")
    print(testing_data.isna().sum())

    # Store the original sizes before removing incomplete records
    training_records_before = len(training_data)
    testing_records_before = len(testing_data)

    # Remove records containing one or more missing values
    training_data = training_data.dropna().copy()
    testing_data = testing_data.dropna().copy()

    # Calculate how many incomplete records were removed
    training_records_removed = (
        training_records_before - len(training_data)
    )

    testing_records_removed = (
        testing_records_before - len(testing_data)
    )

    # Count possible duplicate records without removing them
    # because the dataset does not contain a unique respondent ID
    training_duplicates = int(
        training_data.duplicated().sum()
    )

    testing_duplicates = int(
        testing_data.duplicated().sum()
    )

    # Reset the row indexes after incomplete records were removed
    training_data.reset_index(
        drop=True,
        inplace=True
    )

    testing_data.reset_index(
        drop=True,
        inplace=True
    )

    # Define the two permitted income classifications
    valid_income_values = [
        "<=50K",
        ">50K"
    ]

    # Identify records containing unexpected income labels
    invalid_training_income = training_data[
        ~training_data["income"].isin(valid_income_values)
    ]

    invalid_testing_income = testing_data[
        ~testing_data["income"].isin(valid_income_values)
    ]

    # Stop the programme if an invalid income label is found
    if not invalid_training_income.empty:
        raise ValueError(
            "The training dataset contains invalid income values."
        )

    if not invalid_testing_income.empty:
        raise ValueError(
            "The testing dataset contains invalid income values."
        )

    # Display the final cleaning results
    print("\nData cleaning completed successfully.")

    print(
        f"Training records before cleaning: "
        f"{training_records_before}"
    )

    print(
        f"Training records removed because of missing values: "
        f"{training_records_removed}"
    )

    print(
        f"Possible duplicate training records retained: "
        f"{training_duplicates}"
    )

    print(
        f"Training records remaining: "
        f"{len(training_data)}"
    )

    print(
        f"Testing records before cleaning: "
        f"{testing_records_before}"
    )

    print(
        f"Testing records removed because of missing values: "
        f"{testing_records_removed}"
    )

    print(
        f"Possible duplicate testing records retained: "
        f"{testing_duplicates}"
    )

    print(
        f"Testing records remaining: "
        f"{len(testing_data)}"
    )

    # Display the distribution of both income classes
    print("\nTraining income distribution:")
    print(training_data["income"].value_counts())

    print("\nTesting income distribution:")
    print(testing_data["income"].value_counts())

    # Confirm the training income totals using Counter
    training_income_counter = Counter(
        training_data["income"]
    )

    print("\nTraining income totals using Counter:")
    print(training_income_counter)

    print("\nThe datasets are clean and ready for analysis.")

    # Return the cleaned datasets to the main programme
    return training_data, testing_data

# ------------------------------------------------------------
# REFERENCES FOR ANALYSIS-GROUP CREATION
# Source: pandas Development Team
# Title: pandas.cut
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
#
# Source: NumPy Developers
# Title: numpy.where
# URL: https://numpy.org/doc/stable/reference/generated/numpy.where.html
#
# These references supported understanding of grouped numerical
# categories and conditional column creation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# ANALYSIS GROUP CREATION
# Create grouped and numerical attributes used to investigate
# patterns in the training and testing datasets.
# ------------------------------------------------------------

def add_analysis_groups(training_data, testing_data):
    """Add age, working-hours, income and capital-gain groups."""

    # Define the boundaries and labels for the age categories
    age_bins = [
        16,
        24,
        34,
        44,
        54,
        64,
        np.inf
    ]

    age_labels = [
        "17-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+"
    ]

    # Define the boundaries and labels for weekly working hours
    hours_bins = [
        0,
        19,
        29,
        39,
        40,
        49,
        59,
        np.inf
    ]

    hours_labels = [
        "1-19",
        "20-29",
        "30-39",
        "40",
        "41-49",
        "50-59",
        "60+"
    ]

    # Assign every training and testing record to an age category
    training_data["age_group"] = pd.cut(
        training_data["age"],
        bins=age_bins,
        labels=age_labels
    )

    testing_data["age_group"] = pd.cut(
        testing_data["age"],
        bins=age_bins,
        labels=age_labels
    )

    # Assign every record to a working-hours category
    training_data["hours_group"] = pd.cut(
        training_data["hours_per_week"],
        bins=hours_bins,
        labels=hours_labels
    )

    testing_data["hours_group"] = pd.cut(
        testing_data["hours_per_week"],
        bins=hours_bins,
        labels=hours_labels
    )

    # Convert the income class into a numerical analysis indicator
    # <=50K becomes 0 and >50K becomes 1
    training_data["high_income"] = (
        training_data["income"] == ">50K"
    ).astype(int)

    testing_data["high_income"] = (
        testing_data["income"] == ">50K"
    ).astype(int)

    # Record whether each person has a capital gain above zero
    training_data["has_capital_gain"] = np.where(
        training_data["capital_gain"] > 0,
        "Capital gain",
        "No capital gain"
    )

    testing_data["has_capital_gain"] = np.where(
        testing_data["capital_gain"] > 0,
        "Capital gain",
        "No capital gain"
    )

    # Return both datasets with the new analysis columns
    return training_data, testing_data
# ------------------------------------------------------------
# REFERENCES FOR GROUPED FEATURE ANALYSIS
# Source: pandas Development Team
# Title: pandas.DataFrame.groupby
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
#
# Source: GeeksforGeeks
# Title: Pandas GroupBy
# URL: https://www.geeksforgeeks.org/pandas/pandas-groupby/
#
# These resources supported understanding of grouping records and
# calculating aggregate totals, sums and means for each category.
# ------------------------------------------------------------


# ------------------------------------------------------------
# TRAINING-DATA PATTERN ANALYSIS
# Analyse how selected attributes relate to the income class.
# The first function performs the reusable calculation, while the
# second applies the calculation to each feature and prints the results.
# ------------------------------------------------------------

def analyse_feature(training_data, feature):
    """Calculate record totals and high-income rates for one feature."""

    # Group the training records by the selected feature and calculate
    # the total records, higher-income records and higher-income rate
    analysis = training_data.groupby(
        feature,
        observed=True
    ).agg(
        total_records=("income", "size"),
        high_income_records=("high_income", "sum"),
        high_income_rate=("high_income", "mean")
    )

    # Convert the higher-income rate into a percentage
    analysis["high_income_percentage"] = (
        analysis["high_income_rate"] * 100
    )

    # Return the categories from highest to lowest percentage
    return analysis.sort_values(
        by="high_income_percentage",
        ascending=False
    )


def analyse_training_patterns(training_data):
    """Calculate and display training-data income patterns."""

    # Analyse income patterns for each selected feature
    education_analysis = analyse_feature(
        training_data,
        "education"
    )

    occupation_analysis = analyse_feature(
        training_data,
        "occupation"
    )

    age_analysis = analyse_feature(
        training_data,
        "age_group"
    )

    hours_analysis = analyse_feature(
        training_data,
        "hours_group"
    )

    marital_status_analysis = analyse_feature(
        training_data,
        "marital_status"
    )

    capital_gain_analysis = analyse_feature(
        training_data,
        "has_capital_gain"
    )

    # Select the analysis columns that will be displayed
    displayed_columns = [
        "total_records",
        "high_income_records",
        "high_income_percentage"
    ]

    # Display the results for each analysed feature
    print("\nIncome patterns by education:")
    print(
        education_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nIncome patterns by occupation:")
    print(
        occupation_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nIncome patterns by age group:")
    print(
        age_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nIncome patterns by working-hours group:")
    print(
        hours_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nIncome patterns by marital status:")
    print(
        marital_status_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nIncome patterns by capital gain:")
    print(
        capital_gain_analysis[
            displayed_columns
        ].round(2)
    )

    print("\nTraining-data pattern analysis completed.")

# ------------------------------------------------------------
# RULE-SUPPORT CALCULATION
# Measure how many training records satisfy one proposed rule and
# calculate the income distribution among those matching records.
# ------------------------------------------------------------

def calculate_rule_support(
    training_data,
    rule_name,
    condition
):
    """Calculate the training evidence supporting one proposed rule."""

    # Select the training records that satisfy the rule condition
    matching_records = training_data[condition]

    # Count the total number of records supporting the rule
    support_count = len(matching_records)

    # Count matching records in the higher-income class
    high_income_count = int(
        (matching_records["income"] == ">50K").sum()
    )

    # Count matching records in the lower-income class
    low_income_count = int(
        (matching_records["income"] == "<=50K").sum()
    )

    # Calculate the higher-income percentage while preventing
    # division by zero when no training records match the rule
    if support_count > 0:
        high_income_percentage = (
            high_income_count / support_count
        ) * 100
    else:
        high_income_percentage = 0.0

    # Return the evidence as a dictionary for the evidence table
    return {
        "rule_name": rule_name,
        "support_count": support_count,
        "high_income_count": high_income_count,
        "low_income_count": low_income_count,
        "high_income_percentage": high_income_percentage
    }

# ------------------------------------------------------------
# EVIDENCE FOR ALL CLASSIFICATION RULES
# Apply the rule-support calculation to each proposed rule and
# display the evidence used to justify the eight rule conditions.

# Each rule is measured independently. A training record may support
# more than one rule, although the classifier later uses the first
# matching rule only.
# ------------------------------------------------------------

def build_rule_evidence(training_data):
    """Build and display the evidence table for the eight rules."""

    # Measure support for the capital-gain rule
    rule_1_evidence = calculate_rule_support(
        training_data,
        "Rule 1: Capital gain of at least 5000",
        training_data["capital_gain"] >= 5000
    )

    # Measure support for the advanced-education rule
    rule_2_evidence = calculate_rule_support(
        training_data,
        "Rule 2: Advanced education and professional occupation",
        (
            training_data["education"].isin(
                ["Doctorate", "Prof-school"]
            )
            & training_data["occupation"].isin(
                ["Exec-managerial", "Prof-specialty"]
            )
            & (training_data["hours_per_week"] >= 35)
        )
    )

    # Measure support for the Masters and long-hours rule
    rule_3_evidence = calculate_rule_support(
        training_data,
        "Rule 3: Masters, professional occupation and long hours",
        (
            (training_data["education"] == "Masters")
            & training_data["occupation"].isin(
                ["Exec-managerial", "Prof-specialty"]
            )
            & (training_data["hours_per_week"] > 40)
        )
    )

    # Measure support for the experienced graduate manager rule
    rule_4_evidence = calculate_rule_support(
        training_data,
        "Rule 4: Experienced graduate manager",
        (
            (training_data["education"] == "Bachelors")
            & (
                training_data["occupation"]
                == "Exec-managerial"
            )
            & (training_data["age"] >= 35)
            & (training_data["hours_per_week"] >= 40)
        )
    )

    # Measure support for the higher-education professional rule
    rule_5_evidence = calculate_rule_support(
        training_data,
        "Rule 5: Higher education, professional role and long hours",
        (
            (training_data["education_num"] >= 13)
            & training_data["occupation"].isin(
                ["Exec-managerial", "Prof-specialty"]
            )
            & (training_data["hours_per_week"] >= 45)
        )
    )

    # Measure support for the low-working-hours rule
    rule_6_evidence = calculate_rule_support(
        training_data,
        "Rule 6: Working fewer than 30 hours",
        training_data["hours_per_week"] < 30
    )

    # Measure support for the young-person education rule
    rule_7_evidence = calculate_rule_support(
        training_data,
        "Rule 7: Young person with school-level education",
        (
            (training_data["age"] < 25)
            & training_data["education"].isin(
                [
                    "HS-grad",
                    "Some-college",
                    "11th",
                    "12th"
                ]
            )
        )
    )

    # Measure support for the selected service-occupation rule
    rule_8_evidence = calculate_rule_support(
        training_data,
        "Rule 8: Selected service occupation",
        training_data["occupation"].isin(
            [
                "Priv-house-serv",
                "Other-service",
                "Handlers-cleaners"
            ]
        )
    )

    # Store the evidence from all eight rules in one list
    rule_evidence = [
        rule_1_evidence,
        rule_2_evidence,
        rule_3_evidence,
        rule_4_evidence,
        rule_5_evidence,
        rule_6_evidence,
        rule_7_evidence,
        rule_8_evidence
    ]

    # Convert the rule-evidence list into a DataFrame
    rule_evidence_table = pd.DataFrame(
        rule_evidence
    )

    # Display the evidence supporting each proposed rule
    print("\nEvidence supporting the proposed rules:")

    print(
        rule_evidence_table[
            [
                "rule_name",
                "support_count",
                "high_income_count",
                "low_income_count",
                "high_income_percentage"
            ]
        ].round(2).to_string(index=False)
    )

    # Return the completed evidence table
    return rule_evidence_table

# ------------------------------------------------------------
# RULE-BASED INFERENCE ENGINE
# Apply eight ordered IF-THEN rules to one Adult Dataset record.
# The first matching rule determines the predicted income class.
# Race and sex are not used as classification conditions.
# ------------------------------------------------------------

def classify_income(record):
    """Return an income prediction and the rule that produced it."""

    # Retrieve the record values required by the classification rules
    age = record["age"]
    education = record["education"]
    education_num = record["education_num"]
    occupation = record["occupation"]
    capital_gain = record["capital_gain"]
    hours_per_week = record["hours_per_week"]

    # Rule 1: Substantial capital gain indicates higher income
    if capital_gain >= 5000:
        return (
            ">50K",
            "Rule 1: Capital gain of at least 5000"
        )

    # Rule 2: Advanced education and professional employment
    if (
        education in ["Doctorate", "Prof-school"]
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week >= 35
    ):
        return (
            ">50K",
            "Rule 2: Advanced education and professional occupation"
        )

    # Rule 3: Masters qualification, professional work and long hours
    if (
        education == "Masters"
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week > 40
    ):
        return (
            ">50K",
            "Rule 3: Masters, professional occupation and long hours"
        )

    # Rule 4: Experienced graduate working in management
    if (
        education == "Bachelors"
        and occupation == "Exec-managerial"
        and age >= 35
        and hours_per_week >= 40
    ):
        return (
            ">50K",
            "Rule 4: Experienced graduate manager"
        )

    # Rule 5: Higher education, professional work and extended hours
    if (
        education_num >= 13
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week >= 45
    ):
        return (
            ">50K",
            "Rule 5: Higher education, professional role and long hours"
        )

    # Rule 6: Working fewer than 30 hours indicates lower income
    if hours_per_week < 30:
        return (
            "<=50K",
            "Rule 6: Working fewer than 30 hours"
        )

    # Rule 7: Younger person with school-level education
    if (
        age < 25
        and education in [
            "HS-grad",
            "Some-college",
            "11th",
            "12th"
        ]
    ):
        return (
            "<=50K",
            "Rule 7: Young person with school-level education"
        )

    # Rule 8: Selected service occupations indicate lower income
    if occupation in [
        "Priv-house-serv",
        "Other-service",
        "Handlers-cleaners"
    ]:
        return (
            "<=50K",
            "Rule 8: Selected service occupation"
        )

    # Use the majority income class when no specific rule matches
    return (
        "<=50K",
        "Default rule: Majority income class"
    )

# ------------------------------------------------------------
# COMPLETE RULE-MATCH INSPECTION
# Check every specific rule for one record instead of stopping after
# the first match. This supports the rule-overlap analysis.
# ------------------------------------------------------------

def find_matching_rules(record):
    """Return every specific classification rule matched by a record."""

    matching_rules = []

    # Retrieve the values required by the classification rules
    age = record["age"]
    education = record["education"]
    education_num = record["education_num"]
    occupation = record["occupation"]
    capital_gain = record["capital_gain"]
    hours_per_week = record["hours_per_week"]

    # Check whether the record satisfies Rule 1
    if capital_gain >= 5000:
        matching_rules.append("Rule 1")

    # Check whether the record satisfies Rule 2
    if (
        education in ["Doctorate", "Prof-school"]
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week >= 35
    ):
        matching_rules.append("Rule 2")

    # Check whether the record satisfies Rule 3
    if (
        education == "Masters"
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week > 40
    ):
        matching_rules.append("Rule 3")

    # Check whether the record satisfies Rule 4
    if (
        education == "Bachelors"
        and occupation == "Exec-managerial"
        and age >= 35
        and hours_per_week >= 40
    ):
        matching_rules.append("Rule 4")

    # Check whether the record satisfies Rule 5
    if (
        education_num >= 13
        and occupation in [
            "Exec-managerial",
            "Prof-specialty"
        ]
        and hours_per_week >= 45
    ):
        matching_rules.append("Rule 5")

    # Check whether the record satisfies Rule 6
    if hours_per_week < 30:
        matching_rules.append("Rule 6")

    # Check whether the record satisfies Rule 7
    if (
        age < 25
        and education in [
            "HS-grad",
            "Some-college",
            "11th",
            "12th"
        ]
    ):
        matching_rules.append("Rule 7")

    # Check whether the record satisfies Rule 8
    if occupation in [
        "Priv-house-serv",
        "Other-service",
        "Handlers-cleaners"
    ]:
        matching_rules.append("Rule 8")

    # Return every specific rule matched by the record
    return matching_rules


# ------------------------------------------------------------
# RULE-OVERLAP ANALYSIS
# Identify records satisfying multiple rules. The normal classifier
# still uses the first matching rule to produce its final prediction.
# ------------------------------------------------------------

def analyse_rule_overlaps(testing_data):
    """Analyse records that satisfy more than one classification rule."""

    # Create a separate copy for the overlap analysis
    overlap_data = testing_data.copy()

    # Find every specific rule matched by each testing record
    overlap_data["matching_rules"] = overlap_data.apply(
        find_matching_rules,
        axis=1
    )

    # Count how many specific rules each record satisfies
    overlap_data["rule_match_count"] = (
        overlap_data["matching_rules"].apply(len)
    )

    # Convert each list of rules into readable text
    overlap_data["matching_rules_text"] = (
        overlap_data["matching_rules"].apply(
            lambda rules: (
                ", ".join(rules)
                if rules
                else "No specific rule"
            )
        )
    )

    # Select records that do not satisfy a specific rule
    no_rule_matches = overlap_data[
        overlap_data["rule_match_count"] == 0
    ]

    # Select records that satisfy exactly one rule
    single_rule_matches = overlap_data[
        overlap_data["rule_match_count"] == 1
    ]

    # Select records that satisfy more than one rule
    overlapping_records = overlap_data[
        overlap_data["rule_match_count"] > 1
    ].copy()

    # Count the most common overlapping rule combinations
    overlap_combinations = (
        overlapping_records["matching_rules_text"]
        .value_counts()
    )

    # Display the overlap-analysis results
    print("\nRule-overlap analysis:")

    print(
        f"Records matching no specific rule: "
        f"{len(no_rule_matches)}"
    )

    print(
        f"Records matching exactly one specific rule: "
        f"{len(single_rule_matches)}"
    )

    print(
        f"Records matching multiple rules: "
        f"{len(overlapping_records)}"
    )

    print("\nMost common overlapping rule combinations:")

    if overlap_combinations.empty:
        print("No overlapping rule combinations were found.")
    else:
        print(
            overlap_combinations.head(10).to_string()
        )

    # Return the complete analysis and overlapping records
    return overlap_data, overlapping_records


# ------------------------------------------------------------
# TESTING-DATA PREDICTION GENERATION
# Apply the rule-based classification engine to every testing record
# and store the prediction, rule used and prediction result.
# ------------------------------------------------------------

# this code line is used to generate predictions for the testing dataset using the rule-based classifier. 
def generate_predictions(testing_data):
    """Generate and store rule-based predictions."""

    # Apply classify_income() to every row in the testing dataset
    rule_decisions = testing_data.apply(
        classify_income,
        axis=1
    )

    # Store the predicted income class from each returned decision
    testing_data["predicted_income"] = rule_decisions.apply(
        lambda decision: decision[0]
    )

    # Store the name of the rule responsible for each prediction
    testing_data["rule_used"] = rule_decisions.apply(
        lambda decision: decision[1]
    )

    # Record whether each prediction matches the actual income class
    testing_data["correct_prediction"] = (
        testing_data["predicted_income"]
        == testing_data["income"]
    )

    print("\nRule-based predictions have been generated.")

    # Return the testing dataset with the new prediction columns
    return testing_data


# ------------------------------------------------------------
# PERFORMANCE EVALUATION AND BASELINE COMPARISON
# Calculate the classification measurements, compare the rule-based
# system with the majority-class baseline and report rule usage.
# ------------------------------------------------------------

def evaluate_predictions(training_data, testing_data):
    """Calculate and display classification performance measurements."""

    # Count the total number of testing predictions
    total_predictions = len(testing_data)

    # Stop the programme if no usable testing records are available
    if total_predictions == 0:
        raise ValueError(
            "The testing dataset contains no usable records."
        )

    # Count the correct and incorrect predictions
    correct_predictions = int(
        testing_data["correct_prediction"].sum()
    )

    incorrect_predictions = (
        total_predictions - correct_predictions
    )

    # Calculate the overall rule-based accuracy
    rule_based_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    # Count actual >50K records correctly predicted as >50K
    true_positive = int(
        (
            (testing_data["income"] == ">50K")
            & (
                testing_data["predicted_income"]
                == ">50K"
            )
        ).sum()
    )

    # Count actual <=50K records correctly predicted as <=50K
    true_negative = int(
        (
            (testing_data["income"] == "<=50K")
            & (
                testing_data["predicted_income"]
                == "<=50K"
            )
        ).sum()
    )

    # Count actual <=50K records incorrectly predicted as >50K
    false_positive = int(
        (
            (testing_data["income"] == "<=50K")
            & (
                testing_data["predicted_income"]
                == ">50K"
            )
        ).sum()
    )

    # Count actual >50K records incorrectly predicted as <=50K
    false_negative = int(
        (
            (testing_data["income"] == ">50K")
            & (
                testing_data["predicted_income"]
                == "<=50K"
            )
        ).sum()
    )

    # Calculate how many predicted >50K records were actually >50K
    if true_positive + false_positive > 0:
        precision = (
            true_positive
            / (true_positive + false_positive)
        ) * 100
    else:
        precision = 0.0

    # Calculate how many actual >50K records were correctly identified
    if true_positive + false_negative > 0:
        recall = (
            true_positive
            / (true_positive + false_negative)
        ) * 100
    else:
        recall = 0.0

    # Calculate the balance between precision and recall
    if precision + recall > 0:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0.0

    # Identify the most common income class in the training dataset
    majority_class = training_data["income"].mode()[0]

    # Use the majority class as a simple baseline prediction
    testing_data["baseline_prediction"] = majority_class

    # Record whether each baseline prediction is correct
    testing_data["baseline_correct"] = (
        testing_data["baseline_prediction"]
        == testing_data["income"]
    )

    # Count the correct baseline predictions
    baseline_correct_predictions = int(
        testing_data["baseline_correct"].sum()
    )

    # Calculate the majority-class baseline accuracy
    baseline_accuracy = (
        baseline_correct_predictions / total_predictions
    ) * 100

    # Calculate the improvement over the baseline in percentage points
    accuracy_difference = (
        rule_based_accuracy - baseline_accuracy
    )

    # Count how many predictions were produced by each rule
    rule_usage = Counter(
        testing_data["rule_used"]
    )

    # Display the overall prediction results
    print("\nOverall rule-based results:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Rule-based accuracy: {rule_based_accuracy:.2f}%")

    # Display the classification measurements
    print("\nClassification breakdown:")
    print(f"True positive: {true_positive}")
    print(f"True negative: {true_negative}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"Precision for >50K: {precision:.2f}%")
    print(f"Recall for >50K: {recall:.2f}%")
    print(f"F1 score for >50K: {f1_score:.2f}%")

    # Display the baseline comparison
    print("\nMajority-class baseline results:")
    print(f"Majority class: {majority_class}")
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    print(f"Rule-based accuracy: {rule_based_accuracy:.2f}%")
    print(
        f"Accuracy difference: "
        f"{accuracy_difference:.2f} percentage points"
    )

    # Display how often each rule produced a prediction
    print("\nRule usage totals:")

    for rule_name, activation_count in rule_usage.most_common():
        print(
            f"{rule_name}: "
            f"{activation_count}"
        )

    # Store the measurements for the CSV results-saving function
    metrics = {
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "rule_based_accuracy": rule_based_accuracy,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference": accuracy_difference
    }

    # Return the measurements for save_results()
    return metrics

# ------------------------------------------------------------
# MAJORITY-CLASS BASELINE ERROR ANALYSIS
# Identify records incorrectly classified by the majority-class
# baseline and determine which errors the rule system corrected.
# ------------------------------------------------------------

def analyse_baseline_errors(testing_data):
    """Analyse baseline errors and rule-based recoveries."""

    # Select records incorrectly classified by the baseline
    baseline_errors = testing_data[
        ~testing_data["baseline_correct"]
    ].copy()

    # Select baseline errors corrected by the rule-based classifier
    recovered_records = baseline_errors[
        baseline_errors["correct_prediction"]
    ].copy()

    # Select records incorrectly classified by both approaches
    still_missed_records = baseline_errors[
        ~baseline_errors["correct_prediction"]
    ].copy()

    # Count the rules responsible for correcting baseline errors
    recovery_rule_counts = (
        recovered_records["rule_used"]
        .value_counts()
    )

    # Calculate the percentage of baseline errors corrected
    if len(baseline_errors) > 0:
        recovery_percentage = (
            len(recovered_records)
            / len(baseline_errors)
        ) * 100
    else:
        recovery_percentage = 0.0

    # Display the baseline-error analysis
    print("\nMajority-class baseline error analysis:")

    print(
        f"Records incorrectly classified by the baseline: "
        f"{len(baseline_errors)}"
    )

    print(
        f"Baseline errors corrected by the rule system: "
        f"{len(recovered_records)}"
    )

    print(
        f"Baseline errors still missed by the rule system: "
        f"{len(still_missed_records)}"
    )

    print(
        f"Percentage of baseline errors corrected: "
        f"{recovery_percentage:.2f}%"
    )

    print("\nRules responsible for correcting baseline errors:")

    if recovery_rule_counts.empty:
        print("No baseline errors were corrected.")
    else:
        print(recovery_rule_counts.to_string())

    # Return the records for later inspection and CSV export
    return (
        baseline_errors,
        recovered_records,
        still_missed_records
    )


# ------------------------------------------------------------
# PREDICTION EXAMPLES
# Select and display ten correct and ten incorrect classifications
# to demonstrate how the rule-based system performs on individual records.
# ------------------------------------------------------------

def display_prediction_examples(testing_data):
    """Display ten correct and ten incorrect prediction examples."""

    # Select the first ten correctly classified testing records
    correct_examples = testing_data[
        testing_data["correct_prediction"]
    ].head(10)

    # Select the first ten incorrectly classified testing records
    incorrect_examples = testing_data[
        ~testing_data["correct_prediction"]
    ].head(10)

    # Select the relevant columns for displaying each example
    example_columns = [
        "age",
        "education",
        "occupation",
        "capital_gain",
        "hours_per_week",
        "income",
        "predicted_income",
        "rule_used"
    ]

    # Display the correctly classified records
    print("\nSample correct predictions:")

    print(
        correct_examples[
            example_columns
        ].to_string(index=False)
    )

    # Display the incorrectly classified records
    print("\nSample incorrect predictions:")

    print(
        incorrect_examples[
            example_columns
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# CLASSIFICATION ERROR ANALYSIS
# Identify false-positive and false-negative predictions and count
# how many errors were produced by each rule.
# ------------------------------------------------------------

def analyse_errors(testing_data):
    """Analyse prediction errors by the rule responsible."""

    # Select lower-income records incorrectly predicted as >50K
    false_positive_examples = testing_data[
        (testing_data["income"] == "<=50K")
        & (testing_data["predicted_income"] == ">50K")
    ]

    # Select higher-income records incorrectly predicted as <=50K
    false_negative_examples = testing_data[
        (testing_data["income"] == ">50K")
        & (testing_data["predicted_income"] == "<=50K")
    ]

    # Count the rules responsible for false-positive errors
    false_positive_rule_counts = (
        false_positive_examples["rule_used"]
        .value_counts()
    )

    # Count the rules responsible for false-negative errors
    false_negative_rule_counts = (
        false_negative_examples["rule_used"]
        .value_counts()
    )

    # Display false-positive errors grouped by rule
    print("\nFalse-positive errors by rule:")
    print(false_positive_rule_counts.to_string())

    # Display false-negative errors grouped by rule
    print("\nFalse-negative errors by rule:")
    print(false_negative_rule_counts.to_string())

    # Display the total number of each error type
    print(
        f"\nTotal false-positive errors: "
        f"{len(false_positive_examples)}"
    )

    print(
        f"Total false-negative errors: "
        f"{len(false_negative_examples)}"
    )

# ------------------------------------------------------------
# REFERENCE FOR CSV OUTPUT
# Source: pandas Development Team
# Title: pandas.DataFrame.to_csv
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
#
# This documentation supported understanding of exporting DataFrames
# to CSV files without including the DataFrame index.
# ------------------------------------------------------------

def save_results(testing_data, metrics):
    """Save detailed predictions and evaluation metrics as CSV files."""

    # Create the output file paths inside the programme folder
    results_file = os.path.join(
        PROGRAM_FOLDER,
        "adult_rule_based_results.csv"
    )

    metrics_file = os.path.join(
        PROGRAM_FOLDER,
        "adult_rule_based_metrics.csv"
    )

    # Select the testing-data columns included in the detailed results
    results_columns = [
        "age",
        "workclass",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
        "predicted_income",
        "rule_used",
        "correct_prediction",
        "baseline_prediction",
        "baseline_correct"
    ]

    # Save the detailed record-level predictions
    testing_data[
        results_columns
    ].to_csv(
        results_file,
        index=False
    )

    # Create a one-row DataFrame containing the evaluation measurements
    metrics_summary = pd.DataFrame(
        [
            {
                "total_predictions": metrics[
                    "total_predictions"
                ],
                "correct_predictions": metrics[
                    "correct_predictions"
                ],
                "incorrect_predictions": metrics[
                    "incorrect_predictions"
                ],
                "true_positive": metrics[
                    "true_positive"
                ],
                "true_negative": metrics[
                    "true_negative"
                ],
                "false_positive": metrics[
                    "false_positive"
                ],
                "false_negative": metrics[
                    "false_negative"
                ],
                "rule_based_accuracy": round(
                    metrics["rule_based_accuracy"],
                    2
                ),
                "precision_over_50K": round(
                    metrics["precision"],
                    2
                ),
                "recall_over_50K": round(
                    metrics["recall"],
                    2
                ),
                "f1_score_over_50K": round(
                    metrics["f1_score"],
                    2
                ),
                "baseline_accuracy": round(
                    metrics["baseline_accuracy"],
                    2
                ),
                "accuracy_difference": round(
                    metrics["accuracy_difference"],
                    2
                )
            }
        ]
    )

    # Save the summary evaluation measurements
    metrics_summary.to_csv(
        metrics_file,
        index=False
    )

    # Confirm where the two output files were saved
    print(
        f"\nDetailed results saved to: "
        f"{results_file}"
    )

    print(
        f"Evaluation metrics saved to: "
        f"{metrics_file}"
    )

# ------------------------------------------------------------
# INDIVIDUAL RECORD CLASSIFICATION
# Collect and validate one hypothetical record entered through the
# terminal, then apply the existing rule-based inference engine.
# This feature is separate from the formal testing-dataset evaluation.
# ------------------------------------------------------------

def classify_individual_record():
    """Collect and classify one hypothetical Adult Dataset record."""

    print("\nIndividual Income Classification")
    print("Enter the requested information.")

    # Collect the numerical inputs and reject non-numerical values
    try:
        age = int(
            input("Age: ")
        )

        education_num = int(
            input("Education number, between 1 and 16: ")
        )

        capital_gain = int(
            input("Capital gain: ")
        )

        hours_per_week = int(
            input("Hours worked per week: ")
        )

    except ValueError:
        print(
            "\nInvalid input. Numerical fields must contain "
            "whole numbers."
        )
        return

    # Collect the categorical inputs used by the classification rules
    education = input(
        "Education, for example Bachelors or Masters: "
    ).strip()

    occupation = input(
        "Occupation, for example Exec-managerial: "
    ).strip()

    # Validate that the numerical inputs are within sensible ranges
    if age < 17 or age > 100:
        print("\nAge must be between 17 and 100.")
        return

    if education_num < 1 or education_num > 16:
        print("\nEducation number must be between 1 and 16.")
        return

    if capital_gain < 0:
        print("\nCapital gain cannot be negative.")
        return

    if hours_per_week < 1 or hours_per_week > 99:
        print("\nHours per week must be between 1 and 99.")
        return

    # Store the entered values in the format required by classify_income()
    individual_record = {
        "age": age,
        "education": education,
        "education_num": education_num,
        "occupation": occupation,
        "capital_gain": capital_gain,
        "hours_per_week": hours_per_week
    }

    # Apply the same inference engine used for the testing dataset
    predicted_income, rule_used = classify_income(
        individual_record
    )

    # Display the prediction and the rule responsible
    print("\nIndividual Classification Result")
    print(f"Predicted income: {predicted_income}")
    print(f"Rule used: {rule_used}")


# ------------------------------------------------------------
# MAIN PROGRAMME ROUTINE
# Run each stage of the Adult Dataset rule-based AI system in the
# correct order, from loading the data to saving the final results.
# ------------------------------------------------------------

def main():
    """Run the complete Adult Dataset rule-based AI programme."""

    print("\nStarting the Adult Dataset rule-based AI programme.")

    # Load the original training and testing datasets
    training_data, testing_data = load_datasets()

    # Clean and validate both datasets
    training_data, testing_data = clean_datasets(
        training_data,
        testing_data
    )

    # Add the grouped attributes required for pattern analysis
    training_data, testing_data = add_analysis_groups(
        training_data,
        testing_data
    )

    # Analyse relationships between selected attributes and income
    analyse_training_patterns(
        training_data
    )

    # Calculate and display the evidence supporting the eight rules
    build_rule_evidence(
        training_data
    )

    # Apply the rule-based classifier to every testing record
    testing_data = generate_predictions(
        testing_data
    )

    # Calculate the performance measurements and baseline comparison
    metrics = evaluate_predictions(
        training_data,
        testing_data
    )

    # Analyse records that satisfy multiple classification rules
    analyse_rule_overlaps(
        testing_data
    )

    # Analyse which majority-class baseline errors were corrected
    analyse_baseline_errors(
        testing_data
    )

    # Display examples of correct and incorrect predictions
    display_prediction_examples(
        testing_data
    )

    # Analyse false-positive and false-negative errors
    analyse_errors(
        testing_data
    )

    # Save the detailed predictions and summary metrics
    save_results(
        testing_data,
        metrics
    )

    # Offer the optional individual-record classification feature
    user_choice = input(
        "\nWould you like to classify an individual record? "
        "(yes/no): "
    ).strip().lower()

    if user_choice in ["yes", "y"]:
        classify_individual_record()

    elif user_choice not in ["no", "n"]:
        print(
            "\nInput not recognised. "
            "Individual classification was skipped."
        )

    print(
        "\nAdult Dataset rule-based AI programme "
        "completed successfully."
    )

# ------------------------------------------------------------
# PROGRAMME ENTRY POINT AND ERROR HANDLING
# Start the programme when this file is run directly and display
# clear messages if an error occurs during execution.
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print("\nFile error:")
        print(error)

    except pd.errors.EmptyDataError:
        print("\nData error:")
        print("One of the dataset files is empty.")

    except pd.errors.ParserError as error:
        print("\nDataset parsing error:")
        print(error)

    except KeyError as error:
        print("\nColumn error:")
        print(f"Required column not found: {error}")

    except ValueError as error:
        print("\nValue error:")
        print(error)

    except PermissionError as error:
        print("\nPermission error:")
        print(error)

    except Exception as error:
        print("\nUnexpected programme error:")
        print(f"{type(error).__name__}: {error}")



# ------------------------------------------------------------
# DEVELOPMENT TOOLS, SOURCES AND GENERATIVE AI DISCLOSURE
# ------------------------------------------------------------
#
# This section provides a summary of the materials, development tools
# and generative AI assistance used during the project. Full references
# are included in the written report.
#
#
# DATASET AND MODULE MATERIALS
#
# The Adult Dataset package was supplied through the CPU5006-20
# Artificial Intelligence assessment area on the university's
# Ultra learning page.
#
# Files used directly for classification:
# - adult.data
# - adult.test
#
# Documentation file checked by the programme:
# - adult.names
#
# Additional documentation retained from the supplied package:
# - Index
# - old.adult.names
#
# The Index and old.adult.names files were examined using a separate
# documentation-inspection script. This confirmed information about
# the dataset record totals, attributes, missing values and class
# distribution.
#
# Index and old.adult.names are supporting documentation and are not
# required during normal execution of the rule-based classifier.
#
# The supplied adult.names documentation and module CodeLab materials
# supported understanding of the dataset structure, Python functions,
# file handling, data structures and programme organisation.
#
#
# TECHNICAL REFERENCES
#
# Python Software Foundation
# Title: os.path - Common pathname manipulations
# URL: https://docs.python.org/3/library/os.path.html
# Relevant code: PROGRAM_FOLDER, os.path.abspath(),
# os.path.dirname() and os.path.join().
#
# Python Software Foundation
# Title: os - Miscellaneous operating system interfaces
# URL: https://docs.python.org/3/library/os.html
# Relevant code: find_file() and os.walk().
#
# Python Software Foundation
# Title: collections - Container datatypes
# URL: https://docs.python.org/3/library/collections.html
# Relevant code: Counter was used for income totals and rule usage.
#
# NumPy Developers
# Title: NumPy documentation
# URL: https://numpy.org/doc/stable/
# Relevant code: np.inf, np.nan and np.where().
#
# pandas Development Team
# Title: pandas documentation
# URL: https://pandas.pydata.org/docs/
# Relevant code: DataFrame loading, cleaning, analysis and export.
#
# pandas Development Team
# Title: pandas.read_csv
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# Relevant code: load_datasets().
#
# pandas Development Team
# Title: Working with missing data
# URL: https://pandas.pydata.org/docs/user_guide/missing_data.html
# Relevant code: clean_datasets().
#
# pandas Development Team
# Title: pandas.cut
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
# Relevant code: add_analysis_groups().
#
# NumPy Developers
# Title: numpy.where
# URL: https://numpy.org/doc/stable/reference/generated/numpy.where.html
# Relevant code: add_analysis_groups().
#
# pandas Development Team
# Title: pandas.DataFrame.groupby
# URL:  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
# Relevant code: analyse_feature() and analyse_training_patterns().
#
# GeeksforGeeks
# Title: Pandas GroupBy
# URL: https://www.geeksforgeeks.org/pandas/pandas-groupby/
# Relevant code: Grouped feature analysis and aggregation.
#
# Codecademy
# Title: Cleaning and Transforming Columns
# URL: https://www.codecademy.com/learn/paths/getting-started-with-python-for-data-science/tracks/getting-started-with-python-for-data-science/modules/cleaning-and-transforming-columns/cheatsheet
# Relevant code: Data cleaning and column transformation.
#
# pandas Development Team
# Title: pandas.DataFrame.to_csv
# URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
# Relevant code: save_results().
#
#
# COMPARABLE ADULT DATASET SOURCES
#
# The following resources were identified during the final source
# review. They contain similar Adult Dataset cleaning or exploratory
# analysis, but they are not presented as the original source of the
# submitted programme.
#
# Aya Sabry
# Title: Adult Census Income Dataset | EDA & Data Cleaning
# Platform: YouTube
# URL: https://www.youtube.com/watch?v=Ol6W1jTTlZA
#
# Aishashafrin
# Title: Adult-income-EDA-analysis
# Platform: GitHub
# URL: https://github.com/Aishashafrin/Adult-income-EDA-analysis
#
# vaibhavikumar453
# Title: Adult Census Income EDA
# Platform: GitHub
# URL: https://github.com/vaibhavikumar453/Adult-Census-Income-EDA/blob/main/Adult%20Census%20Income%20EDA.py
#
# These comparison resources include similar approaches to dataset
# loading, missing-value inspection, duplicate checking and grouped
# feature analysis. They do not contain the complete eight-rule system,
# rule order, thresholds, rule-support calculations, baseline
# comparison, error analysis or interactive classification feature
# implemented in this project.
#
#
# VISUAL STUDIO CODE AND EXTENSIONS
#
# Visual Studio Code was used to write, run, inspect and test the
# programme.
#
# Code Runner by Jun Han was used to execute the Python programme
# during development and testing.
#
# Error Lens by Alexander was used to display errors and warnings
# beside relevant lines of code. This supported investigation of
# syntax errors, undefined names and other diagnostics.
#
# Data Preview by Random Fractals Inc. was used to inspect structured
# data and generated CSV files where required.
#
# Extension diagnostics identified possible problems. Suggested
# corrections were reviewed before they were applied, and the
# programme was rerun after changes.
#
#
# COLLECTIVE GENERATIVE AI DISCLOSURE
#
# Tool: Microsoft Copilot
#
# Microsoft Copilot was used as a supporting development and review
# tool. Assistance included suggesting possible approaches, explaining
# programming concepts, investigating errors, reviewing code comments
# and checking the organisation of the final programme.
#
# The following conversation requests relate to particular sections of
# the programme. The wording below summarises the requests made during
# development rather than claiming that each line is a verbatim prompt.
#
#
# File searching and dataset loading:
# - Asked where file-searching and dataset-loading code should be
#   placed.
# - Asked how adult.data, adult.test and adult.names should be found
#   and loaded.
#
# Related code:
# - PROGRAM_FOLDER
# - find_file()
# - load_datasets()
#
#
# Data cleaning:
# - Asked what information was required from the dataset output.
# - Asked how missing values, numerical conversion and possible
#   duplicates should be handled and explained.
#
# Related code:
# - clean_datasets()
#
#
# Training-data analysis:
# - Asked how the analysis groups and grouped income percentages
#   should be organised and commented.
#
# Related code:
# - add_analysis_groups()
# - analyse_feature()
# - analyse_training_patterns()
#
#
# Rule evidence:
# - Asked what training support meant and how the rules were being
#   checked against the training records.
#
# Related code:
# - calculate_rule_support()
# - build_rule_evidence()
#
#
# Rule-based inference:
# - Asked whether the eight rules and thresholds were required by the
#   assessment brief.
# - Asked why rule order matters and whether ordered rules are a
#   standard rule-based AI technique.
#
# Related code:
# - classify_income()
#
#
# Prediction generation:
# - Asked how each prediction and the responsible rule were returned
#   and stored.
#
# Related code:
# - generate_predictions()
#
#
# Performance evaluation:
# - Asked what the F1 score means and why it was needed.
# - Asked why recall was approximately 37 percent when overall accuracy
#   was approximately 81 percent.
# - Asked what the majority-class baseline meant.
#
# Related code:
# - evaluate_predictions()
#
#
# Rule-overlap analysis:
# - Asked how to identify testing records that satisfy multiple rules
#   without changing the existing first-match classifier.
# - Asked how to count the most common overlapping rule combinations.
#
# Related code:
# - find_matching_rules()
# - analyse_rule_overlaps()
#
#
# Majority-class baseline error analysis:
# - Asked how to identify records incorrectly classified by the
#   majority-class baseline.
# - Asked how to determine which baseline errors were corrected or
#   still missed by the rule-based classifier.
#
# Related code:
# - analyse_baseline_errors()
#
#
# Prediction and error examples:
# - Asked what false positives and false negatives meant.
# - Asked how errors could be grouped by the responsible rule.
#
# Related code:
# - display_prediction_examples()
# - analyse_errors()
#
#
# CSV output:
# - Asked what the CSV output structure meant and what information
#   should be saved.
#
# Related code:
# - save_results()
#
#
# Individual-record classification:
# - Asked whether an optional terminal feature could be added to
#   classify one hypothetical record.
# - Asked why a person's name did not appear and whether a name was
#   required by the rule engine.
#
# Related code:
# - classify_individual_record()
#
#
# Programme organisation:
# - Asked where supporting functions should be placed.
# - Asked what the Python programme entry point does.
# - Asked how the exception-handling blocks report errors.
#
# Related code:
# - main()
# - if __name__ == "__main__"
# - Final exception handling
#
#
# Comment and presentation review:
# - Asked for the complete programme to be reviewed section by section.
# - Asked for spelling, comment placement, duplicated comments,
#   section headings and code organisation to be checked.
#
# Copilot suggestions were reviewed and adapted before inclusion.
# The final programme was executed using adult.data and adult.test.
# The terminal output, generated CSV files, evaluation measurements
# and classification errors were inspected after the changes.
#
#
# RELATIONSHIP BETWEEN SOURCES AND THE FINAL PROGRAMME
#
# The technical references explain standard Python, NumPy and Pandas
# operations used by the programme.
#
# The comparable projects contain similar Adult Dataset analysis
# techniques but do not contain this programme's complete implementation.
#
# The eight rules, their ordering and thresholds were evaluated using
# the supplied adult.data file and tested using adult.test.
#
# No single external source contains the complete submitted programme,
# its exact rule sequence, evidence table, rule-labelled predictions,
# baseline comparison, rule-overlap analysis, baseline-error analysis,
# error analysis, CSV outputs and optional individual-record
# classification feature.
#
# I remain responsible for the final implementation, testing, written
# report and video walkthrough.
# ------------------------------------------------------------

