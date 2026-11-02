Adult Dataset Rule-Based AI Classification System
Project Overview

This project implements an explainable rule-based artificial intelligence system using the Adult Census Income Dataset.

The programme addresses three related problems:

Income classification: Predict whether a dataset record belongs to the <=50K or >50K annual income class using eight ordered IF-THEN rules.
Majority-class baseline recovery: Identify genuine >50K records missed by a baseline that always predicts <=50K, then measure which errors the rule-based classifier corrects.
Rule-overlap detection: Identify records satisfying multiple rules and examine how first-match ordering affects the recorded decision.

Each classification returns the predicted income class and the name of the rule responsible, making the decision process transparent and explainable.

Dataset

The Adult Census Income Dataset was supplied through the CPU5006-20 Artificial Intelligence assessment area on Bath Spa University’s Ultra learning environment.

Required dataset files:

adult.data
adult.test
adult.names

Original dataset structure:

32,561 training records
16,281 testing records
48,842 records in total
14 input attributes
One target attribute named income
15 columns per record

The target classes are:

<=50K
>50K

The adult.data file was used to investigate patterns and develop the rules. The separate adult.test file was used for final evaluation.

Dataset Validation and Cleaning

A separate validation utility confirmed:

48,842 records were present across both data files
Every record contained exactly 15 columns
No malformed records were detected

Question marks represented missing values in workclass, occupation and native_country. Records containing missing values were removed.

Dataset	Original	Removed	RemainingTraining	32,561	2,399	30,162
Testing	16,281	1,221	15,060
Total	48,842	3,620	45,222

The programme also found 23 possible duplicate training records and five possible duplicate testing records. These were retained because the dataset does not contain a unique respondent identifier.

Rule-Based Classification

The classifier uses capital gain, education, education number, occupation, age and weekly working hours.

Rules are checked from top to bottom. The first matching rule produces the prediction, and later rules are not evaluated.

Rules
Capital gain of at least 5,000 predicts >50K.
Doctorate or Prof-school education, professional employment and at least 35 weekly working hours predict >50K.
Masters education, professional employment and more than 40 weekly working hours predict >50K.
Bachelors education, management employment, age of at least 35 and at least 40 weekly working hours predict >50K.
Education number of at least 13, professional employment and at least 45 weekly working hours predict >50K.
Fewer than 30 weekly working hours predict <=50K.
Age under 25 with selected school-level education predicts <=50K.
Selected service occupations predict <=50K.

If no specific rule matches, the programme predicts <=50K, the majority class in the cleaned training data.

Requirements

The project was developed and tested using:

Python 3.10.2
pandas 2.3.3
NumPy 2.2.6

Install the required packages:

Shell
python -m pip install -r requirements.txt

The requirements.txt file contains:

Plain Text
numpy==2.2.6
pandas==2.3.3
Running the Programme

Run the programme from the project folder:

Shell
python adult_rule_based_ai.py

The programme will:

Locate the required dataset files.
Load, clean and validate the records.
Analyse patterns in the training data.
Calculate the evidence supporting each rule.
Apply the ordered classifier to the testing data.
Calculate accuracy, precision, recall and F1 score.
Compare the classifier with the majority-class baseline.
Identify records matching multiple rules.
Measure which baseline errors were corrected.
Display prediction and error examples.
Save detailed predictions and summary metrics.
Offer an optional individual-record classification feature.
Final Results

The final evaluation used 15,060 cleaned testing records.

Measurement	ResultCorrect predictions	12,148
Incorrect predictions	2,912
Accuracy	80.66%
Baseline accuracy	75.43%
Improvement over baseline	5.23 percentage points
True positives	1,367
True negatives	10,781
False positives	579
False negatives	2,333
Precision for >50K	70.25%
Recall for >50K	36.95%
F1 score for >50K	48.42%
Majority-Class Baseline Recovery

The majority-class baseline predicts <=50K for every testing record. It therefore misclassified all 3,700 genuine >50K records.

The rule-based classifier:

Corrected 1,367 baseline errors
Still missed 2,333 higher-income records
Recovered 36.95% of the baseline errors
Rule	Baseline errors correctedRule 1	683
Rule 4	198
Rule 2	179
Rule 3	178
Rule 5	129
Total	1,367
Rule-Overlap Analysis

The separate overlap analysis checks every rule for each testing record rather than stopping after the first match.

Results:

No specific rule matched: 8,832 records
Exactly one rule matched: 4,004 records
Multiple rules matched: 2,224 records
Multiple-rule match rate: 14.77%

The most common overlaps were:

Rule combination	RecordsRules 6 and 7	395
Rules 7 and 8	380
Rules 6 and 8	306
Rules 3 and 5	285
Rules 6, 7 and 8	262
Rules 4 and 5	176
Rules 2 and 5	145

This confirms that independent rule support differs from actual rule activation under first-match ordering.

Output Files

The programme generates:

adult_rule_based_results.csv
adult_rule_based_metrics.csv

adult_rule_based_results.csv contains record-level predictions, actual classes, rule names, correctness values and baseline results.

adult_rule_based_metrics.csv contains accuracy, precision, recall, F1 score, confusion-matrix values and the baseline comparison.

The rule-overlap and baseline-recovery analyses are displayed in the terminal and do not currently create additional CSV files.

Limitations

The main limitation is the recall of 36.95% for the >50K class. The programme missed 2,333 genuine higher-income records, including 2,144 assigned by the default rule.

Other limitations include:

Heavy reliance on the default rule
Manually created conditions
Overlapping rules
Hard numerical thresholds
No automatic learning
Class imbalance
Removal of incomplete records
Limited generalisability
Ethical Considerations

Race and sex are present in the dataset but are not used as rule conditions. Excluding protected characteristics does not guarantee fairness because education, occupation and location may act as proxy variables for existing inequalities.

This programme is an educational demonstration and should not be used to make employment, lending, eligibility or other consequential decisions about individuals.

Generative AI Use

Microsoft Copilot supported programming explanations, debugging, evaluation design, refactoring, comment review, rule-overlap analysis and baseline-error analysis.

Suggestions were reviewed, adapted and tested against the supplied datasets. AI assistance is disclosed in the Python comments and written report. The final implementation, testing, interpretation, report and video walkthrough remain the responsibility of the author.

Repository Contents
adult_rule_based_ai.py
adult.data
adult.test
adult.names
Index
old.adult.names
adult_rule_based_results.csv
adult_rule_based_metrics.csv
README.md
requirements.txt
.gitignore

Author
Finn Sheridan-Smith