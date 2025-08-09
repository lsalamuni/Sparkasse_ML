Build a model f that classifies each e-mail from the Kundenanfrage field into one of the
12 sub-categories in Unterkategorie 2 (e.g. TAN procedure, fees, phishing, ...). 
Spreadsheet Dataset.xlsx with 249 observations (label and email columns).

The project involves developing models for classifying banking e-mails into 12 categories
with a high degree of class imbalance. Extensive feature engineering, including structural
indicators, banking-specific keyword counts, and n-gram TF-IDF representations, is applied
to address potential overfitting and reduce dimensionality. Three classification algorithms
are implemented: Support Vector Machines, Random Forest, and Ensemble. Model perfor-
mance is evaluated using a broad set of metrics, including Macro-F1, Weighted-F1, accuracy,
precision, recall, ROC-AUC, and Log-Loss, alongside confusion matrices and per- class error
rates.
