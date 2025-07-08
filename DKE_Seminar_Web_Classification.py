import os
import json
import shutil
import random
import re
import spacy
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Global Configuration
COLS_TO_STR = ["section", "unit", "group", "class", "subclass"]
COLS_TO_DROP_FOR_FEATURES = ["company", "section", "unit", "group", "class", "subclass"]

# Font sizes for plots
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
TICK_FONTSIZE = 10
ANNOT_FONTSIZE = 10 # For annotation in confusion matrix

# Data Splitting 

def copy_company_data(company_dir, target_root):
    section = "UNKNOWN"
    code_file = company_dir / "code_desc.json"
    json_file = company_dir / "company_data.json"
    if json_file.exists() and code_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)
            with open(code_file, "r", encoding="utf-8") as f:
                code_desc = json.load(f)
            code = company_data.get("code", "")
            label_info = code_desc.get(code, {})
            section = label_info.get("section", "UNKNOWN")
        except Exception as e:
            print(f"Error reading code_desc.json or company_data.json in {company_dir.name}: {e}")

    dest_dir = target_root / section / company_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["company_data.json", "code_desc.json"]:
        src_file = company_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dest_dir / filename)

    merged_text = ""
    txt_files = list(company_dir.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                merged_text += content + "\n\n"
        except Exception as e:
            print(f"Error reading {txt_file.name} in {company_dir.name}: {e}")

    if merged_text:
        with open(dest_dir / "all_data.txt", "w", encoding="utf-8") as out:
            out.write(merged_text.strip())

def split_data_into_train_test(source_root, train_root, test_root):
    print("\nStep 1: Splitting Data into Training and Test Sets")
    
    for root in [train_root, test_root]:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    unit_map = defaultdict(list)

    for company_dir in source_root.iterdir():
        if company_dir.is_dir():
            json_file = company_dir / "company_data.json"
            code_file = company_dir / "code_desc.json"
            if not json_file.exists() or not code_file.exists():
                print(f"Skipping {company_dir.name}: Missing company_data.json or code_desc.json")
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    company_data = json.load(f)
                with open(code_file, "r", encoding="utf-8") as f:
                    code_desc = json.load(f)
                code = company_data.get("code", "")
                label_info = code_desc.get(code, {})
                section = label_info.get("section", "UNKNOWN")
                unit = label_info.get("unit", "UNKNOWN")
                if section != "UNKNOWN" and unit != "UNKNOWN":
                    key = (section, unit)
                    unit_map[key].append(company_dir)
                else:
                    print(f"Skipping {company_dir.name}: Missing 'section' or 'unit' in code_desc.json for code '{code}'")
            except Exception as e:
                print(f"Error reading data in {company_dir.name}: {e}")

    for (section, unit), companies in unit_map.items():
        if len(companies) < 2:
            print(f"Skipping ({section}, {unit}) due to insufficient data (need at least 2 companies for split).")
            continue

        random.shuffle(companies)
        split_idx = int(0.7 * len(companies))
        train_set = companies[:split_idx]
        test_set = companies[split_idx:]

        for company in train_set:
            copy_company_data(company, train_root)
        for company in test_set:
            copy_company_data(company, test_root)

    print(f"\nData splitting complete. Companies copied to training and test folders.")
    print(f"Training data copied to: {train_root.resolve()}")
    print(f"Test data copied to: {test_root.resolve()}")

# Text Preprocessing

def clean_text(text: str) -> str:
    """
    Normalizes German umlauts, removes special characters, lowercases,
    and collapses whitespace in the given text.
    """
    umlaut_map = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"
    }
    for k, v in umlaut_map.items():
        text = text.replace(k, v)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def clean_dataset(base_dir: Path):
    print(f"\nStep 2: Cleaning Text Data in {base_dir.name}")
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return

    all_data_files = base_dir.rglob("all_data.txt")
    count = 0

    for txt_file in all_data_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned = clean_text(raw_text)
            output_path = txt_file.parent / "cleaned_data.txt"

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(cleaned)
            count += 1
        except Exception as e:
            print(f"Failed to clean {txt_file.relative_to(base_dir)}: {e}")

    print(f"\nCleaned {count} all_data.txt files in {base_dir.name}.")

# Load German spaCy model
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("SpaCy 'de_core_news_sm' model not found. Downloading...")
    spacy.cli.download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

CHUNK_SIZE = 100_000

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([
        token.lemma_
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
    ])

def lemmatize_text_chunked(text: str) -> str:
    lemmas = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        doc = nlp(chunk)
        lemmas.extend([
            token.lemma_
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop
        ])
    return " ".join(lemmas)

def process_lemmatization(base_dir: Path):

    print(f"\nStep 3: Lemmatizing Text Data in {base_dir.name}")
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return

    cleaned_files = base_dir.rglob("cleaned_data.txt")
    count = 0
    skipped = 0

    for cleaned_file in cleaned_files:
        lemmatized_path = cleaned_file.parent / "lemmatized_data.txt"

        if lemmatized_path.exists():
            skipped += 1
            continue

        try:
            with open(cleaned_file, "r", encoding="utf-8") as f:
                text = f.read()

            try:
                lemmatized = lemmatize_text(text)
            except Exception as e:
                print(f"Normal lemmatization failed for {cleaned_file.relative_to(base_dir)}, falling back. Error: {e}")
                lemmatized = lemmatize_text_chunked(text)

            with open(lemmatized_path, "w", encoding="utf-8") as out:
                out.write(lemmatized)
            count += 1

        except Exception as e:
            print(f"Failed on {cleaned_file.relative_to(base_dir)}: {e}")

    print(f"\nLemmatized: {count}, Skipped: {skipped} files in {base_dir.name}.")

# TF-IDF Vectorization

def load_texts_and_labels(base_dir):
    texts = []
    labels = []
    companies = []
    label_levels = ["section", "unit", "group", "class", "subclass"]

    for company_dir in base_dir.rglob("lemmatized_data.txt"):
        try:
            with open(company_dir, "r", encoding="utf-8") as f:
                text = f.read().strip()
            texts.append(text)

            meta_path = company_dir.parent / "company_data.json"
            code_path = company_dir.parent / "code_desc.json"

            if not meta_path.exists() or not code_path.exists():
                print(f"Skipping metadata for {company_dir.parent.name}: Missing company_data.json or code_desc.json")
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            code = meta.get("code", "")

            with open(code_path, "r", encoding="utf-8") as f:
                code_map = json.load(f)
            label_data = code_map.get(code, {})

            label_row = {lvl: label_data.get(lvl, "UNKNOWN") for lvl in label_levels}
            labels.append(label_row)
            companies.append(company_dir.parent.name)

        except Exception as e:
            print(f"Failed reading from {company_dir.parent.name}: {e}")

    return texts, labels, companies

def generate_tfidf_features(train_dir, test_dir, output_dir):
    print("\nStep 4: Generating TF-IDF Features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data for TF-IDF...")
    train_texts, train_labels, train_companies = load_texts_and_labels(train_dir)

    print("Vectorizing training data...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
    X_train = vectorizer.fit_transform(train_texts)

    df_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    df_train["company"] = train_companies
    for lvl in COLS_TO_STR:
        df_train[lvl] = [row[lvl] for row in train_labels]
    train_tfidf_path = output_dir / "tfidf_train.csv"
    df_train.to_csv(train_tfidf_path, index=False)
    print(f"Saved: {train_tfidf_path}")

    print("Loading test data for TF-IDF...")
    test_texts, test_labels, test_companies = load_texts_and_labels(test_dir)
    print("Vectorizing test data...")
    X_test = vectorizer.transform(test_texts)

    test_tfidf_path = output_dir / "tfidf_test.csv"
    df_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
    df_test["company"] = test_companies
    for lvl in COLS_TO_STR:
        df_test[lvl] = [row[lvl] for row in test_labels]
    df_test.to_csv(test_tfidf_path, index=False)
    print(f"Saved: {test_tfidf_path}")

    return train_tfidf_path, test_tfidf_path

#Classification and Evaluation

def load_and_preprocess_data(train_path, test_path, cols_to_str):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Error: TF-IDF training or testing CSV file not found. Please run TF-IDF generation first.")
        exit()

    for col in cols_to_str:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)
    return train_df, test_df

def run_hierarchical_classification(model_name, section_classifier, unit_classifier_builder,
                                  train_df, test_df,
                                  section_label_col, unit_label_col,
                                  cols_to_drop_for_features):
    print(f"\n{'='*50}\nRunning Hierarchical Classification with {model_name} \n{'='*50}")

    X_train_processed = train_df.drop(columns=cols_to_drop_for_features, errors='ignore')
    X_test_processed = test_df.drop(columns=cols_to_drop_for_features, errors='ignore')
    
    if not isinstance(X_train_processed, pd.DataFrame):
        X_train_processed = pd.DataFrame(X_train_processed, index=train_df.index)
        X_test_processed = pd.DataFrame(X_test_processed, index=test_df.index)

    # 1. Section-Level Classification (Stage 1) 
    print(f"\nStage 1: Section-Level Classification ({model_name})")
    
    y_train_section_labels = train_df[section_label_col]
    y_test_section_labels = test_df[section_label_col]

    section_encoder = LabelEncoder()
    y_train_section_encoded = section_encoder.fit_transform(y_train_section_labels)

    valid_section_mask = y_test_section_labels.isin(section_encoder.classes_)
    X_test_section_features_valid = X_test_processed.loc[valid_section_mask]
    y_test_section_labels_valid = y_test_section_labels[valid_section_mask]
    y_test_section_encoded_actual = section_encoder.transform(y_test_section_labels_valid)

    section_classifier.fit(X_train_processed, y_train_section_encoded)
    y_pred_section_encoded = section_classifier.predict(X_test_section_features_valid)

    acc_section = accuracy_score(y_test_section_encoded_actual, y_pred_section_encoded)
    section_report_dict = classification_report(y_test_section_encoded_actual, y_pred_section_encoded,
                                                target_names=section_encoder.classes_, zero_division=0, output_dict=True)
    f1_section = section_report_dict['weighted avg']['f1-score']

    print(f"Accuracy (Section Level): {acc_section:.4f}")
    print(f"F1-Score (weighted, Section Level): {f1_section:.4f}")
    print("\nClassification Report (Section Level):")
    print(classification_report(y_test_section_encoded_actual, y_pred_section_encoded,
                                target_names=section_encoder.classes_, zero_division=0))

    conf_mat_section = confusion_matrix(y_test_section_encoded_actual, y_pred_section_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat_section, annot=True, fmt='d', cmap="Blues",
                xticklabels=section_encoder.classes_, yticklabels=section_encoder.classes_,
                annot_kws={"size": ANNOT_FONTSIZE})
    plt.title(f"Section Level - {model_name}", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Actual", fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=90, fontsize=TICK_FONTSIZE)
    plt.yticks(rotation=0, fontsize=TICK_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_section_level_{model_name.replace(' ', '_').lower()}.png")
    plt.show()
    plt.close()

    #2. Unit-Level Classification (Stage 2)
    print(f"\n-Stage 2: Unit-Level Classification by Predicted Section ({model_name})")

    y_true_units_original_overall = []
    y_pred_units_original_overall = []

    temp_test_df = test_df.loc[valid_section_mask].copy()
    temp_test_df["predicted_section_str"] = section_encoder.inverse_transform(y_pred_section_encoded)

    unique_predicted_sections = temp_test_df["predicted_section_str"].unique()
    if unique_predicted_sections.size == 0:
        print("No sections were predicted. Cannot proceed with unit classification.")
        return f1_section, 0.0, section_report_dict, {}

    for predicted_section_str_label in unique_predicted_sections:
        print(f"\nUnit classification for items PREDICTED as section '{predicted_section_str_label}'")

        train_indices_current_section = train_df[train_df[section_label_col] == predicted_section_str_label].index
        X_train_unit_features = X_train_processed.loc[train_indices_current_section]
        y_train_unit_labels = train_df.loc[train_indices_current_section, unit_label_col]

        test_indices_current_predicted_section = temp_test_df[temp_test_df["predicted_section_str"] == predicted_section_str_label].index
        X_test_unit_features = X_test_processed.loc[test_indices_current_predicted_section]
        y_test_unit_labels_actual = test_df.loc[test_indices_current_predicted_section, unit_label_col]

        if len(X_train_unit_features) < 2:
            print(f"Skipping section '{predicted_section_str_label}': Insufficient training data ({len(X_train_unit_features)} samples) for unit model.")
            continue
        if len(X_test_unit_features) < 1:
            print(f"Skipping section '{predicted_section_str_label}': No test data predicted for this section to evaluate unit model.")
            continue
        if len(y_train_unit_labels.unique()) < 2:
            print(f"Skipping section '{predicted_section_str_label}': Only one unique unit label ('{y_train_unit_labels.unique()[0]}') in training data for this section. Cannot train a classifier with single class.")
            continue

        unit_encoder = LabelEncoder()
        try:
            y_train_unit_encoded = unit_encoder.fit_transform(y_train_unit_labels)

            unit_labels_known_to_encoder_mask = y_test_unit_labels_actual.isin(unit_encoder.classes_)

            if not unit_labels_known_to_encoder_mask.any():
                print(f"No test samples in predicted section '{predicted_section_str_label}' have unit labels that were seen during this section's unit training. Skipping evaluation for these.")
                continue

            X_test_unit_features_valid = X_test_unit_features[unit_labels_known_to_encoder_mask]
            y_test_unit_labels_actual_valid = y_test_unit_labels_actual[unit_labels_known_to_encoder_mask]
            y_test_unit_encoded_actual = unit_encoder.transform(y_test_unit_labels_actual_valid)

            if len(X_test_unit_features_valid) == 0:
                print(f"After filtering for known unit labels, no test samples remain for section '{predicted_section_str_label}'.")
                continue

            model_unit = unit_classifier_builder()
            model_unit.fit(X_train_unit_features, y_train_unit_encoded)
            y_pred_unit_encoded = model_unit.predict(X_test_unit_features_valid)

            y_true_units_original_overall.extend(y_test_unit_labels_actual_valid.tolist())
            y_pred_units_original_overall.extend(unit_encoder.inverse_transform(y_pred_unit_encoded).tolist())

            acc_unit_section = accuracy_score(y_test_unit_encoded_actual, y_pred_unit_encoded)
            f1_unit_section = f1_score(y_test_unit_encoded_actual, y_pred_unit_encoded, average='weighted', zero_division=0)
            print(f"Accuracy for units in predicted section '{predicted_section_str_label}': {acc_unit_section:.4f}")
            print(f"F1-Score (weighted) for units in predicted section '{predicted_section_str_label}': {f1_unit_section:.4f}")
            print(f"Classification Report for units in predicted section '{predicted_section_str_label}':")
            
            report_labels_indices = [i for i, cls_name in enumerate(unit_encoder.classes_) if cls_name in y_test_unit_labels_actual_valid.unique() or cls_name in pd.Series(unit_encoder.inverse_transform(y_pred_unit_encoded)).unique()]
            report_target_names = [unit_encoder.classes_[i] for i in report_labels_indices]
            
            if not report_target_names:
                print("Could not generate detailed report for this section due to label mismatch after filtering.")
            else:
                print(classification_report(y_test_unit_encoded_actual, y_pred_unit_encoded,
                                            labels=np.arange(len(unit_encoder.classes_)),
                                            target_names=unit_encoder.classes_,
                                            zero_division=0))

        except ValueError as ve:
            if "y contains previously unseen labels" in str(ve):
                print(f"ValueError during unit classification for section '{predicted_section_str_label}': {ve}. This might indicate an issue with label filtering or insufficient training labels.")
            elif "Found array with 0 sample(s)" in str(ve) or "Found input variables with inconsistent numbers of samples" in str(ve):
                print(f"ValueError: Not enough samples for unit classification in section '{predicted_section_str_label}' after filtering: {ve}")
            else:
                print(f"An unexpected ValueError occurred for section '{predicted_section_str_label}': {e}")
        except Exception as e:
            print(f"Failed unit classification for section '{predicted_section_str_label}': {e}")

    # 3. Evaluate Overall Unit-Level (Aggregate Results)
    overall_f1_unit = 0.0
    unit_overall_report_dict = {}
    if y_true_units_original_overall:
        print(f"\n{'='*50}\n Overall Unit-Level Evaluation ({model_name})\n{'='*50}")
        
        overall_acc_unit = accuracy_score(y_true_units_original_overall, y_pred_units_original_overall)
        overall_f1_unit = f1_score(y_true_units_original_overall, y_pred_units_original_overall, average='weighted', zero_division=0)
        
        print(f"Overall Accuracy (Unit Level): {overall_acc_unit:.4f}")
        print(f"Overall F1-Score (weighted, Unit Level): {overall_f1_unit:.4f}")

        all_involved_unit_labels = sorted(list(set(y_true_units_original_overall) | set(y_pred_units_original_overall)))

        print("\nOverall Classification Report (Unit Level):")
        unit_overall_report_dict = classification_report(y_true_units_original_overall, y_pred_units_original_overall,
                                                        labels=all_involved_unit_labels, zero_division=0, output_dict=True)
        print(classification_report(y_true_units_original_overall, y_pred_units_original_overall,
                                    labels=all_involved_unit_labels, zero_division=0))

        conf_mat_unit_overall = confusion_matrix(y_true_units_original_overall, y_pred_units_original_overall, labels=all_involved_unit_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat_unit_overall, annot=False, cmap="Greens",
                    xticklabels=all_involved_unit_labels, yticklabels=all_involved_unit_labels)
        plt.title(f"Unit Level - {model_name}", fontsize=TITLE_FONTSIZE)
        plt.xlabel("Predicted Unit", fontsize=LABEL_FONTSIZE)
        plt.ylabel("Actual Unit", fontsize=LABEL_FONTSIZE)
        plt.xticks(rotation=90, fontsize=TICK_FONTSIZE)
        plt.yticks(rotation=0, fontsize=TICK_FONTSIZE)
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_unit_level_overall_{model_name.replace(' ', '_').lower()}.png")
        plt.show()
        plt.close()
    else:
        print("\nNo unit predictions were made to evaluate overall.")

    print(f"\n{'='*50}\nFinished {model_name}\n{'='*50}")
    return f1_section, overall_f1_unit, section_report_dict, unit_overall_report_dict


# Main Execution Block
if __name__ == "__main__":
    print("Starting Company Classification Pipeline")

    source_root_input = input("Please enter the full path to your 'website-data' folder (e.g., /Users/youruser/OVGU/DKE Seminar/website-data): ")
    source_root = Path(source_root_input)

    if not source_root.exists() or not source_root.is_dir():
        print(f"Error: The provided path '{source_root_input}' does not exist or is not a directory.")
        exit()

    project_root = Path("./Company_Classification_Project")
    train_root = project_root / "Training-Data"
    test_root = project_root / "Test-Data"
    tfidf_output_dir = project_root / "TFIDF"

    if project_root.exists():
        print(f"Removing existing project directory: {project_root}")
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    print(f"Created new project directory: {project_root}")

    # 1. Data Splitting
    split_data_into_train_test(source_root, train_root, test_root)

    # 2. Text Cleaning
    clean_dataset(train_root)
    clean_dataset(test_root)

    # 3. Lemmatization
    process_lemmatization(train_root)
    process_lemmatization(test_root)

    # 4. TF-IDF Vectorization
    train_tfidf_path, test_tfidf_path = generate_tfidf_features(train_root, test_root, tfidf_output_dir)

    train_df, test_df = load_and_preprocess_data(train_tfidf_path, test_tfidf_path, COLS_TO_STR)

    lr_section = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', C=0.1, class_weight='balanced')
    lr_unit_builder = lambda: LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', C=0.1, class_weight='balanced')

    svm_section = LinearSVC(max_iter=2000, random_state=42, C=0.1, class_weight='balanced', dual=True)
    svm_unit_builder = lambda: LinearSVC(max_iter=2000, random_state=42, C=0.1, class_weight='balanced', dual=True)

    model_results = {}

    #5. Logistic Regression
    lr_section_f1, lr_unit_f1, lr_section_report_dict, lr_unit_overall_report_dict = run_hierarchical_classification(
        "Logistic Regression", lr_section, lr_unit_builder,
        train_df, test_df,
        section_label_col="section", unit_label_col="unit",
        cols_to_drop_for_features=COLS_TO_DROP_FOR_FEATURES
    )
    model_results["Logistic Regression"] = {
        "Section_F1": lr_section_f1,
        "Unit_F1": lr_unit_f1,
        "Section_Report": lr_section_report_dict,
        "Unit_Overall_Report": lr_unit_overall_report_dict
    }

    #6. LinearSVC (SVM)
    current_svm_section = type(svm_section)(**svm_section.get_params())
    current_svm_unit_builder = lambda: type(svm_unit_builder())(**svm_unit_builder().get_params())
    
    svm_section_f1, svm_unit_f1, svm_section_report_dict, svm_unit_overall_report_dict = run_hierarchical_classification(
        "LinearSVC (SVM)", current_svm_section, current_svm_unit_builder,
        train_df, test_df,
        section_label_col="section", unit_label_col="unit",
        cols_to_drop_for_features=COLS_TO_DROP_FOR_FEATURES
    )
    model_results["LinearSVC (SVM)"] = {
        "Section_F1": svm_section_f1,
        "Unit_F1": svm_unit_f1,
        "Section_Report": svm_section_report_dict,
        "Unit_Overall_Report": svm_unit_overall_report_dict
    }

    print("\nAll hierarchical classification experiments finished!!!")
    print("\nF1-Scores for Comparison")
    
    for model_name, results in model_results.items():
        print(f"{model_name}:")
        print(f"Section Level Weighted F1-Score: {results['Section_F1']:.4f}")

        print(f"Unit Level Weighted F1-Score: {results['Unit_F1']:.4f}")

    #  Overall F1-Scores Comparison Graph
    model_names = list(model_results.keys())
    section_f1s = [model_results[model]["Section_F1"] for model in model_names]
    unit_f1s = [model_results[model]["Unit_F1"] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x - width/2, section_f1s, width, label='Section Level F1-Score', color='skyblue')
    ax.bar(x + width/2, unit_f1s, width, label='Unit Level F1-Score', color='lightcoral')

    ax.set_xlabel('Classification Model', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Comparison of Overall F1-Scores: Logistic Regression vs. SVM', fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE-2)

    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig("f1_score_comparison_overall_lr_svm.png")
    plt.show()
    plt.close()

    #  Section-Level F1-Scores Per Class Comparison Graph
    all_section_labels = sorted(list(set(model_results["Logistic Regression"]["Section_Report"].keys()) |
                                     set(model_results["LinearSVC (SVM)"]["Section_Report"].keys())))
    all_section_labels = [label for label in all_section_labels if label not in ['accuracy', 'macro avg', 'weighted avg']]
    all_section_labels.sort()

    lr_section_f1_per_class = [model_results["Logistic Regression"]["Section_Report"].get(label, {}).get('f1-score', 0.0) for label in all_section_labels]
    svm_section_f1_per_class = [model_results["LinearSVC (SVM)"]["Section_Report"].get(label, {}).get('f1-score', 0.0) for label in all_section_labels]

    fig, ax = plt.subplots(figsize=(max(10, len(all_section_labels) * 0.8), 7))
    x_pos = np.arange(len(all_section_labels))

    ax.bar(x_pos - width/2, lr_section_f1_per_class, width, label='Logistic Regression F1-Score', color='skyblue')
    ax.bar(x_pos + width/2, svm_section_f1_per_class, width, label='LinearSVC (SVM) F1-Score', color='lightcoral')

    ax.set_xlabel('Section Label', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('F1-Score (Per Class)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Comparison of Section-Level F1-Scores Per Class', fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_section_labels, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE-2)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig("f1_score_comparison_section_per_class.png")
    plt.show()
    plt.close()

    #  Unit-Level F1-Scores Per Class Comparison Graph
    all_unit_labels = sorted(list(set(model_results["Logistic Regression"]["Unit_Overall_Report"].keys()) |
                                  set(model_results["LinearSVC (SVM)"]["Unit_Overall_Report"].keys())))
    all_unit_labels = [label for label in all_unit_labels if label not in ['accuracy', 'macro avg', 'weighted avg']]
    all_unit_labels.sort()

    lr_unit_f1_per_class = [model_results["Logistic Regression"]["Unit_Overall_Report"].get(label, {}).get('f1-score', 0.0) for label in all_unit_labels]
    svm_unit_f1_per_class = [model_results["LinearSVC (SVM)"]["Unit_Overall_Report"].get(label, {}).get('f1-score', 0.0) for label in all_unit_labels]

    fig, ax = plt.subplots(figsize=(max(12, len(all_unit_labels) * 0.5), 8))
    x_pos = np.arange(len(all_unit_labels))

    ax.bar(x_pos - width/2, lr_unit_f1_per_class, width, label='Logistic Regression F1-Score', color='skyblue')
    ax.bar(x_pos + width/2, svm_unit_f1_per_class, width, label='LinearSVC (SVM) F1-Score', color='lightcoral')

    ax.set_xlabel('Unit Label', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('F1-Score (Per Class)', fontsize=LABEL_FONTSIZE)
    ax.set_title('Comparison of Unit-Level F1-Scores Per Class', fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_unit_labels, rotation=90, ha="center", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE-2)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig("f1_score_comparison_unit_per_class.png")
    plt.show()
    plt.close()
