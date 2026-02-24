<img width="720" height="65" alt="Screenshot 2026-02-24 at 8 28 56 AM" src="https://github.com/user-attachments/assets/00d71c54-962c-4994-8068-d893d7085e48" />

# Overview
This article is intended to fill in a conceptual gap–something I’ve identified when trying to find good resources that allow an experienced computational biologist, with a general understanding of machine learning, to blend their unique skillset and create a biomarker panel that can be used to predict a given outcome, like treatment response in triple negative breast cancer (TNBC), for example. Importantly, this article assumes that the reader has familiarity with concepts like…

- Basic python programming and scientific computing (if you need a refresher, check out [Python Fundamentals for Biologists](https://github.com/evanpeikon/Python_Fundamentals_Biology)). 
- RNA sequencing, high-throughput proteomics, metabolomics (ie., the types of data we’ll be discussing in this piece). 
- Pairwise and one-vs-all differential expression analysis (what these are, how to interpret the results and what they mean biologically). 
- Machine learning fundamentals (what are classification vs regression methods, what dimensionality reduction is, how to read and interpret basic model evaluation metrics. Ideally, you’ll have at least done some simple projects like predicting Boston housing prices). 

Part of my inspiration for writing this piece is that most of the resources I've previously come across, or written myself, are about isolated components of ML model development such as data processing, algorithm spot-checking, training a model, evaluating it, and tuning it. However, I've yet to come across a resource that explains the big picture–how all of these things fit together, and then guides you through the thinking process at each step from data acquisition to model deployment. That is the goal of this piece. By virtue of that, I see this as a constantly evolving resource that will grow, adapt, and change over time based on what I’ve learned, what readers do / don’t understand on first or second pass, as well as critique and feedback from people with more experience than me. To that effect, if there’s anything that strongly resonates with you in this piece, that you feel is not articulated clearly, or that you feel is not factually correct and/or optimal please feel free to shoot me an email at evanpeikon@gmail.com. 

### How to use this guide
This guide is written to serve two different types of readers, and knowing which one you are will help you get the most out of it.
If you're newer to building ML models on omics data—or if you want to follow along with a complete working example—read it linearly. Every section builds on the previous one, and the code at each step picks up exactly where the last left off. By the end, you'll have a complete, functional pipeline that you can adapt to your own dataset.

If you already have ```sklearn``` experience and are primarily here for the conceptual and domain-specific depth—how to handle proteomics missingness, which algorithms suit omics data and why, what makes a biomarker biologically interpretable—you can use The Big Picture section as a map and jump directly to the steps most relevant to your current problem. Each step is written to be self-contained enough to read in isolation, with the practical best-practice guidance clearly separated from the worked example code.

# The Big Picture
Before working through the step-by-step guide to building a biomarker development pipeline, it's important to understand how all of the different components of model building fit together. This section will provide the high-level, birds eye view, of the process, then in the subsequent sections we'll get into the particulars. The steps described below map directly onto the sections that follow, so you can use this as a roadmap and return to it whenever you need to reorient yourself within the broader pipeline.

The pipeline has a natural three-phase structure, and understanding this structure explains why the steps appear in the order they do.
- **Phase 1 (Steps 1–4)** is about getting your data into a state where it can be used for modeling. Data acquisition gives you raw material; data cleaning removes noise and handles missingness; splitting creates the firewall between what the model learns from and what it will ultimately be judged on; and scaling puts all features on comparable footing. The non-negotiable rule across all four steps is that nothing from your validation or test sets should influence decisions made on your training data—this principle, called preventing data leakage, is easy to violate accidentally and catastrophic when you do, because it produces inflated performance estimates that won't hold up in the real world.
- **Phase 2 (Steps 5–9)** is where the actual model is built. Feature selection narrows the input space to proteins with genuine predictive signal; choosing evaluation methods and metrics defines what "good" means before you start; spot-checking identifies which algorithm families are promising; and hyperparameter tuning squeezes the best performance out of those candidates. Every decision in Phase 2 is made using only training data, with cross-validation standing in for the held-out sets. The order matters here too: you must define your evaluation criteria (Steps 6–7) before you start comparing models (Steps 8–9), otherwise you risk unconsciously selecting metrics that flatter your preferred model.
- **Phase 3 (Steps 10–13)** is about honest assessment and interpretation. The validation set provides a first look at out-of-sample performance and forces a final model selection; the test set provides the one unimpeachable performance estimate you'll report; model interpretation connects the statistical output back to biology; and external validation stress-tests whether the model generalizes beyond the data it was built on. The critical constraint in Phase 3 is irreversibility—once you evaluate on the test set, you're done. Going back to modify the model based on test set results converts it into a second validation set and invalidates your performance claims.

Skipping or reordering steps tends to cause specific, predictable problems. Skipping the split (Step 3) and evaluating on training data makes almost any model look excellent. Doing feature selection before splitting (Step 5 before Step 3) leaks information about the test set into your feature set. Tuning hyperparameters against the test set (using Step 11 to inform Step 9) turns your "final evaluation" into another round of optimization. These aren't random warnings—they are the most common sources of irreproducible biomarker studies in the literature.

### Step-By-Step Guide
- **1. Data Acquisition:** This is exactly what it sounds like. If we’re working with RPPA proteomics data, this is the actual acquisition process, and making sure it’s formatted correctly for the subsequent steps. If using multiple datasets, within the same data type, such as X number of RPPA proteomics datasets, we’d also combine them at this stage as well. 
- **2. Data Cleaning:**  At this stage we’ll remove (or fill in) missing values in our dataset, remove duplicates, encode categorical variables as numerical, or even remove irrelevant features (for example, a column where every data point is identical across samples). 
- **3. Split Dataset:** First, separate the dataset into input features (X) and outputs/targets (y). Then, split the dataset into train, validation, and test sets. A common split is as follows: 64% train, 16% validation, 20% test.
- **4. Scaling/Transformation:** Scale and transform the training dataset (fit the scaler on training data only). These same transformation parameters will later be applied to the validation and test sets.
- **5. Feature Selection/ Dimensionality Reduction:** Reduce total number of features to those with greater predictive potential (this is only done on the training set). 
- **6. Select Evaluation Method:** Select which evaluation method you’ll use to evaluate performance on the training dataset itself. K-fold cross-validation is standard, but other options include leave of out cross validation (LOOCV) and bootstrapping. 
- **7. Select Evaluation Metric:** Select which evaluation metric you’ll use to assess model performance–these vary depending on whether you’re using a classification or regression model. 
- **8. Algorithm Spot Checking:**  Test 3-5 algorithms to see which performs best. 
- **9. Tune Hyperparameters:** Tune hyperparameters from best 2-3 performing models (Step 8) on training dataset. 
- **10 Evaluation On Validation Set:** Evaluate the 2-3 tuned models (step 9) to make final model selection. 
  - **10.5 (Optional):** Retrain the selected model on combined training + validation set to use maximum available data for the final model. 
- **11. Final Evaluation on Test Set:** Final performance evaluation. This can only be done ONCE. These results are reported for final model performance. 
- **12. Interpret Model:** Assess feature importance (SHAP, etc), assess its biological validity and where it succeeds/fails, etc. 
- **13. External Validation:** When possible, evaluate the model on entirely different patient cohorts (ideally from a different institution). 

### Caveats
The above guide assumes you’re working with a single omics data type. If you’re working with multi omics data there are additional steps. 
- **When using [MOFA](https://biofam.github.io/MOFA2/):** When using MOFA: Perform steps 1-2 as normal. At step 3 split each omics dataset separately (same patients, so splits stay aligned), then at step 4 scale each omics type separately. Then fit MOFA on training data only to get factors. You can optionally perform feature selection on the factor matrix, or move directly to step 6. The validation/test datasets will need to be transformed (not fit) with the same MOFA model.
- **When using [DIABLO](https://pubmed.ncbi.nlm.nih.gov/30657866/):** When using DIABLO: Perform steps 1-2 as normal. At step 3 split each omics dataset separately (same patients, so splits stay aligned), then at step 4 scale each omics type separately. Then fit DIABLO on training data only (supervised - uses training labels). Since DIABLO performs feature selection itself, skip step 5 and resume with step 6. The validation/test datasets will need to be transformed (not fit) with the same DIABLO model.

# Step 1—Data Acquisition
Data acquisition is the foundational step where we obtain and prepare our raw data for analysis. In the context of proteomics research, this may involve receiving RPPA (Reverse Phase Protein Array) data, mass spectrometry results, or other high-throughput proteomic measurements from the lab or a collaborator. The first critical task is ensuring the data is properly formatted—rows should represent individual samples (patients), while columns represent features (proteins) plus any clinical or outcome variables.

If you're working with multiple datasets of the same type—for example, RPPA data from multiple institutions or batches—this is the stage where you'll want to combine them. Combining datasets at this early stage allows your model to learn from the technical and biological variation across different sources, which can improve generalization to new datasets. However, combining datasets requires careful consideration of batch effects and technical differences. Tools like ComBat for batch correction or careful documentation of batch identifiers can help manage these technical artifacts while preserving true biological signals.

Beyond simple concatenation, you'll want to ensure that protein identifiers are consistent across datasets (e.g., all using gene symbols vs. Uniprot IDs), that missing values are clearly encoded (typically as NaN or NA rather than zeros), and that any categorical variables use consistent coding schemes. For example, if one dataset codes treatment response as "yes/no" while another uses "1/0", you'll need to harmonize these before proceeding. The goal of this step is to have a single, well-formatted dataset ready for cleaning and analysis.

> **Note:** For this tutorial, we'll use synthetic RPPA proteomics data modeled after a real clinical trial dataset—specifically, data structured similarly to the I-SPY2 trial, a multi-center, adaptive platform trial in early-stage breast cancer. In the real I-SPY2 dataset, RPPA proteomics was used to measure the expression and phosphorylation state of ~200 proteins across hundreds of tumor biopsies collected at the time of diagnosis. The outcome we're predicting is pathologic complete response (pCR), a binary variable indicating whether a patient had no residual invasive cancer remaining after neoadjuvant chemotherapy. Our synthetic dataset mirrors this structure: ~325 patients as rows, ~200 proteins as columns, and a single outcome column. Protein values are log2-transformed RPPA measurements, and the outcome is encoded as 1 (pCR) or 0 (no pCR). We use synthetic data here to allow free sharing and reproducibility, while preserving the statistical properties and clinical context of the real dataset.

# Step 2—Data Cleaning 
Data cleaning is the process of identifying and correcting systematic errors in your data. For example, data cleaning could include tasks such as correcting mistyped data, removing corrupted or missing (i.e., null) data points, removing duplicate data, and sometimes adding missing data values back in. Data cleaning may also include tasks such as encoding categorical variables into integer variables (i.e., turning no/maybe/yes into 0/1/2) or categorical variables in binary variables (i.e., turning male/female into 0/1). 

For what seems like a rote task, data cleaning often requires a high degree of domain expertise. For example, let’s say you have a dataset and VO2 (i.e., volume of oxygen consumption) as one of your features. You may notice a row of data with a VO2 value of 92. As a subject matter expert, you may quickly realize how improbable this value is, whereas someone without domain-specific knowledge of the various measurements may overlook this. Similarly, when looking at missing values in proteomic data you can spot that there is a pattern to the missingness which can give you clues as to whether these values should be dropped, or whether you can impute values in. 

Here I'm going to focus on three basic data-cleaning operations that you’re likely to encounter in the bulk of your machine-learning projects, including encoding categorical variables, removing zero-variance predictors, and removing both null and duplicate values from your dataset. Below you’ll find a short description of each of these tasks plus a few best practices for completing them:
- **Encoding categorical variables:** many machine learning algorithms require inputs to be numerical values, and as a result, a common data cleaning task involves encoding categorical variables into numerical and/or binary variables. For example, turning male and female into 0 and 1. These changes can be made to the CSV directly using the Find and Replace function in Excel or they can be made in your coding notebook. 
- **Removing zero-variance predictors:** zero-variance predictors are input variables where all instances (i.e., rows of data) contain the same value. Inputs without variance offer no predictive value and can be identified with Panda’s nunique() function (zero variance predictors will have one unique value). 
- **Removing null and duplicate values:** rows of data containing no input or duplicated input values offer no predictive value and may hurt model performance. Null values can easily be removed with the dropna() function, while duplicate rows can be removed with the drop_duplicates() function. However, in proteomics data, missing values often have biological meaning—a protein might be "missing" because it's below the detection limit. In these cases, consider imputation strategies (mean, median, or KNN imputation) rather than simply dropping samples or features with missing data.

The above list of data cleaning procedures is not inclusive and is meant to get you started in thinking about data cleaning. Personally, I have a handful of programs I’ve collected over the years that automate data cleaning and provide me without diagnostic outputs I can use to guide this process. If you have any tips or tricks you’ve found useful for this, feel free to share in the comment section below. 

### ### Handling Missing Values in Proteomics Data: A Special Case
Missing values in proteomics data present some unique challenges that require careful consideration of both your instrumentation and the underlying sample's biology. Unlike missing data in many other contexts, "missingness" in proteomics is seldom random. This creates a tension between best practices for avoiding data leakage (which typically means handling missing data after splitting the dataset) and the practical realities of working with proteomics data.
When looking at missing values in proteomic data, you may spot a pattern to the missingness, which can give clues as to whether these values should be dropped, whether you can impute values in, or whether you should leave them be. In proteomics experiments, missing values typically result from one of two sources:
- **Missing completely at random (MCAR):** Some samples may have a high proportion of missing protein measurements (e.g., >50% of proteins unmeasured) due to insufficient tumor material, poor sample quality, or technical failures during sample processing. These samples represent failed experiments rather than a biological signal and should be removed from the dataset entirely before model development. This removal is a quality control step, analogous to excluding failed experiments in any other context, and does not constitute data leakage. For example, suppose you observed NaN values for ERBB2 across ~15% of samples in a dataset of 500 breast cancer patients, but with no clear separation by patient classification. If those same samples also had many other protein columns blank, this would suggest there was not enough tumor material to process — a technical failure rather than a biological signal, and a candidate for removal.
- **Missing not at random (MNAR):** Individual proteins may be missing in specific samples because their expression levels fall below the instrument's detection limit. This is a common source of missingness in proteomics data and is informationally rich — the absence of a measurement often indicates very low or absent protein expression, which may be biologically and clinically relevant. For example, you might notice NaN values for ERBB2 concentrated specifically in patients classified as HR+HER2- and HR-HER2-, suggesting missingness is driven by expression falling below the detection threshold rather than a technical failure. This pattern carries biological meaning: the protein is likely present but at very low levels. For example, the absence of a measurable signal for an oncogene in a tumor sample might indicate good prognosis.

Notice that identifying the cause of missing data isn't just about looking at the data and finding patterns — you also have to sufficiently understand the experimental context, the underlying biology, and potential sources of technical failure, whether during biological sample acquisition or processing. It's for these reasons that I don't advise fully automating data cleaning or outsourcing it to coding agents. While they may better understand the statistical nature of your dataset, they often lack the context needed to (a) understand the cause of missingness and (b) determine how to address it appropriately for a specific problem.
For MNAR-based missingness, a standard approach is to impute missing values with a small fraction of the protein-specific minimum detected value (typically 10% of the minimum for linear-scale data, or minimum minus 1-2 units for log-transformed data). This approach is biologically justified because it represents a plausible lower bound for expression (the protein is present but below detection) and preserves the biological information that "this protein is very low in this sample". Additionally, this approach is protein-specific, respecting that different proteins have different detection ranges and avoids the assumption that missing = zero, which would often be an incorrect interpretation. For log-transformed proteomics data (such as log2-scaled measurements), imputing with the protein-specific minimum minus 1.0 in log space corresponds to expression approximately half as abundant as the minimum detected level in linear space, which represents a biologically reasonable lower bound.

The challenge for machine learning workflows is that this imputation ideally should occur before train-test splitting, which appears to violate the principle of preventing information leakage from test to training data. However, this pre-split imputation is widely accepted in the proteomics biomarker literature for several reasons. For starters, the imputation is based on biological knowledge (below detection = very low) rather than statistical properties of the full dataset. Additionally, the calculation uses only protein-specific minima, not cross-sample statistics that would truly constitute leakage. Most importantly though, attempting to impute separately within train/validation/test splits can lead to different imputation schemes for the same protein across splits, which is biologically inconsistent.

Of course, there is an alternative to imputation. We could also just remove all proteins with any missing values. The problem with this approach is that in most cases it would eliminate nearly the entire dataset as some degree of below-detection missingness is nearly universal in proteomics. As a result, I often use the following three stage approach: 
- **Stage 1:** Remove samples with >20% missing proteins (quality control for MCAR) before splitting. This stage removes samples with insufficient tumor material or severe technical failures, excluding poor-quality samples that don't represent successful experiments.
- **Stage 2:** Remove proteins with >20% missing values before splitting. After removing poor-quality samples, this stage identifies and removes proteins that are frequently unmeasured across the remaining high-quality samples. Proteins missing in more than 20% of samples likely reflect unreliable assays or technical measurement issues (MCAR) rather than biological signals. This ensures that the feature set consists only of proteins that can be reliably measured (after removing bad samples, high missingness in a protein indicates a fundamental measurement problem rather than sample-specific issues).
- **Stage 3:** Impute remaining sparse missing values (representing MNAR below-detection measurements) before splitting. At this point, remaining missing values are sparse (typically <10% per protein) and likely represent true biological phenomena where specific samples have expression below the detection limit. For log-transformed data, impute with the protein-specific minimum log value minus 1.0 (corresponding to half the minimum expression in linear space). For linear-scale data, use 10% of the protein-specific minimum. While this technically uses information from the full dataset, it is justified by the biological nature of the imputation and the practical impossibility of the alternative (we can't train models with missing values).

Regardless of how you handle this approach you have to be clear in documenting your rationale, explaining all three types of missingness (sample-level MCAR, protein-level MCAR, and MNAR) and justifying the pre-split imputation based on the MNAR nature of below-detection measurements. You should also acknowledge that this represents a pragmatic compromise between statistical ideals and biological realities and when possible perform sensitivity analyses showing that your results are robust to alternative imputation strategies. For other omics data types (transcriptomics, metabolomics), similar principles apply, though the specific imputation strategies may differ based on the measurement technology and the meaning of missingness in that context. The key is always to understand the biological and technical sources of missingness in your specific dataset and to choose imputation strategies that respect the underlying biology while minimizing the risk of introducing spurious patterns that could compromise model generalization.

### Example Code
The code block below implements the three-stage framework above. Notably, this code assumes that your data is formatted with patients as rows, stored in the index, proteins as columns, and one additional column for the output you’re predicting. 

```python
def clean_proteomics_data(data, output_name, sample_threshold=0.40, protein_threshold=0.20, log_transformed=True):
   # Remove samples with missing outcome
   n_start = len(data)
   data_clean = data.dropna(subset=[output_name])
   n_removed_outcome = n_start - len(data_clean)
   if n_removed_outcome > 0:
       print(f"\nRemoved {n_removed_outcome} samples with missing outcome")
  
   # Separate x and y for processing
   X = data_clean.drop(output_name, axis=1)
   y = data_clean[output_name]
   print(f"\nStarting: {X.shape[0]} samples × {X.shape[1]} proteins")
   print(f"Total missing: {X.isna().sum().sum()} values")
  
   # Stage 1: remove samples with high missingness
   sample_missing_pct = X.isna().mean(axis=1)
   sufficient_samples = sample_missing_pct <= sample_threshold
   n_removed_samples = (~sufficient_samples).sum()
   X_stage1 = X[sufficient_samples]
   y_stage1 = y[sufficient_samples]
   print(f"\nStage 1: Removed {n_removed_samples} samples (>{sample_threshold*100:.0f}% missing)")
  
   # Stage 2: remove proteins with high missingness
   protein_missing_pct = X_stage1.isna().mean(axis=0)
   reliable_proteins = protein_missing_pct <= protein_threshold
   n_removed_proteins = (~reliable_proteins).sum()
   X_stage2 = X_stage1.loc[:, reliable_proteins]
   print(f"Stage 2: Removed {n_removed_proteins} proteins (>{protein_threshold*100:.0f}% missing)")

   # Stage 3: impute remaining missing values
   n_to_impute = X_stage2.isna().sum().sum()
   if n_to_impute > 0:
       X_imputed = X_stage2.copy()
       for protein in X_imputed.columns:
           if X_imputed[protein].isna().any():
               min_val = X_imputed[protein].min()
               if log_transformed:
                   impute_val = min_val - 1.0
               else:
                   impute_val = 0.1 * min_val
               X_imputed[protein].fillna(impute_val, inplace=True)
       impute_method = "min - 1.0 (log2)" if log_transformed else "0.1 × min"
       print(f"Stage 3: Imputed {n_to_impute} values ({impute_method})")
   else:
       X_imputed = X_stage2.copy()
       print(f"Stage 3: No imputation needed")
  
   # Reconstruct full dataframe
   data_final = X_imputed.copy()
   data_final[output_name] = y_stage1
  
   # Summary
   print(f"\n{'='*60}")
   print(f"Final: {data_final.shape[0]} samples × {X_imputed.shape[1]} proteins")
   print(f"Missing values: {X_imputed.isna().sum().sum()}")
   return data_final

# Usage example
proteomic_data_clean = clean_proteomics_data(proteomic_data, output_name='pCR', sample_threshold=0.40, protein_threshold=0.20, log_transformed=True)
```
```
Starting: 324 samples × 176 proteins
Total missing: 11685 values

Stage 1: Removed 109 samples (>20% missing)
Stage 2: Removed 46 proteins (>20% missing)
Stage 3: Imputed 182 values (min - 1.0 (log2))

============================================================
Final: 215 samples × 130 proteins
Missing values: 0
```

# Step 3– Splitting The Dataset 
In a sense, this is the first true step of building our machine learning model and its importance cannot be understated. In order to ensure any model we build is generalizable to new, unseen, data we have to split the dataset we acquired and cleaned in the previous steps. In practice, there are two sequential operations when splitting data.

First, we separate the dataset into inputs (features, denoted as X) and outputs (targets, denoted as y). For example, if we're predicting treatment non-response from proteomic data, our inputs/features (X) would be all of our protein expression columns—typically hundreds to thousands of proteins measured across all samples. Our outputs/targets (y) would be our non-response indicator, such as whether a patient achieved pathologic complete response (no = 0, yes = 1), or we could even use something like a residual cancer burden index (0-3 score, with zero indicating no residual cancer). For regression problems predicting survival, y might be recurrence-free survival time in days. The critical point is that X and y must be perfectly aligned—the first row of X must correspond to the first element of y, representing the same patient (as a result, we need to make sure to split X and y simultaneously using the same random seed to maintain alignment). 

Second, after separating X and y, we split both into three parts: a training dataset, a validation dataset, and a testing dataset. A common split is as follows: 64% train, 16% validation, 20% test. The training set (64%) is used for model development—fitting algorithms, performing cross-validation, and tuning hyperparameters. The validation set (16%) is used to compare different modeling approaches and select the best final model. The test set (20%) is reserved exclusively for the final performance evaluation and should only be used once at the very end of the pipeline. 

In practice, this step is simple. But, there are some traps that are easy to fall into. For example, let’s say your dataset contains 1000 patient samples (stored in rows) from breast cancer patients. Because of how the dataset is formatted, the first 250 rows are all HR+/HER2- patients, the next 250 are HR-/HER2-, then the following sets of 250 are composed of HR+/HER2+ and HR-/HER2+ patients. If we just naively split the dataset we could inadvertently end up in a scenario where all of our training samples are from one or two HR/HER2 subtypes and the same for our validation and testing data. In order to avoid this, it’s often helpful to shuffle both the rows and columns (breaking up proteins organized by family), to ensure we remove a potential source of confounding. 

### Example Code
The code block below demonstrates a basic function that can be used for splitting your dataset. Notably, this assumes that your data is formatted with patients as rows, stored in the index, proteins as columns, and one additional column for the output you’re predicting. Additionally, it assumes your data has already been cleaned, as per step 2, and that there are no missing values or duplicates:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_test_split(data, output_name='', test_size_1=0.20, test_size_2=0.20):
 # Separate x (protein columns) and y (outcome column)
 x = data.drop(output_name, axis=1)
 y = data[output_name]

 # Shuffle rows (patients) in case data is ordered by subtype
 np.random.seed(13)
 shuffled_indices = np.random.permutation(x.index)
 x_shuffled = x.loc[shuffled_indices]
 y_shuffled = y.loc[shuffled_indices]

 # Split 1: Separate test set (20%)
 x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(
   x_shuffled, y_shuffled, test_size=test_size_1, stratify=y_shuffled, random_state=13)

 # Split 2: Split x_train_and_val into training (64%) and validation (16%)
 x_train, x_val, y_train, y_val = train_test_split(
   x_train_and_val, y_train_and_val, test_size=test_size_2, stratify=y_train_and_val, random_state=13)
 # Summary
 print(f"\nTraining set:")
 print(f"  X_train shape: {x_train.shape}")
 print(f"  y_train shape: {y_train.shape}")
 print(f"  Proportion of total: {len(x_train)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_train.value_counts().to_dict()}")
 print(f"\nValidation set:")
 print(f"  X_val shape: {x_val.shape}")
 print(f"  y_val shape: {y_val.shape}")
 print(f"  Proportion of total: {len(x_val)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_val.value_counts().to_dict()}")
 print(f"\nTest set:")
 print(f"  X_test shape: {x_test.shape}")
 print(f"  y_test shape: {y_test.shape}")
 print(f"  Proportion of total: {len(x_test)/len(x)*100:.1f}%")
 print(f"  Class distribution: {y_test.value_counts().to_dict()}")
 return x_train, x_val, y_train, y_val, x_test, y_test

# Usage example 
x_train, x_val, y_train, y_val, x_test, y_test= train_val_test_split(proteomic_data_clean, output_name='pCR', test_size_1=0.20, test_size_2=0.20)
```
```
Training set:
  X_train shape: (137, 130)
  y_train shape: (137,)
  Proportion of total: 63.7%
  Class distribution: {1: 69, 0: 68}

Validation set:
  X_val shape: (35, 130)
  y_val shape: (35,)
  Proportion of total: 16.3%
  Class distribution: {0: 18, 1: 17}

Test set:
  X_test shape: (43, 130)
  y_test shape: (43,)
  Proportion of total: 20.0%
  Class distribution: {0: 22, 1: 21}
```

# Step 4–Data Scaling/Transformation
Many machine learning algorithms perform better when numerical input variables in your dataset are scaled to a standard range. For example, algorithms that use a weighted sum of input variables, such as linear regression, logistic regression, and deep learning, are impacted by the scale of input data as are  algorithms that rely on distance measures between samples, such as k-nearest neighbors (KNN) and support vector machines (SVM). As a result, it is nearly always advantageous to scale numerical input values to a standard range, hence the need for data transformation techniques.

Data transformation is a process used to change a given dataset's scale or distribution of features. The two primary techniques used to change the scale of data are standardization transforms and normalization transforms. Standardization is used to transform data attributes with a Gaussian (i.e., normal) distribution and differing means and standard deviations, so all attributes have a mean of 0 and a standard deviation of 1. Additionally, standardization is most suitable when performing linear regression, logistic regression, or linear discriminate analysis.

The formula for data standardization is as follows: Standardized value = (valueᵢ - mean) / standard deviation, where mean and standard deviation are calculated as follows: mean = Σ valueᵢ / count(values) and standard deviation = √(Σ (valuesᵢ - mean)² / counts(values)-1). 

Normalization (re-scaling) is best used when your input data contains attributes with varying scales. For example, if one attribute has values ranging from 0-20, while another has values ranging from 655-9023. In these cases, normalization puts all data attributes on the same scale, which is usually 0-1. Normalization is best used for optimization algorithms such as gradient descent. However, it can also be used before algorithms that weigh inputs, such as neural networks, and algorithms that use distance components, such as k-nearest neighbors (KNN). The formula for data rescaling (normalization) is as follows: normalized value = (valueᵢ - min) / (max - min).

Whereas standardization and normalization are used to alter the scale of data, power transforms and quantile transforms are used to change the data’s distribution. Power transforms are used to change on input variables that are nearly Gaussian, but slightly skewed, and make them Gaussian. Power transforms are useful since many algorithms such as gaussian naive bayes, or even simple linear regression, assume that numerical input variables have a Gaussian (i.e., normal) distribution . Thus, using a power transform can improve your model’s performance when the input variables are skewed towards a non-gaussian probability distribution. Quantile transforms, on the other hand, are used to force variables with unusual ‘organic’ distributions into uniform or Gaussian distribution patterns. For example, it’s a common practice to use a quantile transform on input variables with outlier values since ordinary standardization techniques can skew the data in a maladaptive manner. 

In practice when working with transcriptomic, proteomic, or metabolomic data, the best practice is to use standardization (StandardScaler in scikit-learn) after log2-transformation. Most data is already log2-transformed when you receive it, as this transformation helps normalize the typically right-skewed distribution of biological measurements. If your data is not already log-transformed, apply log2-transformation first, then standardize. This two-step process is nearly universal in omics ML applications because it handles both the scale differences between features and the distributional properties of biological data. Critically, fit the scaler only on your training data, then apply the same transformation parameters to validation and test sets. This ensures that your validation and test sets are transformed using the exact same parameters (mean and standard deviation) learned from the training set, preventing data leakage.

### Example Code
The code block below x_train, x_val, and x_test datasets from step 3, then fits the scaler on the training data, before applying it to all three datasets. The resultant output contains three scaled input datasets with a mean of 0 and standard deviation of 1:
```python
def scale_proteomics_data(X_train, X_val, X_test, log_transformed=True):
   from sklearn.preprocessing import StandardScaler
   import numpy as np
  
   # Step 1: log transformation if needed
   if not log_transformed:
       X_train_log = np.log2(X_train + 1)
       X_val_log = np.log2(X_val + 1)
       X_test_log = np.log2(X_test + 1)
      
       # Preserve DataFrame structure
       X_train_log = pd.DataFrame(X_train_log, index=X_train.index, columns=X_train.columns)
       X_val_log = pd.DataFrame(X_val_log, index=X_val.index, columns=X_val.columns)
       X_test_log = pd.DataFrame(X_test_log, index=X_test.index, columns=X_test.columns) 
   else:
       X_train_log = X_train.copy()
       X_val_log = X_val.copy()
       X_test_log = X_test.copy()
  
   # Step 2: standardization (fit on training data only)
   print("\nApplying standardization (z-score)...")
   print("  Fitting scaler on training data only...")
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train_log)
   print("  Applying scaler to validation data...")
   X_val_scaled = scaler.transform(X_val_log)
   print("  Applying scaler to test data...")
   X_test_scaled = scaler.transform(X_test_log)
  
   # Convert back to DataFrames
   X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_log.index, columns=X_train_log.columns)
   X_val_scaled = pd.DataFrame(X_val_scaled,index=X_val_log.index,columns=X_val_log.columns)
   X_test_scaled = pd.DataFrame(X_test_scaled,index=X_test_log.index, columns=X_test_log.columns)
  
   # Verification
   print(f"\nTraining set (should have mean≈0, std≈1):")
   print(f"  Mean: {X_train_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_train_scaled.std().mean():.6f}")
   print(f"\nValidation set (will differ - uses train parameters):")
   print(f"  Mean: {X_val_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_val_scaled.std().mean():.6f}")
   print(f"\nTest set (will differ - uses train parameters):")
   print(f"  Mean: {X_test_scaled.mean().mean():.6f}")
   print(f"  Std:  {X_test_scaled.std().mean():.6f}")
   return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Usage example:
X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_proteomics_data(x_train, x_val, x_test, log_transformed=True)
```
```
Applying standardization (z-score)...
  Fitting scaler on training data only...
  Applying scaler to validation data...
  Applying scaler to test data...

Training set (should have mean≈0, std≈1):
  Mean: 0.000000
  Std:  1.000863

Validation set (will differ - uses train parameters):
  Mean: -0.003324
  Std:  1.040088

Test set (will differ - uses train parameters):
  Mean: -0.040258
  Std:  0.960544
```

# Step 5–Feature Selection/ Dimensionality Reduction
Feature selection and dimensionality reduction are both techniques used to reduce the number of different input variables in a dataset. 

## Feature Selection
Feature selection is the process of selecting features in a given dataset that best contribute to predicting the desired outcome variable in your model and removing redundant or irrelevant features that decrease the accuracy of your model’s predictions. As a general rule of thumb, using only as much data as necessary to make a prediction is desirable, and we should strive to produce the simplest well-performing model.

> "“Many models, especially those based on regression slopes and intercepts, will estimate parameters for every term in the model. Because of this, the presence of non- informative variables can add uncertainty to the predictions and reduce the overall effectiveness of the model.” -Kuhn & Johnson (2013). Applied Predictive Modeling"

Feature selection techniques are subdivided into two major groups: those that use the target variable to select features (supervised learning) and those that do not (unsupervised learning). In this piece, I will only discuss supervised learning methods, which can be further divided into filter, wrapper, and intrinsic methods. 

**Filter-based feature selection methods** use statistical tests to score the correlation between input variables (i.e., features) and the selected output variable (i.e., target) and select the features that have the strongest relationship with the target. These methods are fast and effective, but the choice of statistical tests used to score features depends on both the input and output data types. As a result, it can be challenging to select appropriate statistical measures when performing filter-based feature selection. The chart below provides a simple heuristic for selecting appropriate statistical tests for filter-based methods: 

<img width="417" height="172" alt="Screenshot 2026-02-24 at 10 11 50 AM" src="https://github.com/user-attachments/assets/552e798a-dd0d-4f60-bc76-c652b84bb693" />

An example filter method is ```SelectKBest```, which can be used to select features in a dataset using an ANOVA F-value as a statistical test.  Additionally, in the context of working with proteomics data where we are trying to predict treatment non-response (classification) or distance recurrence free survival (DRFS) we may also use the outputs from a differential expression analysis for this purpose. For example, identifying proteins that are up or down regulated in non responders versus responders and utilizing those as our features in downstream models. Additionally, we could even perform a weighted co-expression analysis and identify proteins that are hubs in non-responder networks, but not responders, and use those as features as well. 

Unlike filter-based feature selection methods, **wrapper methods** explicitly choose features that result in the best-performing model. Wrapper feature selection methods create many different models with different sets of input features and then select the features that result in the best-performing model. The advantage of wrapped-based methods is that they are unconcerned with data types, making them easy to implement for practitioners. However, wrapper methods are more computationally expensive. A popular example of a  wrapper method is recursive feature elimination, which eliminates the least valuable features in a dataset one by one until a specified number of strong features remain. Recursive feature elimination is a popular feature selection technique because it’s easy to configure and understand the results, and it’s effective for selecting input features that are most relevant for predicting the target variable. Additionally, two hyper-parameters can be configured during recursive feature elimination, which will impact your results: the specific algorithm used to select features and the number of features you want to select.

Finally, **intrinsic methods** automatically select features as part of fitting the chosen machine learning model and include machine learning algorithms such classification and regression trees. 

In practice when working with transcriptomic, proteomic, or metabolomic data, the most effective approach is typically a hybrid strategy that combines biological knowledge with statistical feature selection. Start with a filter method based on differential expression analysis to identify proteins or genes that show significant changes between your outcome groups (e.g., responders vs. non-responders). This leverages your domain expertise and reduces the feature space to biologically relevant candidates—for example, from 5,000 proteins down to 200-500 that are differentially expressed at FDR < 0.05. Then, apply an embedded method like LASSO (Lasso regression or LassoCV) to this filtered set, which will automatically select the most predictive features while avoiding overfitting. This two-stage approach combines biological interpretability with statistical rigor, typically yielding a final feature set of 10-50 proteins that form your biomarker signature. Importantly, perform all feature selection exclusively on your training data to avoid data leakage.

## Dimensionality Reduction
A related concept to feature selection is that of dimensionality reduction, which  is a data preparation technique that compresses high-dimensional data into a lower-dimensional space, reducing the number of input variables. The key difference between feature selection and dimensionality reduction is that feature selection decides which input variables to keep or remove from the dataset. In contrast, dimensionality reduction creates a projection of the data resulting in entirely new input features.

Unlike feature selection, the variables you are left with after performing dimensionality reduction are not directly related to the original input variables (i.e., features), making it difficult to interpret the results. As such, dimensionality reduction is an alternative to feature selection rather than a type of feature selection.\

For simplicity's sake, you can think of dimensions as the number of input features in a dataset. Many issues can arise when analyzing high-dimensional data that otherwise wouldn't be an issue in lower-dimensional space. This problem is called the curse of dimensionality and is the reason we employ dimensionality reduction techniques, such as principle component analysis (PCA), which is an unsupervised learning technique that uses linear algebra to reduce the dimensionality of a dataset by removing linear dependencies (i.e., strong correlations) between input variables. Reducing the dimensionality of a dataset results in fewer input values, which simplifies predictive models and helps them perform better on unseen data.

An important property of principal component analysis is the ability to choose the number of dimensions (i.e., principal components) the dataset will have after the algorithm has transformed it. Again, it's important to note that principal components are not features in your original dataset. Instead, a principal component can include attributes from multiple different features. Thus, principal component analysis aims to tell you what percent of the variance each principal component accounts for in your target outcome. 

### Example Code
For this worked example, we apply LASSO (L1-regularized logistic regression) directly to the scaled training data from Step 4. While the hybrid approach described above—starting with differential expression analysis before applying LASSO—is generally preferred in practice, we use LASSO alone here to keep the example self-contained and to demonstrate how regularization itself can drive feature selection. The key parameter is C, the inverse of regularization strength: smaller C values impose stronger regularization and select fewer features, while larger C values are more permissive. Here we use C=0.2, which yields a compact, interpretable feature set.
```python
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectFromModel

def select_features_lasso(X_train, X_val, X_test, y_train, C=1.0, task='classification', n_display=20):
   print(f"\nTask: {task}")
   print(f"Starting features: {X_train.shape[1]}")
  
   # Select appropriate LASSO model
   if task == 'classification':
       print(f"Regularization parameter C: {C}")
       lasso_model = LogisticRegression( penalty='l1', solver='liblinear', C=C, random_state=13, max_iter=1000)
       lasso_model.fit(X_train, y_train)
       coefficients = lasso_model.coef_[0]
      
   elif task == 'regression':
       print("Using LassoCV with cross-validation for alpha selection")
       lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
       lasso_model.fit(X_train, y_train)
       print(f"Optimal alpha: {lasso_model.alpha_:.6f}")
       coefficients = lasso_model.coef_
   else:
       raise ValueError("task must be 'classification' or 'regression'")
  
   # Get selected features (non-zero coefficients)
   selected_mask = coefficients != 0
   selected_features = X_train.columns[selected_mask].tolist()
   print(f"\nFeatures selected: {len(selected_features)}")
  
   # Fallback if no features selected
   if len(selected_features) == 0:
       print("\n No features selected with strict threshold")
       print("→ Using SelectFromModel with 'median' threshold")
       selector = SelectFromModel(lasso_model, threshold='median', prefit=True)
       selected_mask = selector.get_support()
       selected_features = X_train.columns[selected_mask].tolist()
       print(f"Features selected: {len(selected_features)}")
  
   # Apply selection to all sets
   X_train_selected = X_train[selected_features]
   X_val_selected = X_val[selected_features]
   X_test_selected = X_test[selected_features]
  
   # Display top features
   print(f"\n{'='*60}")
   print(f"Top {min(n_display, len(selected_features))} Selected Features")
   print(f"{'='*60}")
  
   # Get feature importances (absolute coefficients)
   feature_coefs = [(feat, coefficients[X_train.columns.get_loc(feat)])
                    for feat in selected_features]
   feature_coefs_sorted = sorted(feature_coefs, key=lambda x: abs(x[1]), reverse=True)
   print(f"\n{'Feature':<40} {'Coefficient':>12} {'Direction':>10}")
   print("-"*60)
   for i, (feature, coef) in enumerate(feature_coefs_sorted[:n_display], 1):
       direction = "↑ (pos)" if coef > 0 else "↓ (neg)"
       print(f"{i:2d}. {feature:<35} {coef:>12.4f} {direction:>10}")
   if len(selected_features) > n_display:
       print(f"\n... and {len(selected_features) - n_display} more features")
  
   # Summary
   reduction_pct = (1 - len(selected_features)/X_train.shape[1]) * 100
   print(f"\nReduction: {reduction_pct:.1f}% ({X_train.shape[1]} → {len(selected_features)} features)")
   return X_train_selected, X_val_selected, X_test_selected, selected_features, lasso_model

# Usage example:
X_train_selected, X_val_selected, X_test_selected, selected_features, lasso_model = select_features_lasso(X_train_scaled, X_val_scaled,  X_test_scaled, y_train, C=0.2, task='classification', n_display=10
```
```
Task: classification
Starting features: 130
Regularization parameter C: 0.2

Features selected: 25

============================================================
Top 10 Selected Features
============================================================

Feature                                   Coefficient  Direction
------------------------------------------------------------
 1. Tuberin_TSC2 Y1571                       -0.5013    ↓ (neg)
 2. mTOR S2448                                0.3027    ↑ (pos)
 3. IkB alpha S32/S36                         0.3005    ↑ (pos)
 4. H2A.X S139                               -0.2561    ↓ (neg)
 5. cKIT Y703                                -0.2350    ↓ (neg)
 6. cABL Y245                                 0.2198    ↑ (pos)
 7. AMPKb1 S108                               0.2129    ↑ (pos)
 8. AXL Y702                                  0.1564    ↑ (pos)
 9. HLA-DR total                              0.1518    ↑ (pos)
10. NF2R1/COUP1 total                         0.1205    ↑ (pos)

... and 15 more features

Reduction: 80.8% (130 → 25 features)
```

# Step 6–Evaluation Methods
Imagine you're developing a machine learning model to predict whether a newly designed protein sequence will fold into a stable structure. You train your model on a database of known protein sequences and their stability measurements. The model performs beautifully on this training data, correctly predicting stability 99% of the time. Success, right? Not necessarily. This scenario illustrates one of the fundamental challenges in machine learning: overfitting. Just as a biology student might memorize specific exam questions without understanding the underlying concepts, a machine learning model can become too specialized in handling its training data while failing to generalize to new, unseen cases. This is particularly problematic in biotechnology, where our models often need to make predictions about novel compounds, sequences, or cellular behaviors.

To ensure our models perform well on new, unseen, data we should evaluate them during the development phase, which is separate from the validation and test set evaluation that comes later in the pipeline. Remember, at this stage we've already split our data into train/validation/test sets (Step 3). The evaluation methods discussed here are used WITHIN the training set during model development (Steps 8-9), while the validation and test sets remain completely untouched.

The purpose of these evaluation methods is to get an honest estimate of how well our algorithm will perform on unseen data without actually using our held-out validation or test sets. This can be accomplished through resampling techniques that allow us to make multiple evaluations on different subsets of our training data.

One of the most common approaches to evaluation is k-fold cross-validation, which is an evaluation technique that splits your training dataset into k-parts to estimate the performance of a machine learning model.  In the biotech industry, where data collection is often expensive and time-consuming, making the most of limited data is crucial. K-fold cross-validation offers a robust solution for this by allowing us to run multiple experimental replicates “in silico”. Instead of just splitting the training data set into two sub-segments used to build and evaluate our model, we divide it into k parts and perform k different evaluations, each time using a different portion as the test set. So for example, if k=4, then we evaluate our model four times and each time a different fold is used for testing while the other three are used to train the model, as demonstrated in the image below. This results in four performance scores, which can be averaged to determine how well your model performs.

<img width="412" height="148" alt="Screenshot 2026-02-24 at 10 13 55 AM" src="https://github.com/user-attachments/assets/1d4c5009-17aa-4ce7-b2bc-3e03fe6423eb" />
> This figure depicts how k-fold cross validation works using k=4 as an example. Note than on each fold there are k-1 trainng splits and 1 testing split. 

K-fold cross-validation is particularly valuable when working with heterogeneous biological data, where a single split might not be representative. When performing k-fold cross-validation, the most critical decision you'll have to make is how large your k is (i.e., how many splits or folds you create). For moderate to large-sized datasets (thousands or tens of thousands of rows), k-values of 5-10 are common, whereas k-values of 3-5 are more appropriate for smaller datasets. But, what happens if your dataset is so small that splitting it into multiple components results in biased performance? This is where leave-one-out cross validation (LOOCV) comes in. LOOCV takes k-fold cross-validation to its extreme, setting k equal to the number of samples. While computationally expensive, this approach is often worth the cost in scenarios where data collection is the primary bottleneck either due to cost or scarcity of samples. Additionally, there are other methods worth considering, such as bootstrapping which allows you to estimate a statistics variability when working with small datasets, where splitting the data into distinct folds would leave too little for training. 

In practice though, selecting the appropriate evaluation methods for your needs requires careful consideration of several interrelated factors. The size of your dataset often serves as the primary guiding factor. When working with large datasets, which is typically the case in high-throughput sequencing experiments, k-fold cross-validation is often an ideal choice. In cases where data is particularly scarce, such as rare disease studies or expensive clinical trials with few samples, leave-one-out cross-validation becomes my go to, despite its computational cost. For most proteomics biomarker development projects with 50-200 samples, 5-fold or 10-fold cross-validation on the training set strikes the right balance between computational efficiency and robust performance estimation.

# Step 7–Evaluation Metrics
In the previous step we discussed evaluation methods, which help us estimate how well a given machine learning algorithm will perform at making predictions about unseen data with specific examples including k-fold cross-validation and leave one out cross-validation. Whereas evaluation methods estimate a model's performance on unseen data, evaluation metrics are the statistical techniques employed to quantify how well the model works. Thus, to evaluate a machine learning algorithm, we must select an evaluation method and evaluation metric(s). 

## Evaluation Metrics for Classification Models
In this section we’ll cover a handful of common evaluation metrics for classification models, including classification accuracy, confusion matrices, area under the ROC curve, and logistic loss. To start, **classification accuracy** is the percentage of all predictions that are made correctly. Because it's so easy to calculate and interpret, classification accuracy is the most commonly used evaluation metric for classification problems. However, classification accuracy is only effective when there are equal numbers of observations in each output class, which is seldom the case, and thus classification accuracy is often misused. For example, let's say we have 100 patients, 10 of whom are diseased. Our machine learning algorithm predicted that 95 patients were healthy and 5 were diseases. In this case, our classification accuracy would be 90%, which while true, is misleading given that we misclassified 50% of diseased patients (a catastrophic outcome in real life). 

Next, we have the **confusion matrix**, which is a table that summarizes the prediction results for a classification problem with the predicted values on the x-axis and actual values on the y-axis as demonstrated below. 

<img width="434" height="209" alt="Screenshot 2026-02-24 at 10 15 19 AM" src="https://github.com/user-attachments/assets/caae60b8-d687-4b9b-8363-d0e5b043d05f" />

For example, let's say we're predicting whether or not a patient has responded to cancer treatment. If we predict the patient is a responder, and they do respond, then it's a true positive. However, if we predict the patient is a responder but they do not respond, then it's a false positive (type 1 error). 

Next we have the **classification report**, which provides a snapshot of a machine learning model’s performance on classification problems. Specifically, a classification report displays the precision, recall, f1-score, and support (the number of actual occurrences for a given class). One benefit of a classification report is that it’s easy to interpret - the higher the precision, recall, and f1-score, the better. Using the sklearn ```classification_report``` function, you’ll get an output like:

<img width="543" height="158" alt="Screenshot 2026-02-24 at 10 15 59 AM" src="https://github.com/user-attachments/assets/e06c3474-40b4-4b63-b37a-aecbb226adb2" />

Another classification metric is the **area under the ROC curve** which is an evaluation metric for binary classification problems representing a machine learning model's ability to discriminate between output classes. A model's area under the ROC curve is scored from 0.5 to 1.0, where higher scores indicate a greater ability to make correct predictions. For example, a score of 1.0 would mean that a machine learning model perfectly identifies patients with/without a disease, whereas a score of 0.5 would equate to random guessing. 

Finally, we have **logistic loss** (aka logloss) is an evaluation metric that assesses the predictions of probabilities of membership to a given output class. Logistic loss represents the confidence for a given algorithm's predictive capabilities and is scored from 0-1. Additionally, correct and incorrect predictions are rewarded in proportion to the confidence of the prediction. The closer the logloss score is to 1, then more the predicted probability diverges from the actual value. Alternatively, a logloss value of 0 indicates perfect predictions. 

When working with transcriptomic, proteomic, or metabolomic data for classification tasks, the best practice is to report multiple complementary metrics rather than relying on accuracy alone. For binary classification problems (responder vs. non-responder), prioritize ROC-AUC as your primary metric since it's robust to class imbalance, which is common in clinical datasets where one outcome may be much rarer than the other. Always examine the confusion matrix to understand where your model succeeds and fails—for example, you might discover your model correctly identifies 90% of responders but only 40% of non-responders, indicating a need for class balancing strategies like SMOTE or class weights. Additionally, report precision and recall for the minority class (typically non-responders or disease cases), as these metrics directly address clinical relevance. In proteomics biomarker development, a false negative (missing a non-responder) often has different consequences than a false positive (incorrectly predicting non-response), so understanding these trade-offs through precision-recall curves is essential for clinical translation.

## Evaluation Metrics for Regression Models
Regression is a data analysis technique that uses known data points to predict a single unknown but related data point—for example, predicting someone's distance recurrence free survival in days based on proteomic measurements. In this section we’ll cover a handful of common evaluation metrics for classification models, including mean absolute error (MAE), root mean squared error (RMSE) and r-squared. 

The mean absolute error (MAE) is an easy way to determine how wrong your predictions are when solving regression problems. Specifically, the MAE quantifies the error in your predicted values versus the actual, expected values. 

The mean absolute error is defined as the average of the sum of absolute errors. The term 'absolute' conveys that the error values are made positive, so they can be added to determine the magnitude of error. Thus, one limitation of MAE is that there is no directionality to the errors, so we don't know if we're over or underpredicting.  The formula for mean absolute error (MAE) is as follows:

$MAE = \frac{\Sigma^{n}_{i=1}abs(predicted_i-actual_i)}{\text{total predictions}}$

Next, the **root mean squared error (RMSE)** is one of the most commonly used evaluation metrics for regression problems. RMSE is the square root of the mean of squared differences between actual and predicted outcomes. Squaring each error forces the values to be positive. Calculating the square root of the mean squared error (MSE) returns the evaluation metric to the original units, which is useful for presentation and comparison. The formula for root mean squared error (RMSE) is as follows:

$RMSE = \sqrt{\frac{\Sigma^{n}_{i=1}abs(predicted_i-actual_i)^2}{\text{total predictions}}}$

Finally, $R^2$, also known as the **coefficient of determination**, is an evaluation metric that provides an indication of the goodness of fit of a set of predictions as compared to the actual values. The R² metric is scored from 0 to 1, where 1 is a perfect score and 0 means the predictions are entirely random and arbitrary.

When working with transcriptomic, proteomic, or metabolomic data for regression tasks (such as predicting survival time or tumor burden), the best practice is to report both RMSE and R² as complementary metrics. RMSE is particularly valuable because it's in the same units as your outcome variable (e.g., months of survival), making it clinically interpretable—you can say "our model's predictions are typically off by X months." However, RMSE alone doesn't tell you how much better your model is compared to a naive baseline (like always predicting the mean). This is where R² becomes essential, as it quantifies the proportion of variance in survival time that your proteomic features explain. For clinical biomarker panels, also consider concordance index (C-index) for survival prediction, which measures how well your model ranks patients by risk. Additionally, examine residual plots to identify if your model systematically over- or under-predicts for certain patient subgroups, as this can reveal important biological insights or the need for stratified models.

# Step 8–Algorithm Spot Checking
For any machine learning problem, we must select an algorithm to make predictions, an evaluation method to estimate a model's performance on unseen data, and an evaluation metric(s) to quantify how well the model works.  Unfortunately, we can't always know which algorithm will work best on our dataset beforehand. As a result, we have to try several algorithms, then focus our attention on those that seem most promising. Thus, it's important to have quick and easy ways to assess and compare different algorithms' performance before we select one to tune and optimize - this is where spot-checking comes in. 

Spot-checking is a way to quickly discover which algorithms perform well on your machine-learning problem before selecting one to commit to. Generally, I recommend that you spot-check five to ten different algorithms using the same evaluation method and evaluation metric to compare the model's performance. 

## Spot-Checking for Classification Models
A classification problem in machine learning is one in which a class label is predicted given specific examples of input data. The key to fairly comparing different spot-checked machine learning algorithms is to evaluate each algorithm in the same way, which is achieved with a standardized test harness, which means that we use the same evaluation method and evaluation metric. 

Now, imagine we want to spot check the following six algorithms:
- Logistic regression, which is a data analysis technique that uses several known input values to predict a single unknown data point; 
- Linear discriminant analysis (LDA), which makes predictions for both binary and multi-class classification problems; 
- k-nearest neighbors (KNN), which finds the k most similar training data points for a new instance and takes the mean of the selected training data points to make a prediction; 
- Naive bayes (NB), which calculates the probability and conditional probability of each class, given each input value, and then estimates these probabilities for new data; 
- Classification and regression trees (CART), which construct binary trees from the training data and generate splits to minimize a cost function; and
- Support vector machine (SVM), which seeks a line that best separates classes based on the position of various support vectors.

<img width="450" height="216" alt="Screenshot 2026-02-24 at 10 20 09 AM" src="https://github.com/user-attachments/assets/7dfd42b5-49f0-44d3-a5fa-5aa550111592" />

After spot checking we may get an output like that above. Based on these results we can see that linear discriminant analysis (LDA)and logistic regression (LR) produce the best results. Thus, these two algorithms can be selected for tuning and optimization to enhance their ability to make accurate predictions. 

When working with transcriptomic, proteomic, or metabolomic data for classification tasks, best practice is to spot-check the following algorithms: Logistic Regression (with L2 regularization), Random Forest, Gradient Boosting (XGBoost or LightGBM), Support Vector Machines (with RBF kernel), and LASSO (L1 regularization for simultaneous feature selection). These algorithms are well-suited to high-dimensional omics data because they handle correlated features differently and capture different types of relationships. Logistic Regression and LASSO provide interpretable linear models where you can directly see which proteins contribute to predictions. Random Forest and Gradient Boosting capture non-linear relationships and protein-protein interactions without requiring feature scaling. SVM works well with the high-dimensional, low-sample-size regime common in proteomics. Avoid algorithms like standard KNN or Naive Bayes for omics data, as KNN struggles with high dimensionality (curse of dimensionality) and Naive Bayes' independence assumption is violated by the extensive correlations between proteins in biological pathways. After spot-checking, typically select the top 2-3 performers for hyperparameter tuning.

## Spot-Checking for Regression Models
A regression problem in machine learning is one in which a single unknown data point is predicted from related known data points.  Now, imagine we want to spot check the following six algorithms:
- Linear regression⁴, which is a data analysis technique that assumes a linear relationship between known inputs and predicted outputs; 
- Ridge regression, which is a modified version of linear regression that, is best used when the independent variables in the data are highly correlated; 
- LASSO regression, which is also a modified version of linear regression where the model is penalized for the sum of absolute values of the weights; 
- ElasticNet regression, which is a regularized form of regression that combines the characteristics of both ridge and LASSO regression; 
- k-nearest neighbors (KNN), which finds the k most similar training data points for a new instance and takes the mean of the selected training data points to make a prediction; 
- Classification and regression trees (CART), which construct regression trees from the training data and generate splits to minimize a cost function (in this case, MSE); and
- Support vector machine for regression (SVR), which is an extension of SVM for binary classification that has been modified to predict continuous numerical outputs. 

<img width="446" height="199" alt="Screenshot 2026-02-24 at 10 20 56 AM" src="https://github.com/user-attachments/assets/c60f6ca8-1b7f-4eea-adea-6f1328822ae7" />

Based on these results we can see that LASSO, ridge, and linear regression produce the best results. Thus, these three algorithms can be selected for tuning and optimization to enhance their ability to make accurate predictions. 

When working with transcriptomic, proteomic, or metabolomic data for regression tasks (predicting continuous outcomes like survival time), best practice is to spot-check: Ridge Regression, LASSO, ElasticNet, Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regression. For omics data predicting survival or continuous clinical outcomes, the regularized linear models (Ridge, LASSO, ElasticNet) are often your workhorses—they handle the high correlation between proteins well and provide interpretability. ElasticNet is particularly valuable as it combines the feature selection of LASSO with the stability of Ridge, making it robust when you have groups of highly correlated proteins. Random Forest and Gradient Boosting capture non-linear dose-response relationships and interactions between proteins. For survival analysis specifically, consider Cox Proportional Hazards models or survival-specific Random Forest implementations rather than standard regression, as they properly handle censored data (patients who haven't experienced the event yet). Avoid standard Linear Regression without regularization for omics data, as it will catastrophically overfit when you have more features than samples.

### Example Code
The code below runs the spot-checking procedure for a classification task. We're using the LASSO-selected features from Step 5 as our inputs and evaluating six different classification algorithms using 5-fold cross-validation (Step 6) with ROC-AUC as our primary metric (Step 7). Each algorithm is evaluated identically, using the same train/test folds, to ensure a fair comparison. The cross_validate function from scikit-learn handles the mechanics of this—fitting each model k times and returning per-fold scores. We then average across folds to get a stable performance estimate for each algorithm.
```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cv_folds = 5

# Define scoring metrics
scoring = {'roc_auc': 'roc_auc','precision': make_scorer(precision_score, zero_division=0),'recall': make_scorer(recall_score, zero_division=0),'f1': make_scorer(f1_score, zero_division=0)}

# Define models to spot-check
models = {
   'Logistic Regression (L2)': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
   'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
   'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
   'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
   'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1),
   'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)}

print(f"\nEvaluation setup:")
print(f"  Method: {cv_folds}-fold cross-validation")
print(f"  Primary metric: ROC-AUC")
print(f"  Secondary metrics: Precision, Recall, F1-score")
print(f"  Training samples: {X_train_selected.shape[0]}")
print(f"  Features: {X_train_selected.shape[1]}")
print(f"\nSpot-checking {len(models)} algorithms...")
print("-"*60)

# Store results
results = []

# Evaluate each model
for name, model in models.items():
   print(f"\nTraining {name}...")
   # Cross-validation with multiple metrics
   cv_results = cross_validate( model, X_train_selected, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=False)

   # Store mean scores
   results.append({ 'Model': name, 'ROC-AUC': cv_results['test_roc_auc'].mean(),'ROC-AUC Std': cv_results['test_roc_auc'].std(), 'Precision': cv_results['test_precision'].mean(), 'Recall': cv_results['test_recall'].mean(), 'F1-Score': cv_results['test_f1'].mean()})

   print(f"  ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} (+/ {cv_results['test_roc_auc'].std():.4f})")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC-AUC', ascending=False)

# Display results
print("\n" + "="*60)
print("SPOT-CHECKING RESULTS")
print("="*60)
print("\nRanked by ROC-AUC:")
print(results_df.to_string(index=False))

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: ROC-AUC comparison with error bars
ax1 = axes[0]
models_sorted = results_df['Model'].tolist()
aucs = results_df['ROC-AUC'].tolist()
stds = results_df['ROC-AUC Std'].tolist()

y_pos = np.arange(len(models_sorted))
bars = ax1.barh(y_pos, aucs, xerr=stds, alpha=0.7, capsize=5)

# Color bars by performance
colors = ['green' if auc > 0.75 else 'orange' if auc > 0.65 else 'red' for auc in aucs]
for bar, color in zip(bars, colors):
   bar.set_color(color)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(models_sorted)
ax1.invert_yaxis()
ax1.set_xlabel('ROC-AUC Score', fontsize=12)
ax1.set_title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Random Chance')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: All metrics comparison for top 3 models
ax2 = axes[1]
top_3 = results_df.head(3)
metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

for i, (idx, row) in enumerate(top_3.iterrows()):
   values = [row['ROC-AUC'], row['Precision'], row['Recall'], row['F1-Score']]
   ax2.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)

ax2.set_xlabel('Metrics', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Top 3 Models: All Metrics', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.show()

# Identify top performers
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
top_3_models = results_df.head(3)
print(f"\nTop 3 models for hyperparameter tuning:")
for i, (idx, row) in enumerate(top_3_models.iterrows(), 1):
   print(f"  {i}. {row['Model']} (ROC-AUC: {row['ROC-AUC']:.4f})")
```
```
Evaluation setup:
  Method: 5-fold cross-validation
  Primary metric: ROC-AUC
  Secondary metrics: Precision, Recall, F1-score
  Training samples: 137
  Features: 25

Spot-checking 6 algorithms...
------------------------------------------------------------

Training Logistic Regression (L2)...
  ROC-AUC: 0.8462 (+/- 0.0445)

Training Logistic Regression (L1)...
  ROC-AUC: 0.8221 (+/- 0.0583)

Training Random Forest...
  ROC-AUC: 0.8104 (+/- 0.0718)

Training Gradient Boosting...
  ROC-AUC: 0.7966 (+/- 0.0844)

Training XGBoost...
  ROC-AUC: 0.7822 (+/- 0.0440)

Training SVM (RBF)...
  ROC-AUC: 0.8086 (+/- 0.0805)

============================================================
SPOT-CHECKING RESULTS
============================================================

Ranked by ROC-AUC:
                   Model  ROC-AUC  ROC-AUC Std  Precision   Recall  F1-Score
Logistic Regression (L2) 0.846154     0.044544   0.736970 0.751648  0.739241
Logistic Regression (L1) 0.822135     0.058315   0.717535 0.767033  0.736150
           Random Forest 0.810400     0.071767   0.738187 0.753846  0.744280
               SVM (RBF) 0.808634     0.080518   0.712703 0.764835  0.736228
       Gradient Boosting 0.796625     0.084405   0.795580 0.695604  0.734277
                 XGBoost 0.782182     0.043984   0.740073 0.738462  0.738497

```
<img width="542" height="202" alt="Screenshot 2026-02-24 at 10 24 27 AM" src="https://github.com/user-attachments/assets/afd525a1-4280-4e68-8969-6924ca26d1b5" />

The output shows that Logistic Regression (L2) achieves the highest cross-validated ROC-AUC of 0.846, followed closely by Logistic Regression (L1) at 0.822. Notably, the L2 model also has the smallest standard deviation across folds (±0.044), indicating it is the most stable—an important consideration in small-sample settings where variance in performance estimates can be high. The tree-based ensemble methods (Random Forest, Gradient Boosting, XGBoost) perform reasonably well but trail the linear models, which is typical when the underlying signal is relatively linear or when the dataset is small enough that regularized linear models outperform more complex non-linear alternatives. Based on these results, we carry Logistic Regression (L2) and Logistic Regression (L1) forward for hyperparameter tuning in Step 9.

# Step 9–Tune Hyperparameters 
You can think of machine learning algorithms as systems with various knobs and dials, which you can adjust in any number of ways to change how output data (predictions) are generated from input data. The knobs and dials in these systems can be subdivided into parameters and hyperparameters. Parameters are model settings that are learned, adjusted, and optimized automatically. Conversely, hyperparameters need to be manually set manually by whoever is programming the machine learning algorithm.

Generally, tuning hyperparameters has known effects on machine learning algorithms. However, it’s not always clear how to best set a hyperparameter to optimize model performance for a specific dataset. As a result, search strategies are often used to find optimal hyperparameter configurations. In this tutorial, I’m going to cover the following hyperparameter tuning methods:
- **Grid Search** is a cross-validation technique for hyperparameter tuning that finds an optimal parameter value among a given set of parameters specified in a grid; and
- **Random Search** is a tuning technique that randomly samples a specified number of uniformly distributed algorithm parameters.

Typically, when working with transcriptomic, proteomic, or metabolomic data the best practice is to use ```GridSearchCV``` for smaller hyperparameter spaces (fewer than ~100 combinations) and ```RandomizedSearchCV``` for larger spaces.

### Example Code
The code below tunes the two top-performing models from Step 8—Logistic Regression (L2) and Logistic Regression (L1)—using GridSearchCV with 5-fold cross-validation and ROC-AUC as the optimization target. For each model, we define a grid of candidate hyperparameter values (primarily varying C, the regularization strength, and solver). GridSearchCV exhaustively evaluates every combination on the training folds, selecting the combination that maximizes average validation AUC. After finding the best hyperparameters for each model, we then evaluate each tuned model on the held-out validation set to make our final model selection.
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Use your actual training data (replace with balanced if you used SMOTE)
X_train_tune = X_train_selected
y_train_tune = y_train

# Define parameter grids for top 3 models
param_grids = {
   'Logistic Regression (L2)': {
       'model': LogisticRegression(penalty='l2', max_iter=1000, random_state=42),
       'params': { 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}},
   'Logistic Regression (L1)': {
       'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
       'params': { 'C': [0.001, 0.01, 0.1, 1, 10, 100]}}

print(f"\nTuning {len(param_grids)} models using GridSearchCV...")
print(f"  Training samples: {X_train_tune.shape[0]}")
print(f"  Features: {X_train_tune.shape[1]}")
print(f"  Evaluation: 5-fold cross-validation")
print(f"  Metric: ROC-AUC")
print("-"*60)

# Store tuned models
tuned_models = {}
tuning_results = []

# Tune each model
for name, config in param_grids.items():
   print(f"\nTuning {name}...")
   # Grid search
   grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)

   # Fit
   grid_search.fit(X_train_tune, y_train_tune)
   # Store results
   tuned_models[name] = grid_search.best_estimator_

   tuning_results.append({ 'Model': name,  'Best CV ROC-AUC': grid_search.best_score_, 'Best Params': str(grid_search.best_params_)})
   print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
   print(f"  Best parameters: {grid_search.best_params_}")

# Create results DataFrame
tuning_df = pd.DataFrame(tuning_results)
tuning_df = tuning_df.sort_values('Best CV ROC-AUC', ascending=False)

print("\n" + "="*60)
print("TUNING RESULTS")
print("="*60)
print(tuning_df.to_string(index=False))

# Evaluate tuned models on validation set
print("\n" + "="*60)
print("VALIDATION SET EVALUATION")
print("="*60)

val_results = []
for name, model in tuned_models.items():
   # Predict on validation set
   y_val_pred = model.predict(X_val_selected)
   y_val_proba = model.predict_proba(X_val_selected)[:, 1]
   # Calculate metrics
   val_auc = roc_auc_score(y_val, y_val_proba)
   val_results.append({'Model': name,'Validation ROC-AUC': val_auc})
   print(f"\n{name}:")
   print(f"  Validation ROC-AUC: {val_auc:.4f}")

# Final comparison
val_df = pd.DataFrame(val_results)
val_df = val_df.sort_values('Validation ROC-AUC', ascending=False)

# Merge with tuning results
final_comparison = tuning_df.merge(val_df, on='Model')
final_comparison['Performance Drop'] = final_comparison['Best CV ROC-AUC'] - final_comparison['Validation ROC-AUC']

print("\n" + "="*60)
print("FINAL COMPARISON: CV vs Validation")
print("="*60)
print(final_comparison[['Model', 'Best CV ROC-AUC', 'Validation ROC-AUC', 'Performance Drop']].to_string(index=False))

# Select best model
best_model_name = final_comparison.iloc[0]['Model']
best_model = tuned_models[best_model_name]
best_val_auc = final_comparison.iloc[0]['Validation ROC-AUC']

print("\n" + "="*60)
print("FINAL MODEL SELECTION")
print("="*60)
print(f"\nSelected model: {best_model_name}")
print(f"Validation ROC-AUC: {best_val_auc:.4f}")

# Detailed evaluation of best model
print("\n" + "-"*60)
print(f"Detailed Validation Results for {best_model_name}")
print("-"*60)

y_val_pred_best = best_model.predict(X_val_selected)
y_val_proba_best = best_model.predict_proba(X_val_selected)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_best, target_names=['RCB 2/3', 'RCB 0/1']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred_best)
print(cm)
print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
```
```
Tuning 2 models using GridSearchCV...
  Training samples: 137
  Features: 25
  Evaluation: 5-fold cross-validation
  Metric: ROC-AUC
------------------------------------------------------------
Tuning Logistic Regression (L2)...
  Best CV ROC-AUC: 0.8761
  Best parameters: {'C': 0.1, 'solver': 'lbfgs'}

Tuning Logistic Regression (L1)...
  Best CV ROC-AUC: 0.8221
  Best parameters: {'C': 1}
============================================================
TUNING RESULTS
============================================================
                   Model  Best CV ROC-AUC                   Best Params
Logistic Regression (L2)         0.876138 {'C': 0.1, 'solver': 'lbfgs'}
Logistic Regression (L1)         0.822135                      {'C': 1}
============================================================
VALIDATION SET EVALUATION
============================================================
Logistic Regression (L2):
  Validation ROC-AUC: 0.7157

Logistic Regression (L1):
  Validation ROC-AUC: 0.7157
============================================================
FINAL COMPARISON: CV vs Validation
============================================================
                   Model  Best CV ROC-AUC  Validation ROC-AUC  Performance Drop
Logistic Regression (L2)         0.876138            0.715686          0.160452
Logistic Regression (L1)         0.822135            0.715686          0.106449
============================================================
FINAL MODEL SELECTION
============================================================
Selected model: Logistic Regression (L2)
Validation ROC-AUC: 0.7157
------------------------------------------------------------
Detailed Validation Results for Logistic Regression (L2)
------------------------------------------------------------
Classification Report:
              precision    recall  f1-score   support

     RCB 2/3       0.67      0.67      0.67        18
     RCB 0/1       0.65      0.65      0.65        17

    accuracy                           0.66        35
   macro avg       0.66      0.66      0.66        35
weighted avg       0.66      0.66      0.66        35

Confusion Matrix:
[[12  6]
 [ 6 11]]
  [[TN= 12  FP=  6]
   [FN=  6  TP= 11]]
```

Tuning improves the best cross-validated AUC for Logistic Regression (L2) from 0.846 (default hyperparameters, Step 8) to 0.876, with the grid search selecting C=0.1 and the lbfgs solver. This lower C value imposes stronger regularization than the default (C=1), which makes sense given our relatively small sample size—stronger regularization reduces overfitting to the training folds. When we evaluate these tuned models on the held-out validation set, both L2 and L1 models achieve a ROC-AUC of 0.716. The 16-point gap between cross-validated performance (0.876) and validation performance (0.716) is worth noting: it reflects the inherent difficulty of generalizing from 137 training samples to 35 held-out patients, and is a reminder that cross-validation scores on small datasets can be optimistic. Because both models achieve identical validation AUC, we select Logistic Regression (L2) as our final model based on its lower cross-validation variance from Step 8 (±0.044 vs ±0.058 for L1), making it the more stable choice. The confusion matrix shows 12 true negatives, 11 true positives, and 6 errors in each direction—a roughly symmetric error pattern indicating the model is not systematically biased toward one class.

# Step 10–Evaluation On Validation Set
After spot-checking algorithms and tuning hyperparameters using cross-validation on your training data, you should have identified your top 2-3 models. At this stage, it's time to evaluate these models on the held-out validation set to make your final model selection. Remember, the validation set has not been used at all up to this point—it's been sitting aside, completely untouched, waiting for this moment.

The purpose of the validation set is to give you an honest estimate of how your models will perform on truly unseen data and to help you choose between different modeling approaches. For example, you might want to compare a Random Forest model trained on LASSO-selected features versus a Gradient Boosting model trained on all differentially expressed proteins. Or you might want to compare models using different feature selection thresholds (FDR < 0.01 vs. FDR < 0.05).

To evaluate on the validation set, take each of your top-performing models and use them to make predictions on the validation data, using the same preprocessing pipeline you applied during training:

```python
# Combine train and validation
X_train_val = pd.concat([X_train_selected, X_val_selected], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

print(f"  Training samples: {X_train_selected.shape[0]}")
print(f"  Validation samples: {X_val_selected.shape[0]}")
print(f"  Combined samples: {X_train_val.shape[0]}")

# Get best hyperparameters from Step 9
print(f"\nRetraining {best_model_name} with optimal hyperparameters...")
print(f"  Best parameters: {final_comparison.iloc[0]['Best Params']}")

# Recreate model with best parameters
if best_model_name == 'SVM (RBF)':
   # Extract best parameters (assuming SVM won)
   final_model = SVC(C=100, gamma=0.001, kernel='rbf', probability=True, random_state=42)
elif best_model_name == 'Logistic Regression (L2)':
   final_model = LogisticRegression(C=0.1, solver='lbfgs', penalty='l2', max_iter=1000, random_state=42)
elif best_model_name == 'Logistic Regression (L1)':
   final_model = LogisticRegression(C=1, solver='liblinear', penalty='l1', max_iter=1000, random_state=42)

# Fit on combined data
final_model.fit(X_train_val, y_train_val)

print(f"\n✓ Model retrained on {X_train_val.shape[0]} samples")
```
```
Training samples: 137
  Validation samples: 35
  Combined samples: 172

Retraining Logistic Regression (L2) with optimal hyperparameters...
  Best parameters: {'C': 0.1, 'solver': 'lbfgs'}

✓ Model retrained on 172 samples
```
A critical point: you should not iterate excessively on the validation set. If you find that all your models perform poorly on validation, you might make one or two adjustments to your approach (perhaps trying different feature selection methods or algorithms), but be very cautious about repeatedly tweaking your models until validation performance improves. This can lead to overfitting to the validation set, defeating its purpose as an unbiased estimate.

Based on validation set performance, select your single best model. This is the model you'll carry forward to final testing and potential clinical application. In practice, you'll often see a modest performance drop from cross-validation on training (say, 0.85 AUC) to validation set evaluation (0.82 AUC). This is normal and expected—it reflects the challenge of generalizing to new data. A drop of 5-10% is acceptable, but if you see drops of 20% or more, it suggests your model has severely overfit to the training data.

### Optional 
After selecting your final model based on validation set performance, you have the option to retrain that model on the combined training and validation sets before final test set evaluation. This step is optional but commonly used in machine learning competitions and production models. The rationale is straightforward: up until this point, you've only trained your model on 64% of your available data (the training set), while the validation set (16%) has been sitting unused. Now that you've made all your modeling decisions, you can safely combine these two sets to train your final model, giving it access to 80% of your data. This typically results in a modest performance improvement, as more data generally leads to better models.

Here's how it works in practice:
```python
# Combine training and validation data
X_train_val = np.vstack([X_train_scaled, X_val_scaled])
y_train_val = np.hstack([y_train, y_val])

# Retrain your selected model with its optimized hyperparameters
final_model = RandomForestClassifier(**best_hyperparameters)
final_model.fit(X_train_val, y_train_val)

# Now evaluate on test set (next step)
```

# Step 11–Final Evaluation on Test Set
This is the moment of truth. After all the development work—spot-checking, hyperparameter tuning, validation set evaluation, and possibly retraining on combined data—you're finally ready to evaluate your model on the test set. This step should only be performed once, at the very end of your pipeline, and the results you obtain here are what you report as your model's performance.

The test set has been completely locked away since Step 3, never touched, never peeked at, serving as your unbiased estimate of how the model will perform on truly unseen patients in the real world. This is your honest assessment, free from any optimization or tuning that could inflate performance estimates.

### Example Code
The code below generates predictions on the held-out test set using our final retrained model (Logistic Regression L2, C=0.1, trained on combined train+val data). We compute the full suite of performance metrics—ROC-AUC, accuracy, precision, recall, F1—along with a confusion matrix and a set of clinical metrics (sensitivity, specificity, PPV, NPV) that are more directly interpretable in a treatment-response prediction context. We also plot the ROC curve, which visually summarizes how well the model discriminates between pCR and non-pCR patients across all possible classification thresholds.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc)

use_retrained = True  # Set to False to use original best_model
if use_retrained:
   print("Using retrained model (train + val data)")
   test_model = final_model
else:
   print("Using original best model (train data only)")
   test_model = best_model

print(f"Model: {best_model_name}")
print(f"Test samples: {X_test_selected.shape[0]}")

# Make predictions on test set
print("\nMaking predictions on test set...")
y_test_pred = test_model.predict(X_test_selected)
y_test_proba = test_model.predict_proba(X_test_selected)[:, 1]

# Calculate metrics
test_auc = roc_auc_score(y_test, y_test_proba)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print("="*60)
print(f"\nROC-AUC:    {test_auc:.4f}")
print(f"Accuracy:   {test_accuracy:.4f}")
print(f"Precision:  {test_precision:.4f}")
print(f"Recall:     {test_recall:.4f}")
print(f"F1-Score:   {test_f1:.4f}")

# Compare to validation
print("\n" + "-"*60)
print("PERFORMANCE COMPARISON")
print("-"*60)
print(f"\nValidation ROC-AUC: {best_val_auc:.4f}")
print(f"Test ROC-AUC:       {test_auc:.4f}")
print(f"Difference:         {test_auc - best_val_auc:+.4f}")

if abs(test_auc - best_val_auc) < 0.05:
   print("✓ Performance is consistent (< 5% difference)")
elif abs(test_auc - best_val_auc) < 0.10:
   print("  Moderate difference (5-10%)")
else:
   print(" Large difference (> 10%) - possible overfitting or dataset shift")

# Detailed classification report
print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
print(classification_report(y_test, y_test_pred, target_names=['Non-pCR', 'pCR']))

# Confusion matrix
print("\n" + "-"*60)
print("CONFUSION MATRIX")
print("-"*60)

cm = confusion_matrix(y_test, y_test_pred)
print(f"\n{cm}")
print(f"\n  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print("\n" + "-"*60)
print("CLINICAL METRICS")
print("-"*60)
print(f"\nSensitivity (Recall):  {sensitivity:.2%}  ({tp}/{tp+fn} pCR patients correctly identified)")
print(f"Specificity:           {specificity:.2%}  ({tn}/{tn+fp} non-pCR patients correctly identified)")
print(f"PPV (Precision):       {ppv:.2%}  ({tp}/{tp+fp} predicted pCR are correct)")
print(f"NPV:                   {npv:.2%}  ({tn}/{tn+fn} predicted non-pCR are correct)")

# Plot ROC curve (optional)
print("\n" + "-"*60)
print("ROC CURVE")
print("-"*60)

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - Test Set\n{best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_roc_curve.png', dpi=300, bbox_inches='tight')
print("\n→ ROC curve saved as 'test_roc_curve.png'")
plt.show()

print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)
print(f"\nModel: {best_model_name}")
print(f"Features: {X_test_selected.shape[1]} proteins")
print(f"Training approach: {'Combined train+val' if use_retrained else 'Train only'}")
print(f"\nPerformance journey:")
print(f"  Cross-validation (training): {final_comparison.iloc[0]['Best CV ROC-AUC']:.4f}")
print(f"  Validation set:              {best_val_auc:.4f}")
print(f"  Test set (FINAL):            {test_auc:.4f}")
```
```
Using retrained model (train + val data)
Model: Logistic Regression (L2)
Test samples: 43

Making predictions on test set...
============================================================
TEST SET PERFORMANCE
============================================================
ROC-AUC:    0.8139
Accuracy:   0.6744
Precision:  0.6842
Recall:     0.6190
F1-Score:   0.6500
------------------------------------------------------------
PERFORMANCE COMPARISON
------------------------------------------------------------
Validation ROC-AUC: 0.7157
Test ROC-AUC:       0.8139
Difference:         +0.0982
Moderate difference (5-10%)
------------------------------------------------------------
CLASSIFICATION REPORT
------------------------------------------------------------
              precision    recall  f1-score   support

     Non-pCR       0.67      0.73      0.70        22
         pCR       0.68      0.62      0.65        21

    accuracy                           0.67        43
   macro avg       0.68      0.67      0.67        43
weighted avg       0.68      0.67      0.67        43
------------------------------------------------------------
CONFUSION MATRIX
------------------------------------------------------------
[[16  6]
 [ 8 13]]

  [[TN= 16  FP=  6]
   [FN=  8  TP= 13]]
------------------------------------------------------------
CLINICAL METRICS
------------------------------------------------------------
Sensitivity (Recall):  61.90%  (13/21 pCR patients correctly identified)
Specificity:           72.73%  (16/22 non-pCR patients correctly identified)
PPV (Precision):       68.42%  (13/19 predicted pCR are correct)
NPV:                   66.67%  (16/24 predicted non-pCR are correct)
------------------------------------------------------------
ROC CURVE
------------------------------------------------------------
```
<img width="545" height="401" alt="Screenshot 2026-02-24 at 10 32 16 AM" src="https://github.com/user-attachments/assets/40cc977a-258e-4b5f-acb8-24cec7818f3e" />
```
============================================================
FINAL MODEL SUMMARY
============================================================
Model: Logistic Regression (L2)
Features: 25 proteins
Training approach: Combined train+val

Performance journey:
  Cross-validation (training): 0.8761
  Validation set:              0.7157
  Test set (FINAL):            0.8139
```

The final model achieves a test set ROC-AUC of 0.814, which is a meaningful improvement over the validation set AUC of 0.716. This upward shift—the opposite of what we typically expect—is a consequence of retraining on the combined train+val dataset (172 samples vs. 137), which gave the model access to more data before final evaluation. The test set AUC of 0.814 suggests the model has learned a genuine biological signal: it is substantially better than random chance and is consistent with published proteomic biomarker panels in similar clinical settings. In practical terms, the model correctly identifies about 62% of pCR patients (sensitivity) and 73% of non-pCR patients (specificity). The NPV of 67% means that when the model predicts non-response, it is correct roughly two-thirds of the time—a moderate but clinically limited performance for guiding treatment de-escalation. Together, these metrics suggest the biomarker panel captures real signals but would need improvement—or combination with clinical variables—before it could be used to inform individual treatment decisions.

### Outcomes and Expectations 
Having completed the full pipeline, it's worth stepping back to ask: what does the performance we achieved actually mean, and is it good enough?

**What constitutes good performance?** This is highly context-dependent. For cancer treatment response prediction, an AUC of 0.75-0.80 might be clinically useful even if not perfect, as it could help stratify patients for different treatment strategies. For rare disease diagnosis where false negatives are catastrophic, you might need AUC > 0.90 to be clinically viable. Compare your results to published biomarker panels for similar problems to gauge whether your performance is competitive.

**What if test performance is disappointing?** It's tempting to go back and modify your approach when test results are lower than expected, but resist this urge. If you iterate based on test set performance, it's no longer a true held-out test set—you've essentially turned it into another validation set. If test performance is genuinely poor (say, barely better than random chance), you might have a fundamental problem with your data or approach, but at this point, the ethical approach is to report the honest results and potentially collect new data for a fresh attempt.

**Realistic expectations:** You'll almost always see some performance drop from training to validation to test. A typical progression might be: training CV (0.87 AUC) → validation (0.83 AUC) → test (0.80 AUC). This reflects the increasing "distance" from the data the model was optimized on. Drops of 5-15% are normal; drops larger than 20% suggest serious overfitting problems that should be addressed in future iterations.

# Step 12–Interpret The Model
Once you have your final model and its test set performance, the work isn't over—in fact, one of the most important steps for biomarker panels is interpreting what the model has learned. A black-box predictor, no matter how accurate, is rarely sufficient for clinical translation or biological insight. You need to understand which proteins drive predictions, whether these make biological sense, and where the model succeeds or fails.

**Feature Importance Analysis** is your first step. For tree-based models (Random Forest, Gradient Boosting), you can extract feature importance scores that indicate how much each protein contributes to predictions. For linear models (Logistic Regression, LASSO), the coefficients themselves indicate importance and direction—positive coefficients increase the probability of one class, negative coefficients decrease it. These coefficient directions should align with your biological understanding. For example, if you're predicting treatment non-response and your model gives high importance to immune checkpoint proteins like PD-L1 with positive coefficients, this aligns with known biology that high PD-L1 expression can indicate immune evasion. 

SHAP (SHapley Additive exPlanations) values provide a more sophisticated interpretation, showing how each feature contributes to individual prediction. Additionally ,SHAP plots can reveal whether certain proteins are consistently important or only matter for specific patients, helping identify patient subgroups that might benefit from different biomarker strategies.

**Biological Validation** is crucial. Ask yourself: Do the identified proteins make biological sense? If your model selects proteins involved in cell cycle regulation and DNA repair for predicting chemotherapy response, that's biologically coherent. If it selects housekeeping proteins or proteins with no known connection to your outcome, that's a red flag suggesting the model might be learning technical artifacts rather than biology.

Perform **pathway enrichment analysis** on your selected proteins using tools like DAVID, Enrichr, or Gene Ontology analysis. If your 20 predictive proteins cluster into 2-3 known biological pathways (e.g., immune response, proliferation, apoptosis), this strengthens confidence that your model has captured real biology rather than noise.

**Clinical Utility Assessment** goes beyond pure performance metrics. For a biomarker panel to be clinically useful, consider: What is the positive predictive value (PPV) and negative predictive value (NPV) at different probability thresholds? If you're building a test to spare patients from unnecessary chemotherapy, you need high NPV—when your model says a patient will respond, you need to be very confident. This might mean setting a conservative threshold where you only predict "responder" when the model is >80% confident, accepting more false negatives to minimize false positives.

Finally, **document failure modes**. Which patients does your model consistently misclassify? Are they at disease stage boundaries? Do they have unusual protein expression patterns? Understanding failure modes helps future researchers improve the model and helps clinicians know when to be skeptical of the model's predictions.

### Example Code:
The code below extracts and visualizes the model coefficients from our final Logistic Regression (L2) model. Because logistic regression is a linear model, the coefficients have a direct interpretation: each coefficient represents the change in log-odds of predicting pCR per unit change in the standardized protein expression level, holding all other features constant. Proteins with large positive coefficients push predictions toward pCR; proteins with large negative coefficients push predictions toward non-pCR. We sort features by their absolute coefficient magnitude to identify the most influential proteins, and we save the full ranked list to a CSV for downstream biological analysis.

```python
# Extract coefficients from the model
if best_model_name in ['Logistic Regression (L2)', 'Logistic Regression (L1)']:
   coefficients = final_model.coef_[0]
   # Create feature importance DataFrame
   feature_importance = pd.DataFrame({'Protein': X_train_selected.columns, 'Coefficient': coefficients, 'Abs_Coefficient': np.abs(coefficients)})
   feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

   print("\nTop 20 Most Influential Proteins:")
   print("-"*60)
   print(f"{'Rank':<6} {'Protein':<35} {'Coefficient':>12} {'Direction':>12}")
   print("-"*60)

   for i, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
       direction = "pCR +" if row['Coefficient'] > 0 else "Non-pCR +"
       print(f"{i:<6} {row['Protein']:<35} {row['Coefficient']:>12.4f} {direction:>12}")
``
```
Top 20 Most Influential Proteins:
------------------------------------------------------------
Rank   Protein                              Coefficient    Direction
------------------------------------------------------------
1      Tuberin Y1571                            -0.5891    Non-pCR +
2      cABL Y245                                 0.3838        pCR +
3      AMPKb1 S108                               0.3550        pCR +
4      p70S6K S371                               0.3182        pCR +
5      eIF4E S209                               -0.2883    Non-pCR +
6      Ephrin A3 Y799/A4 Y799/A5 Y833            0.2864        pCR +
7      cKIT Y703                                -0.2824    Non-pCR +
8      RET Y905                                 -0.2750    Non-pCR +
9      ERBB3 total                              -0.2690    Non-pCR +
10     Estrogen Rec alpha                       -0.2675    Non-pCR +
11     STAT1 Y701                                0.2573        pCR +
12     ALK Y1586                                 0.2489        pCR +
13     AKT T308                                 -0.2335    Non-pCR +
14     STAT3 Y705                               -0.2298    Non-pCR +
15     mTOR S2448                                0.2275        pCR +
16     Cyclin D1                                -0.2224    Non-pCR +
17     ERBB2                                    -0.2217    Non-pCR +
18     HLA-DR                                    0.2118        pCR +
19     Androgen Receptor                         0.2103        pCR +
20     NF2R1/COUP1                               0.2057        pCR +
```

Several features in this list have clear biological interpretability in the context of treatment response in breast cancer. For example, Tuberin phosphorylation (Y1571) is the strongest negative predictor of pCR—higher TSC2 phosphorylation is associated with non-response, which aligns with TSC2's role as an mTOR suppressor; its inactivation (via phosphorylation) would activate mTOR signaling, a known resistance pathway. Conversely, AMPKβ1 phosphorylation (S108) is positively associated with pCR: AMPK activation suppresses anabolic metabolism and can sensitize cells to chemotherapy-induced stress. Additionally, the positive association of HLA-DR (an MHC class II antigen) with pCR aligns with known evidence that immune-inflamed tumors, particularly those with high antigen presentation machinery, respond better to chemotherapy and immunotherapy . 

# Step 13–External Validation 
No matter how well your model performs on your held-out test set, external validation on completely independent data is the gold standard for demonstrating that your biomarker panel will generalize to real-world clinical use. External validation involves applying your model to data from different institutions, different patient populations, different time periods, or even different measurement platforms—anything that represents a true out-of-distribution test.

**Why is external validation so critical?** Your training, validation, and test sets, despite being split randomly, all come from the same source and likely share hidden similarities—they might all be measured on the same instrument, processed by the same lab, or drawn from patients treated at the same hospital with similar treatment protocols. These subtle systematic similarities can allow models to achieve high performance on your test set while failing when applied elsewhere. External validation is where many published biomarkers fail, revealing that what looked like robust performance was actually learning institution-specific technical artifacts.

Types of external validation:
- **Temporal validation:** If you trained your model on patients treated from 2015-2018, validate on patients treated from 2019-2022 at the same institution. This tests whether the model's performance holds up as treatment protocols, measurement techniques, or patient populations shift over time.
- **Geographic validation:** Apply your model to data from a different hospital or country. This is particularly important for biomarkers intended for broad clinical use, as it tests robustness to different patient demographics, healthcare systems, and laboratory practices.
- **Platform validation:** If you built your model on RPPA data, test it on mass spectrometry-based proteomics from the same patients. This is challenging because absolute protein levels often don't translate directly across platforms, but if your biomarker is truly capturing biology rather than technical artifacts, the relative ranking of proteins should remain consistent.

# What If We’re Using Multi-Omic Data?
Up to this point, we've primarily discussed building biomarker panels from a single data type—typically proteomics or transcriptomics. However, one of the most powerful approaches in modern computational biology is integrating multiple omics layers (proteomics, transcriptomics, metabolomics, etc.) to build multi-omics signatures. The biological rationale is compelling: different molecular layers capture complementary aspects of disease biology. Gene expression tells you what the cell is trying to make, protein levels tell you what's actually there, and metabolite profiles tell you what the cell is doing. By integrating these views, you can often achieve better predictive performance and deeper biological insight than any single omics layer alone.

However, multi-omics integration presents unique challenges. You can't simply concatenate all features together—you'd end up with 20,000 genes + 5,000 proteins + 500 metabolites = 25,500 features with likely fewer than 200 samples, an extreme case of the curse of dimensionality. Moreover, different omics types have different scales, noise characteristics, and biological meanings. Specialized methods have been developed to handle multi-omics integration, and in this section, we'll focus on two of the most popular approaches for biomarker development: MOFA (Multi-Omics Factor Analysis) and DIABLO (Data Integration Analysis for Biomarker discovery using Latent cOmponents).

> Note: This section intentionally stays conceptual. Multi-omics integration introduces enough additional complexity—separate preprocessing pipelines per data type, alignment of samples across modalities, method-specific hyperparameters, and interpretation of latent components—that it warrants its own dedicated treatment. If there's enough interest from readers, I plan to write a follow-up guide that works through a complete multi-omics biomarker development pipeline end-to-end, with code, worked examples, and output interpretation for both MOFA and DIABLO. If that's something you'd find useful, feel free to reach out at evanpeikon@gmail.com.

### MOFA
MOFA (Multi-Omics Factor Analysis) is an unsupervised dimensionality reduction method that learns a set of latent factors shared across multiple omics data types. Think of it as PCA generalized to multiple data modalities simultaneously: rather than reducing a single matrix to principal components, MOFA reduces K omics matrices to a shared factor matrix, where each factor may explain variance in one or more of the input data types. This makes MOFA especially useful for exploratory analysis—identifying coordinated sources of biological variation (e.g., a factor driven by both transcriptomic and proteomic changes) without requiring a predefined outcome.

In a biomarker development context, MOFA fits into the pipeline as a dimensionality reduction step (Step 5). After splitting and scaling each omics dataset separately (Steps 3-4), you fit MOFA on the training data only to learn a set of factors. These factors then replace the raw features as inputs to your downstream classification or regression models. Because MOFA is unsupervised, it does not use the outcome variable during factor learning, which means it may not always prioritize factors that are predictive of your outcome—a key limitation compared to supervised integration methods like DIABLO.

### DIABLO
DIABLO (Data Integration Analysis for Biomarker discovery using Latent components) is a supervised multi-omics integration method. Unlike MOFA, DIABLO is outcome-aware: it learns latent components that simultaneously maximize the correlation between omics data types while also discriminating between outcome classes. This makes DIABLO well-suited for biomarker discovery, where the goal is to identify multi-omics features that are jointly predictive of a clinical outcome.

DIABLO operates similarly to Sparse Canonical Correlation Analysis (sCCA) but extended to the supervised, multi-class setting. It selects features from each omics layer simultaneously, yielding a multi-omics signature—for example, a set of 10 proteins, 15 genes, and 5 metabolites—that together predict treatment response. This built-in feature selection means you can skip the dedicated feature selection step (Step 5) when using DIABLO.

# What Makes Someone Skilled?
As someone coming from a research (computational cancer biology)  background, starting to build these models almost felt too simple. An end to end research project could easily wrack up a few thousand lines of code, yet when I started building these ML models I found an end to end project may only be ~250 lines of code, sometimes less. This was incredibly confusing to me. Afterall, we’ve seen the news stories of ML researchers pulling >1M salaries. Surely, there had to be more, right? The short answer is yes and no. From an interview with Andrei Karpathy I learned that an entire LLM can be built in <500 lines of code. What makes someone worth a million dollar salary in a frontier AI lab isn’t making a model that performs well–it’s doing this in a highly efficient way so models can be built, deployed, and scaled to millions of users. According to AK, almost all of the fractional improvements come down to efficiency. But, as a researcher, and developer, building models like the ones we're discussing in this piece, it really doesn’t take that much code. What I've learned from talking with a number of industry computational biologists, and through my own work, is that what makes a great ML practitioner in computational biology is less about the code they write, and more about how they interrogate its output and make small targeted modifications. So, to answer the question above, what makes someone good vs bad at this?

#### Integrates Domain Knowledges: 
- Bad: "LASSO selected Protein X as a feature, let's use it"
- Good: “LASSO selected Protein X as a feature. But, protein X is a housekeeping protein that tends to vary minimally across subjects. Let’s investigate before we include it.” 

This is the skill that most distinguishes a computational biologist from a pure ML practitioner, and it shows up at every step of the pipeline. It's known that a "missing" ERBB2 value in an HR+/HER2- patient probably reflects below-detection expression rather than a failed assay. It's recognizing that LASSO selected 20 proteins and that 15 of them cluster into the mTOR/PI3K pathway, which either validates the model or suggests it's learned a single redundant signal dressed up as 15 independent features. It's flagging that your top predictive feature is a protein involved in sample preparation rather than tumor biology—a pattern that's hard to catch without knowing the literature. The code can't do any of this. You have to.

#### Avoids Overfitting:
- Bad: Tries a bunch of different feature sets, then picks the one with the best validation performance (data leakage)
- Good: Pre-specifies the approach, then evaluates the model after codifying decisions.

The obvious form of overfitting (evaluating on training data) is easy to avoid once you know how to avoid it. The subtle forms are harder. Implicit overfitting happens when you run 15 variations of your pipeline and report only the one that worked—each additional experiment increases the probability that a good result is due to chance rather than signal. Metric overfitting happens when you try several evaluation metrics and report the one that makes your model look best. Threshold overfitting happens when you tune your classification cutoff on the test set. None of these look like cheating from the outside, but they all produce the same outcome: results that won't replicate. Pre-registration of your analytic plan—even informally, in a lab notebook—is underused in computational biology and more valuable than most people realize.

#### Understands When the Model Fails: 
- Bad: "85% accuracy, ship it!"
- Good: "85% overall, but only 40% in HR+/HER2- subtype - need stratified model based on HR/HER2 subtypes"

Overall performance metrics are averages, and averages hide subgroup failures. In clinical settings, the subgroup your model fails on is rarely random—it tends to be the subgroup with the least representation in your training data, or the one where the underlying biology is most distinct. Stratifying your confusion matrix by known clinical covariates (receptor subtype, disease stage, treatment arm) should be standard practice, not optional. A model with good aggregate performance and a critical subgroup failure is often more dangerous than a model with uniformly mediocre performance, because the former creates false confidence.

#### Handles Class Imbalances Properly:
- Bad: 90% responders, 10% non-responders → model predicts "responder" for everyone, claims 90% accuracy
- Good: Uses SMOTE, class weights, or stratified sampling

Class imbalance is nearly universal in clinical biomarker datasets—pCR rates in TNBC are typically 30-50%, rare disease cohorts are worse, and some adverse event prediction tasks have imbalances of 100:1 or more. The failure mode is predictable: models trained without addressing imbalance learn to predict the majority class and look excellent on accuracy while being useless for the clinical question you actually care about. The fix is usually straightforward, but it requires recognizing the problem first.

#### Knows When to Stop
- Bad: Keeps iterating until the model performs well, then publishes.
- Good: Sets a stopping rule in advance, evaluates once on the test set, and reports the result honestly—including when it's underwhelming.

This is perhaps the least discussed skill and arguably the most important one for the field. The reproducibility crisis in biomarker research is largely a consequence of implicit indefinite iteration: researchers keep modifying their pipeline until the numbers look good, then present the final result as if it were the first attempt. A model that achieves AUC 0.72 after one pre-specified evaluation is more scientifically credible than a model that achieves AUC 0.85 after forty undocumented iterations. Learning to be comfortable reporting honest, modest results—and framing them accurately as a foundation for future work rather than a clinical breakthrough—is what separates researchers who contribute durable knowledge from those who contribute noise.


