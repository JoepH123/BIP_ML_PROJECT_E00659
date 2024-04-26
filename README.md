missing: 
section 2 flow chart
section 4 placeholder figure

# BIP-ML-project-E00659
Alysia Yoon E04249
Joep Hillenaar E00659


# [Section 1] Introduction
Our project was developed to produce a model that predicts exemption VAT codes for each invoice line. Particularly using the IVAm field in the dataset. We pursued several avenues to determine the best model and picked the relevant criteria. We also implemented a user-friendly interface to facilitate easy interactions with the model in dashboard form. 

Given resources: 
BIP x Tech Presentation
BIP x Tech - Dataset Description 
BIP x Tech - Project: IVA
BIP X Tech - xlsx Dataset

# [Section 2] Methods

1. **Dataset Analysis**
   Our first step was to perform an initial explorative data analysis (EDA) procedure. We looked for missing values and anomalies to fully grasp the workability of the given dataset. Quantitatively, we were able to determine which columns were imputable and unusable due to the level of NaN's. This was done through:
   
   a) Deleting all rows for which IvaM is missing
   b) Calculating number of NaNs per column
   c) Calculating percentage of NaNs per column
   d) Combine both into a DataFrame for a cleaner display for simpler analysis
   
Three categories emerged from this step.
Columns with no missing values
Columns with a small percentage of missing values (<5%)
Columns with a large percentage of missing values (>50%)
**Note no cases between 2. and 3.**

For columns with no missing value, we **kept** them all to uphold data integrity
For columns (>5%), imputation is promising. All were categorical data types, and we **imputed** with mode.
For columns (>50%), imputation was deemed difficult.
Before deletion, we discussed two questions:
Does the presence of values in a column with many NaNs provide substantial predictive power? (This way we could use empty values)
Is the column with few NaNs valuable enough to apply a data imputation technique?
Concluded columns (>50%) to be **eliminated**. 

Qualitatively, we assumed that the (6) columns specifically mentioned in the given dataset description were important inputs and must be utilized in the model. All 6 columns were perfectly clean and required no imputation. For invoice information, Columns Description and Amount (not within the dataset 6) were also determined to be essential, which fall under (>5%) and  no missing values respectively. No further adjustments to be made.


2. **Proposed Idea**
We explore the performance of machine learning models on high-dimensional data. Specifically, we compare the efficacy of a neural network against that of a random forest and a decision tree. The primary challenge addressed is the management of high dimensionality resulting from:

  a) Text vectorization, which transforms textual data into a high-dimensional space.
  
  b) The creation of dummy variables for categorical features, which significantly increases the feature count with numerous categories.
  
3. **Design Decisions and Algorithm Selection**
Dimensionality Reduction: To manage the high dimensionality, we employ Truncated Singular Value Decomposition (TruncatedSVD). This technique reduces the feature space to a more manageable size while attempting to preserve the variance in the data. This reduction is crucial for improving model training times and avoiding overfitting.

4. **Model Selection**
Neural Network: We hypothesized that a neural network, due to its ability to model complex patterns, would be well-suited for high-dimensional data, even after dimensionality reduction.
Random Forest: Serves as a benchmark due to its robustness and effectiveness in handling numerous features without significant preprocessing. It's also less likely to overfit compared to simpler models.
Decision Tree: Investigated as a simpler alternative to assess if complexity in model architecture translates to significantly better performance.

5. **Training Overview**
Models are trained using the same subset of data to ensure a fair comparison. The training process for each model involves:
  a) Utilizing a standardized pipeline of preprocessing - including the application of TruncatedSVD - followed by model fitting.
  b) Tuning hyperparameters specific to each model type to optimize performance.
  c) Evaluating using common metrics such as accuracy, precision, recall, and F1-score to gauge each model's effectiveness.
  
# [Section 3] Experimental Design
Comparison of a total of 5 models of which three are text-to-vector methods.

**Experiment 1: Model Performance in High-Dimensional Space**
Main Purpose - To assess and compare the performance of neural networks (NN) and random forests (RF) in a high-dimensional feature space.

Baselines
The random forest model served as a baseline, known for handling high-dimensional spaces effectively.
The neural network model was tested against this baseline to determine if its higher complexity provided any significant performance benefit in high-dimensional data.

Evaluation Metrics
Accuracy: Indicates the overall rate of correct predictions made by the model.
F1 Score: a single measure that combines precision and recall. Useful when you have classes of different sizes and you want to ensure your model is both accurate and doesn't miss a significant number of instances. 
Precision and Recall: Additional metrics were considered to understand the trade-offs each model makes between false positives and false negatives.
Confusion Matrix: To gain deeper insights into the type and frequency of classification errors each model makes.

**Experiment 2: Text-to-Vector Transformation Methods**
Main Purpose - To compare the effectiveness of different text-to-vector transformation methods on model performance.

Baselines
English TFIDF used as a primary baseline due to its widespread use and efficacy.
Italian TFIDF used to determine the impact of language-specific vectorization on model performance.
Transformer embeddings were included to determine advancements in natural language processing and their effect on the model's ability to understand and classify textual data.

Evaluation Metrics
Accuracy: Measure for how well each text-to-vector method contributed to correct classifications.
F1 Score: Same definition as previous experiment
Confusion Matrix: Same definition as previous experiment

**Experiment 3: Model Comparison in Low-Dimensional Space**
Main Purpose - To investigate how neural networks (NN), random forests (RF), and decision trees (DT) perform on a lower-dimensional dataset, and to determine if a simpler model like DT could outperform more complex models.

Baselines
Random Forest and Decision Tree models were used as baselines due to their simplicity and interpretability.
The neural network was evaluated against these baselines to see if the dimensionality reduction affected the more complex model disproportionately.

Evaluation Metrics
Accuracy: Provides a general sense of model performance with reduced feature sets.
F1 Score: Helps evaluate the precision-recall balance in a more condensed feature space.
Precision and Recall: How often the model is right when it predicts something and how good it is at catching what it should. This is useful when fewer features change the way the model acts.
Confusion Matrix: Aids in understanding the areas where each model excels or fails, to guide future feature selection.


# [Section 4] Results
Our experiments led to several notable findings:

Model Performance:
Random Forest (RF) and Decision Tree (DT) models yielded similar results to the Neural Network (NN) when applied to the full feature set.
When using a reduced feature set, both RF and DT outperformed the NN, suggesting that these models are more robust to feature set reduction.
The NN's performance dipped more significantly than RF and DT in lower-dimensional spaces, indicating a possible over-reliance on the availability of high-dimensional data.

Feature Reduction:
The experiments demonstrated that both RF and DT maintain commendable performance despite a significant reduction in the number of features. This reinforces the notion that these models can effectively capture the underlying patterns in the data with fewer features (utilizing less computational power).


# [Section 5] Conclusions
Our project concludes the RF and DT models are superior in contexts with fewer features.
Given the comparable performance between RF and DT, along with the added advantage of simplicity and interpretability, **DT emerges as the preferred model.** The NN model does not demonstrate a clear advantage in the reduced feature space setting, reinforcing the suitability of more traditional, interpretable models for such tasks. The data and task fit better with a rule-based approach because the additional columns used in the dataset allow for creating an accurate model using deterministic rules. Random Forests (RF) and Decision Trees (DT) are inherently more aligned with a rule-based methodology. They function by creating decision rules that split the data based on feature values, which is particularly effective when there's a clear and logical structure to the data that can be captured with such rules. Thus the DT model is concluded to be the best-fit choice model for the task. 









