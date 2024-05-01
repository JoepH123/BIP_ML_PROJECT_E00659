missing: 
section 2 flow chart
section 4 placeholder figure

# BIP-ML-project-E00659
Alysia Yoon E04249
Joep Hillenaar E00659


# [Section 1] Introduction
Our project was developed to produce a model that predicts exemption VAT codes for each invoice line. Particularly using the IvaM field in the dataset. We pursued several avenues to determine the best model and picked the relevant criteria explained further in the following report. We also implemented a user-friendly interface to facilitate easy interactions with the model in dashboard form. 

Given resources: 
- BIP x Tech Presentation
- BIP x Tech - Dataset Description 
- BIP x Tech - Project: IVA
- BIP X Tech - xlsx Dataset

# [Section 2] Methods

1. **Dataset Analysis**
   Our first step was to perform an initial explorative data analysis (EDA) procedure. We looked for missing values and anomalies to fully grasp the workability of the given dataset. Quantitatively, we were able to determine which columns were imputable and unusable due to the level of NaN's. This was done through:
   
   a) Deleting all rows for which IvaM is missing
   b) Calculating number of NaNs per column
   c) Calculating percentage of NaNs per column
   d) Combine both into a DataFrame for a cleaner display for simpler analysis
   
Three categories emerged from this step.
Columns with no missing values.
Columns with a small percentage of missing values (<5%).
Columns with a large percentage of missing values (>50%).
**Note no cases between 2. and 3.**

For columns with no missing value, we **kept** them all to uphold data integrity
For columns (>5%), imputation is promising. All were categorical data types, and we chose to impute missing values using the mode of each column. 


For columns (>50%), imputation was deemed difficult.
Before deletion, we discussed two questions:
Does the presence of values in a column with many NaNs provide substantial predictive power? (This way we could use empty values)
Is the column with few NaNs valuable enough to apply a data imputation technique?
Concluded columns (>50%) to be **eliminated**. 

In our analysis, we identified that six specific columns highlighted in the dataset description were crucial inputs for our model. These columns were perfectly clean, requiring no imputation or adjustments, thus they were used in both the full and reduced feature models. However, we would like to highlight that the columns were transformed into features.

Full Model Approach: For the comprehensive model which utilized all available features, we implemented a rigorous imputation strategy where missing values in columns with significant but manageable missing data (>5% and <50%) were imputed using the mode. This approach was aimed at maximizing the dataset's completeness to enable a detailed exploration of all potential predictive signals.

Reduced Feature Model Approach: In contrast, for the smaller models which focused on a reduced set of features, the imputation techniques were simplified or altogether unnecessary. This was due to the selection of mostly clean columns and those critical for the analysis, which either had no missing values or were not significantly impacted by missing data. In these models, we prioritized simplicity and computational efficiency, eliminating the need for complex data imputation processes found in the full model setup.



2. **Proposed Idea**
We explore the performance of machine learning models on high-dimensional data. Specifically, we compare the efficacy of a neural network against that of a random forest and a decision tree. The primary challenge addressed is the management of high dimensionality resulting from:

   a) Text vectorization, which transforms textual data into a high-dimensional space.

   b) The creation of dummy variables for categorical features, which significantly increases the feature count with numerous categories.
  
4. **Design Decisions and Algorithm Selection**
Dimensionality Reduction: To manage the high dimensionality, we employ Truncated Singular Value Decomposition (TruncatedSVD). This technique reduces the feature space to a more manageable size while attempting to preserve the variance in the data. This reduction is crucial for improving model training times and avoiding overfitting.

5. **Model Selection**
Neural Network: We hypothesized that a neural network, due to its ability to model complex patterns, would be well-suited for high-dimensional data, even after dimensionality reduction.
Random Forest: Serves as a benchmark due to its robustness and effectiveness in handling numerous features without significant preprocessing. It's also less likely to overfit compared to simpler models.
Decision Tree: Investigated as a simpler alternative to assess if complexity in model architecture translates to significantly better performance.

6. **Training Overview**
Models are trained using the same subset of data to ensure a fair comparison. The training process for each model involves:

     a) Utilizing a standardized pipeline of preprocessing - including the application of TruncatedSVD - followed by model fitting.
   
     b) Tuning hyperparameters specific to each model type to optimize performance.
   
     c) Evaluating using common metrics such as accuracy, precision, recall, and F1-score to gauge each model's effectiveness.
  
# [Section 3] Experimental Design
In the following report, we evaluate 5 models and 3 text-to-vector methods, broken down below in 3 separate experiments. 


**Experiment 1: Model Performance in High-Dimensional Space**
Main Purpose - To assess and compare the performance of neural networks (NN) and random forests (RF) in a high-dimensional feature space.

Benchmark
The random forest model served as a baseline, known for handling high-dimensional spaces effectively.
The neural network model was tested against this baseline to determine if its higher complexity provided any significant performance benefit in high-dimensional data.

Evaluation Metrics
Accuracy: Indicates the overall rate of correct predictions made by the model.
F1 Score: a single measure that combines precision and recall. Useful when you have classes of different sizes and you want to ensure your model is both accurate and doesn't miss a significant number of instances. 
Precision and Recall: Additional metrics were considered to understand the trade-offs each model makes between false positives and false negatives.
Confusion Matrix: To gain deeper insights into the type and frequency of classification errors each model makes.

**Experiment 2: Text-to-Vector Transformation Methods**
Main Purpose - To compare the effectiveness of different text-to-vector transformation methods on model performance.

Benchmark
English TFIDF with an English WordLemmatizer used as a primary baseline due to its widespread use and efficacy.
Italian TFIDF used to determine the impact of language-specific vectorization on model performance.
Transformer embeddings were included to determine advancements in natural language processing and their effect on the model's ability to understand and classify textual data despite the dataset's bilingual text columns. 
We utilize both English and Italian due to the dateset provided containing text in both languages. 

Evaluation Metrics
Accuracy: Measure for how well each text-to-vector method contributed to correct classifications.
F1 Score: Same definition as previous experiment
Confusion Matrix: Same definition as previous experiment

**Experiment 3: Model Comparison in Low-Dimensional Space**
Main Purpose - To investigate how neural networks (NN), random forests (RF), and decision trees (DT) perform on a lower-dimensional dataset, and to determine if a simpler model like DT could outperform more complex models.

Benchmark
Random Forest and Decision Tree models were used as a benchmark due to their simplicity and interpretability.
The neural network was evaluated against these baselines to see if the dimensionality reduction affected the more complex model disproportionately.

Evaluation Metrics
Accuracy: Provides a general sense of model performance with reduced feature sets.
F1 Score: Helps evaluate the precision-recall balance in a more condensed feature space.
Precision and Recall: How often the model is right when it predicts something and how good it is at catching what it should. This is useful when fewer features change the way the model acts.
Confusion Matrix: Aids in understanding the areas where each model excels or fails, to guide future feature selection.


# [Section 4] Results
**Experiment 1: High-Dimensional Model Performance**
Neural Network (NN) vs. Random Forest (RF):
Neural Network: Achieved an accuracy of 98.0% on the test set, demonstrating strong capability in handling high-dimensional data through complex pattern recognition.
Random Forest: Recorded 97.5% accuracy, slightly lower than the neural network but still effective in managing high-dimensional spaces without extensive preprocessing.
Analysis: The slightly higher neural network performance suggests its potential benefits in handling complex, high-dimensional datasets. However, the close performance indicates that both models are robust options for high-dimensional data.

**Experiment 2: Text-to-Vector Transformation Methods**
Impact of Vectorization Techniques:
English TFIDF: Did not significantly improve model performance, indicating the limited predictive power of the 'DescrizioneRiga' text column in the dataset.
Italian TFIDF: Similar results to English TFIDF, showing no substantial enhancement in model accuracy.
Transformer Embeddings: Despite capturing more nuanced meanings of the text, transformer embeddings did not lead to better predictive performance.
Analysis: The lack of significant improvement with advanced text vectorization methods suggests that the textual content of the 'DescrizioneRiga' column may not hold crucial information for predicting VAT codes. The focus might be better placed on other more predictive features within the dataset.

**Experiment 3: Low-Dimensional Model Comparison**
Performance in Reduced Feature Context:
Neural Network: Showed a marked decrease in performance in the reduced feature set, highlighting its dependency on a broader range of data inputs.
Random Forest and Decision Tree: Both models performed robustly with reduced features. The Random Forest achieved an accuracy of 97.3%, and the Decision Tree was close with 97.2% accuracy.
Analysis: The strong performance of the Random Forest and Decision Tree in a reduced feature set suggests their suitability for scenarios where computational efficiency and model simplicity are prioritized. Their ability to maintain high accuracy with fewer inputs also indicates a better fit for practical applications where interpretability and operational efficiency are crucial.

<img src="images/confusion matrix rf (highlighted).png" width="900" />

<img src="images/confusion matrix nn (highlighted).png" width="900" />

# [Section 5] Conclusions

The experiments underscore the effectiveness of simpler, rule-based models like Decision Trees and Random Forests in various data contexts. In contrast, Neural Networks, while powerful in high-dimensional settings, may not offer substantial advantages in situations where data features are limited or when interpretability and simplicity are required. Although RF and DT have higher disparities in the first class therefore resulting in a lower F1 score, the other classes are outputted with better correlation while the NN performs inversely. Therefore, accuracy can be the better measure as a comparison. 

Random Forest (RF) models yielded similar results to the Neural Network (NN) when applied to the full feature set.
When using a reduced feature set, both RF and DT outperformed the NN, suggesting that these models are more robust to feature set reduction.
The NN's performance dipped more significantly than RF and DT in lower-dimensional spaces, indicating a possible over-reliance on the availability of high-dimensional data. The experiments demonstrated that both RF and DT maintain commendable performance despite significantly reducing the number of features. This reinforces the notion that these models can effectively capture the underlying patterns in the data with fewer features (utilizing less computational power).

Our project concludes the RF and DT models are superior in contexts with fewer features.

Given the comparable performance between RF and DT, along with the added advantage of simplicity and interpretability, **DT emerges as the preferred model.** Weighing the computational power to perform the calculation with the measurements discussed previously, the DT  has the added advantage of simplicity and interprebility. The NN model does not demonstrate a clear advantage in the reduced feature space setting, reinforcing the suitability of more traditional, interpretable models for such tasks. The data and task fit better with a rule-based approach because the additional columns used in the dataset allow for creating an accurate model using deterministic rules. Random Forests (RF) and Decision Trees (DT) are inherently more aligned with a rule-based methodology. They function by creating decision rules that split the data based on feature values, which is particularly effective when there's a clear and logical structure to the data that can be captured with such rules. Thus the DT model is concluded to be the best-fit choice model for the task. 
