# BIP-ML-project-E00659
Alysia Yoon E04249
Joep Hillenaar E00659


# [Section 1] Introduction
Our project was developed to produce a model that predicts exemption VAT codes for each invoice line. This was done by predicting the IvaM field in the dataset while using the other columns in the data. We pursued several avenues to determine the best model and picked the relevant criteria explained further in the following report. We also implemented a user-friendly interface to facilitate easy interactions with the model in dashboard form. 

Given resources: 
- BIP x Tech Presentation
- BIP x Tech - Dataset Description 
- BIP x Tech - Project: IVA
- BIP X Tech - xlsx Dataset

Common terms with their abbreviations:
- Decision Tree (DT)
- Neural Network (NN)
- Random Forest (RF)

# [Section 2] Methods

For details on recreating environment, look at ``main.ipynb``

1. **Dataset Analysis**

   Our first step was to perform an initial explorative data analysis (EDA) procedure. We looked for missing values and anomalies to fully grasp the workability of the given dataset. Quantitatively, we were able to determine which columns were imputable and unusable due to the level of NaN's. This was done through:
   
      a) Deleting all rows for which IvaM is missing
      b) Calculating number of NaNs per column
      c) Calculating percentage of NaNs per column
   
   Three categories emerged from this step. The first one was columns with no missing values. We also found columns with a small percentage of missing values (<5%) and columns with a large percentage of missing values (>50%). There were however no columns with between 5 and 50 percent missing values. For columns with no missing value, we kept them all to uphold data integrity. For columns with little missing values, less than 5 percent, imputation is promising. All these columns were categorical data types, and we chose to impute missing values using the mode of each column. For columns with greater numbers of missing values, more than 50 percent, imputation was deemed difficult. Before deletion, we discussed two questions: Does the presence of values in a column with many NaNs provide substantial predictive power? If the answer is yes, we could use the empty values. Is the column with few NaNs valuable enough to apply a data imputation technique? By analyzing these questions for all the columns with more than 50 percent missing, values, we concluded that they could all be dropped. 
   
   Before continuing, we want to already make the distinction between the different complexities of models that we train and evaluate within this report. We use both low-dimensional models (reduced model), trained on a refined set of features, and high-dimensional models (full model), trained on the complete set of features for which we had enough non-missing data. The high-dimensional models required the missing value analysis and imputation method described above. The low-dimensional models, however, are based on a set of features for which no elaborate missing value methodology is needed. This set of features does not have a lot of missing values. These features only have to be transformed into features suitable for model input. 
   
   High Dimensional Model Approach: For the comprehensive model which utilized all available features, we implemented the imputation technique described above, where missing values in columns with less than 5 percent missing values were imputed using the mode value of this column. This resulted in throwing away only columns with more than 50 percent missing values. This approach was aimed at maximizing the dataset's completeness to enable a detailed exploration of all potential predictive signals.
   
   Low Dimensional Model Approach: In contrast, for the smaller models which focused on a reduced set of features, the imputation techniques were unnecessary altogether. This was due to the selection of mostly clean columns and those critical for the analysis, which either had no or very few missing values. In these models, we prioritized simplicity and computational efficiency.

2. **Proposed Idea**

   We explore the performance of machine learning models on high-dimensional and low-dimensional data. Specifically, we compare the efficacy of a neural network approach against that of a random forest approach for high-dimensional data. Furthermore, we compare a neural network to a random forest model and a decision tree for low-dimensional data. These different models brought about different challenges and choices.
  
3. **Design Decisions and Algorithm Selection**

   The low-dimensional models did not require much special attention, as their process was quite straightforward. Of course, we had to convert the raw features to model input, by creating dummy variables for categorical features and scaling numerical features. 

   The primary challenge addressed is the management of high dimensionality resulting from:

   a) Text vectorization, which transforms textual data into a high-dimensional space.

   b) The creation of dummy variables for categorical features, which significantly increases the feature count with numerous categories. When using all categorical columns, the number of dummy variables goes to very high numbers. 

   To address this issue, we used a dimensional reduction method. We employed Truncated Singular Value Decomposition (TruncatedSVD). This technique reduces the feature space to a more manageable size while attempting to preserve the variance in the data. This reduction is crucial for improving model training times and avoiding overfitting.

   Of course, besides the text-to-vector and dummy variable creation combined with dimensionality reduction, the high-dimensional models, also required the numerical columns to be scaled. 

4. **Model Selection**

   - Neural Network: We hypothesized that a neural network, due to its ability to model complex patterns, would be well-suited for high-dimensional data, even after dimensionality reduction.
   - Random Forest: Serves as a benchmark due to its robustness and effectiveness in handling numerous features without significant preprocessing. It's also less likely to overfit compared to simpler models.
   - Decision Tree: Investigated as a simpler alternative to assess if complexity in model architecture translates to significantly better performance, or whether a simple rule-based approach is suitable.

5. **Training Overview**

   Models are trained using the same subset of data to ensure a fair comparison. The training process for each model involves:

     a) Utilizing a standardized pipeline of preprocessing, as described in section `3` above. Splitting the data into train and test data, using an 80-20 percent split.
   
     b) Tuning hyperparameters specific to each model type to optimize performance. Most importantly: Number of Epochs (NN), Number of hidden layers (NN), Neuron per hidden layer (NN), Max tree depth (DT and RF), Number of trees (RF)
   
     c) Evaluating using common metrics such as accuracy, precision, recall, and F1-score to gauge each model's effectiveness.
  
# [Section 3] Experimental Design

In this report, we evaluate 5 models and 3 text-to-vector methods, broken down below in 3 separate experiments. 

**Experiment 1: Model Performance in High-Dimensional Space**

`Main Purpose` - To assess and compare the performance of neural networks (NN) and random forests (RF) in a high-dimensional feature space.

Benchmark
- The random forest model served as a baseline, known for handling high-dimensional spaces effectively.
- The neural network model was tested against this baseline to determine if its higher complexity provided any significant performance benefit in high-dimensional data.

Evaluation Metrics
- Accuracy: Indicates the overall rate of correct predictions made by the model.
- F1 Score: a single measure that combines precision and recall. Useful when you have classes of different sizes and you want to ensure your model is both accurate and doesn't miss a significant number of instances. 
- Precision and Recall: Additional metrics were considered to understand the trade-offs each model makes between false positives and false negatives.
- Confusion Matrix: To gain deeper insights into the type and frequency of classification errors each model makes.

**Experiment 2: Text-to-Vector Transformation Methods**

`Main Purpose` - To compare the effectiveness of different text-to-vector transformation methods on model performance.

Benchmark
- TF-IDF (= Term Frequency - Inverse Document Frequency) with an English WordLemmatizer (English TF-IDF) used as a primary baseline due to its widespread use and efficacy.
- TF-IDF with an English WordLemmatizer (Italian TF-IDF) used to determine the impact of language-specific vectorization on model performance.
- Transformer embeddings were included to determine advancements in natural language processing and their effect on the model's ability to understand and classify textual data despite the dataset's bilingual text columns. 

We utilize both English and Italian due to the dateset provided containing text in both languages. 

Evaluation Metrics
- Accuracy: Measure for how well each text-to-vector method contributed to correct classifications.
- Confusion Matrix: Same definition as previous experiment

**Experiment 3: Model Comparison in Low-Dimensional Space**

`Main Purpose` - To investigate how neural networks (NN), random forests (RF), and decision trees (DT) perform on a lower-dimensional dataset, and to determine if a simpler model like DT could outperform more complex models.

Benchmark
- Random Forest and Decision Tree models were used as a benchmark due to their simplicity and interpretability.
- The neural network was evaluated against these baselines to see if the dimensionality reduction affected the more complex model disproportionately.

Evaluation Metrics
- Accuracy: Provides a general sense of model performance with reduced feature sets.
- F1 Score: Helps evaluate the precision-recall balance in a more condensed feature space.
- Precision and Recall: How often the model is right when it predicts something and how good it is at catching what it should. This is useful when fewer features change the way the model acts.
- Confusion Matrix: Aids in understanding the areas where each model excels or fails, to guide future feature selection.

# [Section 4] Results
**Experiment 1: High-Dimensional Model Performance**

Neural Network (NN) vs. Random Forest (RF):

The Neural Network model achieved an accuracy of 98.0% on the test set, demonstrating strong capability in handling high-dimensional data through complex pattern recognition.
The Random Forest model obtained 97.5% accuracy, slightly lower than the neural network but still effective in managing high-dimensional spaces without extensive preprocessing.
The slightly higher neural network performance suggests its potential benefits in handling complex, high-dimensional datasets. However, the close performance indicates that both models are robust options for high-dimensional data.

**Experiment 2: Text-to-Vector Transformation Methods**

Impact of word-to-vector methods:

English TF-IDF did not significantly improve model performance, indicating the limited predictive power of the 'DescrizioneRiga' text column in the dataset. Italian TF-IDF, obtained similar results to English TFIDF, showing no substantial enhancement in model accuracy. The transformer embedding, despite being designed to capture more nuanced meanings of the text, did not lead to better predictive performance. This lack of significant improvement with advanced text vectorization methods suggests that the textual content of the 'DescrizioneRiga' column may not hold crucial information for predicting VAT codes. The focus might be better placed on other more predictive features within the dataset.

**Experiment 3: Low-Dimensional Model Comparison**

Performance in Reduced Feature Context:

The neural network model showed a noticeable decrease in performance in the reduced feature set, going from 98.0% in the high-dimensional model to 95.7% in the low-dimensional model, highlighting its dependency on a broader range of data inputs. The random forest and decision tree, however, performed robustly with the reduced set of features. The random forest model achieved an accuracy of 97.3%, and the Decision Tree was close with 97.2% accuracy. The strong performance of the random forest and decision tree in a reduced feature set suggests their suitability for scenarios where computational efficiency and model simplicity are prioritized. Their ability to maintain high accuracy with fewer inputs also indicates a better fit for practical applications where interpretability and operational efficiency are crucial. 

## High Dimensional Models

| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Neural Network       | 0.9803   | 0.8691    | 0.8494  | 0.8545   |
| RandomForestClassifier | 0.9750 | 0.7172    | 0.6643  | 0.6864   |

## Text-to-Vector Models

| Model                 | Accuracy |
|-----------------------|----------|
| English TF-IDF        | 0.886    |
| Italian TF-IDF        | 0.876    |
| Transformer embedding | 0.872    |

## Low Dimensional Models

| Model                | Accuracy | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Neural Network       | 0.9572   | 0.7030    | 0.7067  | 0.7003   |
| RandomForestClassifier | 0.9725 | 0.7113    | 0.6641  | 0.6789   |
| DecisionTreeClassifier | 0.9722 | 0.6864    | 0.6689  | 0.6718   |

**High-Dimensional vs Low-Dimensional:**
The performance of all models slightly decreases when moving from high-dimensional to low-dimensional data. This indicates that reducing the dimensionality removed some useful information beneficial for model accuracy. However, we concluded that this trade-off was acceptable by the reduced number of features which that the dimensionality reduction significantly sped up model training and made the process more explainable.

**Metric Consistency:**
The NN in the high dimensional setting shows a balanced performance across all metrics (accuracy, precision, recall, and F1-Score), which is a desirable trait in a model, indicating it does not overly favor one class over another or sacrifice precision for recall. The RF class performance (especially in precision and recall) suggests that the accuracy is very comparable to the NN model. The F1 score is, however, quite significantly lower than the F1 score of the NN. Why this might not be as much of a problem as initially expected is explained in the next section, where we analyze the confusion matrices (`confusion matrix analysis`). 

**Confusion matrix analysis**

<img src="images/confusion matrix rf (highlighted).png" width="900" />

<img src="images/confusion matrix nn (highlighted).png" width="900" />

Drawing from the confusion matrices, both models have strengths and weaknesses depending on the specific classes being predicted. The RF model shows many concentrated true positives in several classes over the diagonal of the confusion matrix, indicating a strong capacity to correctly identify these classes. Compared to the confusion matrix of the neural network, there are however, also more zeros on the diagonal indicating classes that were never predicted correctly. Besides, it also displays significant misclassifications, particularly for class 0, where a substantial number of instances are incorrectly predicted as this class whereas this class only contains 3 actual cases. The fact that this one class, in particular, is so problematic, leads to a lower F1 score for this class due to both low precision (many false positives) and low recall (many false negatives).

The confusion matrix of the neural network also shows a strong diagonal line with high values indicating many correctly classified test cases. It also has fewer zeros on the diagonal indicating that the model can predict more different classes. In contrast to the Random Forest model, it notably performs better in the main class where RF struggles, such as class 0, achieving higher true positives and fewer false negatives and positives, which implies better precision and recall, and thus a higher F1-score for these classes. This suggests that the NN may have more generalized learning that isn't as tightly fitted to specific classes compared to the RF, at the expense of high accuracy in classes that are easier to predict. This explanation could intuitively make sense from the perspective that random forest models create many rule-based trees. These rules are quite harsh and could cause problems in certain questionable (or edge) cases. Overall, bad performance in one class specifically has a bigger impact on the F1 score than more distributed mistakes in different classes. Therefore, we can state that there is some nuance to the importance of the F1 score when evaluating the models. A model with a higher F1 score, might not always make this a preferred model.

While RF models can perform very well in certain classes they may lack consistency across all classes. This is reflected in the higher disparities in class-specific performance, which could be problematic where uniform performance across categories is crucial. The NN models, on the other hand, tend to provide a more balanced performance across various classes, which can be beneficial in scenarios where it is crucial to maintain a reasonable level of accuracy and precision across a diverse set of categories. However, this comes at the cost of decreased accuracy observed in RF models for certain specific classes. Between the two choices of prioritizing accuracy or consistency across classes, RF is preferable when high performance in specific known classes is more important, while NN is for when overall balance and generalization across classes are needed. This analysis aligns with the notion that in simpler data contexts where interpretability and simplicity are required, simpler models like DT and RF might be advantageous, but with an acceptance of their limitations in handling all classes effectively. A last note on the difference between the confusion matrices is that knowing that RF performs worse on specific classes, can help a human supervisor of the model, and assist the model in its weaknesses. This can be very beneficial in situations where the model is used mostly as a tool for a human. If you were to remove this problematic class, the accuracy of the other classes would become even higher, making the human supported by AI even more effective. 

# [Section 5] Conclusions

RF models yielded similar results to the NN with respect to accuracy when applied to the full feature set. The F1 score was, however, higher for the NN. We then looked into the confusion matrix to find that there is some nuance to the F1 score. The confusion matrices also helped us reason about the benefits and potential downsides of each model. When using a reduced feature set, both RF and DT outperformed the NN, suggesting that these models are more robust to feature set reduction. The NN's performance dipped more significantly than RF and DT in lower-dimensional spaces, indicating a possible over-reliance on the availability of high-dimensional data. The experiments demonstrated that both RF and DT maintain commendable performance despite significantly reducing the number of features. This reinforces the notion that these models can effectively capture the underlying patterns in the data with fewer features (utilizing less computational power). Our project concludes that specifically, the decision tree model is superior in this context. As it can learn the crucial patterns in the data in a simplistic rule-based manner, the model is easily interpretable and explainable, and the computational costs of this model are the lowest. We also believe that it makes a lot of sense given the data and task, that the decision tree performs well. Given the data, we had to leverage the extra columns of the dataset to improve the incomplete "Iva" to "IvaM" column mapping. The deterministic rules learned by the decision tree using the additional columns of the dataset sufficiently inform the model to create an accurate predictor of the "IvaM" column.

**Future Work: Potential for Model Optimization:**
Especially given the lower F1 score of the RF and DT approaches, and the analysis of the confusion matrices, we believe that there is still some unachieved potential. By investigating why certain classes are problematic for the random forest and decision tree models, we can probably strongly improve the model's overall predictive power. We believe that the most promising directions for future research are revising feature selection and feature engineering steps. To do this most effectively, it is also important that you dive into which columns are problematic and why. Collaboration with a domain expert could be of great value to understand this problem more fundamentally.
