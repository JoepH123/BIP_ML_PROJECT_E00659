# BIP-ML-project-E00659
Alysia Yoon E04249

Joep Hillenaar E00659


Project for the Machine Learning course at Luiss for BIP. This project centers around the development of a model that accurately predicts VAT Exemption codes.

Describe what is in the 4 notebooks, which discuss all the steps we took and the progress that we made

Give a good comparison of the different models: by looking at all the results, using the simplest model --> decision tree with the smallest dataset is most logical. Easily understandable. You can even plot the decision tree (however, it is massive).

We ended up implementing this final model decision tree into a usable user interface. 

Furthermore, to run this user interface, we had to create some assets. To create these, the ui_asset_creator.py file can be used.

# [Section 1] Introduction
Our project was developed to produce a model that predicts exemption VAT codes for each invoice line. Particularly using the IVAm field in the dataset. To determine the best model we pursued several avenues and picked the relevant criteria. We also implemented a user-friendly interface to facilitate easy interactions with the model in dashboard form. 

# [Section 2] Methods
Proposed ideas: Features, algorithm(s), training overview, design choices

A reader should be able to recreate your environment (e.g., conda list,
conda envexport, etc

A flowchart illustrating the steps in your machine-learning system

•	Comparing a neural network approach making use of high dimensionality to a random forest with high dimensionality
o	Using all features  however number of features becomes very high because of two reasons:
	the text to vector (high dimensional representation)
	the explosion of the number of features through the creation of dummy features for the categorical variables.
•	If you have A, B, C, in a variable, the you have to make var_B, and var_C dummy variables. This way the number increases heavily if you have high number of categories
	For this reason we use TruncatedSVD to limit the number of features, we summarize high dimensionality into lower dimensionality.
•	Comparing neural network, random forest and decision tree on lower dimensionality. 
•	Using random forest as benchmark for performance of the neural network
•	When realizing random forest wasn’t that much better, looked at if we can reach same results with even simpler model (decision tree).




# [Section 3] Experimental Design
Any experiments you conducted to demonstrate/validate the target contribution(s) of your project; indicate the
following for each experiment:
• The main purpose: 1-2 sentence high-level explanation
• Baseline(s): describe the method(s) that you used to compare your work to.
• Evaluation Metrics(s): which ones did you use and why?

# [Section 4] Results
• Main finding(s): report your results and what you might conclude from your work.
• Include at least one placeholder figure and/or table for communicating your findings.
• All the figures containing results should be generated from the code.

# [Section 5] Conclusions
Summarize in one paragraph the take-away point from your work.
• Include one paragraph to explain what questions may not be fully answered by your work as well as natural next steps for this direction of future work.
