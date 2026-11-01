# LearnML Starter Notes: Data Science, Machine Learning and Deep Learning

**Document type:** Original introductory reference notes prepared for the LearnML demonstration knowledge base.  
**Attribution:** Written for this project; not copied from a textbook.  
**Purpose:** Provides a small starter corpus so the application can be tested before additional books are added.

## 1. What is Data Science?

Data science is an interdisciplinary process for extracting useful knowledge from data. It combines statistical reasoning, programming, domain understanding, data management and communication. A typical workflow begins by defining a question, collecting relevant data, checking its quality, exploring patterns, selecting an analytical or predictive method, evaluating the result and communicating the limitations.

A descriptive question asks what happened. A diagnostic question asks why it happened. A predictive question asks what may happen next. A prescriptive question asks what action should be taken. The distinction matters because a dashboard that describes past sales and a model that predicts future sales solve different problems and need different validation methods.

## 2. Data preparation and exploratory analysis

A dataset is often represented as a table in which rows are observations and columns are variables. Numerical variables measure quantities; categorical variables represent groups or labels. Before modelling, inspect the number of rows and columns, data types, missing values, duplicate records, impossible values, unusual ranges and inconsistent category labels.

Missingness can be MCAR (missing completely at random), MAR (missing at random conditional on observed information) or MNAR (missing not at random). The mechanism affects how safe it is to remove or impute missing values. Common approaches include deleting a small number of unsuitable rows, imputing numerical values with a mean or median, imputing categorical values with a mode or an explicit missing category, and using model-based imputation. The choice should be justified and fitted on training data only when preparing a predictive model.

Exploratory data analysis (EDA) uses summaries and visualisations to understand distributions, relationships, outliers and data quality. Histograms show the distribution of a numerical variable. Box plots summarise the median, quartiles and potential outliers. Scatter plots reveal relationships between two numerical variables. Bar charts compare categories. A correlation coefficient describes linear association, not causation.

An outlier is an observation that is unusual relative to the data or an expected process. It may be a recording error, a rare but valid event or evidence of a different population. Do not automatically delete all outliers. Investigate their origin, compare model performance with and without them, and document the decision.

## 3. Statistics and probability

The population is the full group of interest; a sample is the subset actually observed. A population parameter is a fixed quantity describing the population, while a sample statistic is calculated from the sample and is used to estimate a parameter.

The mean is the arithmetic average and is sensitive to extreme values. The median is the middle ordered value and is more robust to skew and outliers. Variance measures average squared deviation from the mean; standard deviation is its square root and is expressed in the original units. The interquartile range (IQR) is Q3 minus Q1 and measures the spread of the middle half of the observations.

Probability is a numerical description of uncertainty between 0 and 1. For events A and B, the conditional probability is P(A|B) = P(A and B) / P(B), provided P(B) is not zero. Bayes' theorem is P(A|B) = P(B|A)P(A) / P(B). It updates a prior belief about A using observed evidence B.

A confidence interval is a procedure that, over repeated samples under its assumptions, captures the target parameter at the stated rate. A 95% confidence procedure does not mean that 95% of individual observations lie inside one calculated interval. A p-value is the probability, assuming the null hypothesis and model assumptions, of obtaining a test statistic at least as extreme as the one observed. It is not the probability that the null hypothesis is true.

## 4. Machine learning fundamentals

Machine learning uses data to fit a model that can make predictions or discover structure. In supervised learning, the training data contains input features X and a target y. Regression predicts a numerical target, such as house price. Classification predicts a class, such as whether a transaction is fraudulent. In unsupervised learning, there is no labelled target; clustering and dimensionality reduction are common examples.

A dataset is commonly split into training, validation and test portions. The training set fits model parameters. The validation set supports model and hyperparameter selection. The test set estimates performance on unseen data and should not be repeatedly used to tune the model. When observations have a time order, use a chronological split rather than randomly mixing past and future records.

Overfitting occurs when a model captures noise or details specific to the training sample and generalises poorly. Underfitting occurs when a model is too simple to capture useful patterns. Regularisation, cross-validation, more representative data, feature selection and simpler models can help control overfitting.

Data leakage occurs when information unavailable at prediction time enters the training process or evaluation. Examples include using a feature calculated from the target, scaling the full dataset before splitting, or randomly splitting time-dependent data so that future information influences training. Fit preprocessing steps on training data and apply the fitted transformations to validation and test data.

## 5. Linear and logistic regression

Simple linear regression models a numerical response y using one predictor x: y_hat = beta_0 + beta_1 x. The intercept beta_0 is the predicted response when x is zero. The slope beta_1 is the expected change in the predicted response for a one-unit increase in x, under the model. Multiple linear regression extends the equation to several predictors.

Ordinary least squares chooses coefficients that minimise the sum of squared residuals: SSE = sum((y_i - y_hat_i)^2). A residual is the observed value minus the predicted value. Common regression metrics include MAE, MSE, RMSE and R-squared. RMSE penalises large errors more strongly than MAE. R-squared describes the fraction of sample variation explained by the fitted model under its definition; it does not prove causation or guarantee good future predictions.

Logistic regression is used for classification, often binary classification. It models the log-odds as a linear function of the predictors. The sigmoid function converts a real-valued score z into a probability: sigma(z) = 1 / (1 + exp(-z)). A threshold converts the estimated probability into a class label. The threshold should reflect the relative costs of false positives and false negatives rather than automatically being fixed at 0.5.

## 6. Classification evaluation

A confusion matrix counts true positives, true negatives, false positives and false negatives. Accuracy is the fraction of predictions that are correct. Precision = TP / (TP + FP), and recall = TP / (TP + FN). The F1-score is the harmonic mean of precision and recall: F1 = 2 * precision * recall / (precision + recall). Specificity = TN / (TN + FP).

Accuracy can be misleading when classes are imbalanced. For a rare failure-detection problem, a model that predicts 'no failure' for every case may have high accuracy but zero recall for failures. Precision-recall curves, recall, precision, F1, ROC-AUC and the confusion matrix provide complementary views. Select metrics based on the practical cost of different errors.

## 7. Decision trees, random forests and boosting

A decision tree recursively splits the feature space into regions. For classification, a split can be selected by impurity measures such as Gini impurity or entropy. Trees are easy to interpret but can have high variance and overfit if allowed to grow deeply.

A random forest trains many decision trees on bootstrap samples and uses random subsets of features when splitting. Averaging the trees for regression or voting for classification usually reduces variance compared with a single deep tree. Random forests can model nonlinear relationships and interactions, but they can be less interpretable and may require more memory and computation.

Boosting builds a sequence of weak learners, with later learners focusing on patterns not captured by earlier ones. Gradient boosting optimises a loss function by adding learners that follow the loss gradient. Random forests primarily reduce variance through averaging; boosting often reduces bias by sequentially improving the ensemble, though both can be tuned to control overfitting.

## 8. Clustering and dimensionality reduction

K-Means partitions observations into K clusters by assigning each point to the nearest centroid and updating centroids to minimise within-cluster squared distances. The number K must be chosen. The elbow method examines the reduction in within-cluster sum of squares as K increases. The silhouette score compares within-cluster cohesion with separation from other clusters. K-Means is sensitive to feature scale, initial centroids and outliers, so numerical features are often scaled and multiple initialisations are useful.

Principal Component Analysis (PCA) creates new orthogonal directions that capture decreasing amounts of variance. For centred data, PCA can be computed using an eigendecomposition of the covariance matrix or singular value decomposition. The first principal component captures the largest possible variance among one-dimensional linear projections. PCA is sensitive to feature scales, so standardisation is often appropriate when variables have different units or ranges.

## 9. Neural networks and deep learning

An artificial neural network is a composition of parameterised linear transformations and nonlinear activation functions. For one layer, a simplified form is a = f(Wx + b), where x is the input, W is a weight matrix, b is a bias vector and f is an activation function. Without nonlinear activations, stacking linear layers is still equivalent to one linear transformation.

A loss function measures prediction error. Training adjusts model parameters to minimise the loss. Gradient descent updates parameters in the direction opposite the gradient: theta_new = theta_old - learning_rate * gradient. Backpropagation applies the chain rule efficiently to calculate gradients through the network. The learning rate controls the step size and strongly affects training stability.

A Convolutional Neural Network (CNN) uses local filters to detect spatial patterns, making it effective for images. Pooling or strided operations can reduce spatial dimensions. Recurrent Neural Networks (RNNs) process sequences by maintaining a hidden state. LSTMs and GRUs use gates to control information flow and help preserve longer-term dependencies. Transformers use attention mechanisms to model relationships between tokens and are widely used in modern language models.

## 10. Natural language processing and retrieval augmented generation

Text preprocessing may include normalisation, tokenisation, stop-word handling and stemming or lemmatisation. A bag-of-words representation counts or records word occurrences. TF-IDF weights terms by their importance in a document relative to a collection. Word embeddings map tokens or passages to dense vectors so semantic similarity can be estimated geometrically.

Retrieval-Augmented Generation (RAG) combines an information retrieval system with a generative language model. During indexing, documents are extracted, split into chunks, embedded and stored with metadata. At question time, the question is embedded, similar chunks are retrieved, and the selected passages are supplied as context to the language model. The model then generates an answer based on the retrieved evidence.

Chunk size involves a trade-off. Very small chunks may omit context; very large chunks may contain unrelated information and consume the context window. Overlap can preserve information at chunk boundaries but increases storage and retrieval redundancy. Metadata such as document title and page number is important for traceable citations.

RAG quality depends on both retrieval and generation. Retrieval evaluation asks whether the correct passages were found. Generation evaluation asks whether the answer is supported, accurate and useful. A fluent answer is not necessarily grounded. A reliable system should say when the retrieved sources are insufficient and should never invent citations.
