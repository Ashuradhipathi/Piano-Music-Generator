The Basic algorithm for solving a task like this is:

- Collect data and classify it
- Divide data-set to teach-set and test-set
- Collect classifiers and vectorizers
- Fit each classifier with teach-set and calculate accuracy with test-set
- Find classifier with the biggest accuracy
- Write API

- **Fit**() — method for teaching classifier
- **Score**() — method that returns the mean accuracy on the given test data and labels.
- **Predict**() — method for making prediction. For example: ‘ham’ or ‘spam’
- **Predict_score**() — method that returns probability estimates for each predictions. For example: 0.8 for ‘ham’ and 0.2 for ‘spam’.

## Vectorizers

But every of this module does not understand plain text; they need an array of features. How to build feature vectors from plain text?  
For it, Scikit-learn has vectorizers: CountVectorizer, TfidfVectorizer, HashingVectorizer.

1. **CountVectorizer**: This method converts text into a matrix of token (word) counts. In other words, it counts the occurrence of each word in the text. The result is a sparse matrix where each row corresponds to a document and each column is a word from the vocabulary. The value in each cell is the count of the word in the corresponding document.
    
2. **TfidfVectorizer**: This method is similar to `CountVectorizer`, but instead of just counting the occurrences, it calculates the Term Frequency-Inverse Document Frequency (TF-IDF) value for each word. TF-IDF is a statistical measure that reflects how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.
    
3. **HashingVectorizer**: This method also converts text into a matrix of token occurrences, but it uses the hashing trick to encode tokens instead of keeping a dictionary of words. This makes it more memory-efficient than `CountVectorizer` and `TfidfVectorizer`, especially for large vocabularies, but it’s not possible to compute the inverse transform (from feature indices to string feature names) which can be a problem when trying to introspect which features are most important to a model.
    

Each of these methods has its own advantages and is suitable for different use cases. `CountVectorizer` is simple and straightforward, `TfidfVectorizer` can give better results when words’ importance should be weighted, and `HashingVectorizer` is useful when memory efficiency is a concern.

## execution

We need to divide one data-set (spam.csv) to two data-sets (teach-set and test-set) with the ratio 80/20 or 70/30.  
We will use teach-set for teaching classifier and test-set for calculating accuracy.

 
1. The `perform` function is defined, which takes as input a list of classifiers, a list of vectorizers, and training and testing data. For each combination of classifier and vectorizer, it fits the classifier on the vectorized training data and then evaluates it on the vectorized testing data. The score (accuracy) of each classifier-vectorizer combination is printed.
    
2. The script reads a CSV file named ‘spam.csv’ using `pandas.read_csv`. This file is expected to contain the spam dataset.
    
3. The dataset is split into a training set (`learn`) and a test set (`test`). The first 4400 items are used for training and the rest for testing.
    
4. Finally, the `perform` function is called with a list of classifiers, a list of vectorizers, and the training and testing data. The classifiers include various types of models from `sklearn`, such as Naive Bayes (`BernoulliNB`), Random Forest (`RandomForestClassifier`), and Support Vector Machines (`OneVsRestClassifier(SVC(kernel='linear'))`). The vectorizers include `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer`, which are used to convert the text data into numerical vectors that can be used as input to the classifiers.
    


##  machine learning algorithms used for classification tasks

1. **BernoulliNB**: This is a Naive Bayes classifier for data that is distributed according to multivariate Bernoulli distributions; i.e., there may be multiple features, but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.
    
2. **RandomForestClassifier**: This is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
    
3. **AdaBoostClassifier**: This is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
    
4. **BaggingClassifier**: A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
    
5. **ExtraTreesClassifier**: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
    
6. **GradientBoostingClassifier**: GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.
    
7. **DecisionTreeClassifier**: This is a classifier that makes decisions based on a tree structure, where each node in the tree corresponds to a feature in the input data, and the branches represent splits on the feature values that lead to different predictions.
    
8. **CalibratedClassifierCV**: This class performs probability calibration with isotonic regression or sigmoid. It generates calibrated probabilities after fitting.
    
9. **DummyClassifier**: This is a classifier that makes predictions using simple rules. It’s useful as a simple baseline to compare with other (real) classifiers.
    
10. **PassiveAggressiveClassifier**: This is an online learning algorithm that remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.
    
11. **RidgeClassifier**: This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case).
    
12. **RidgeClassifierCV**: Ridge classifier with built-in cross-validation.
    
13. **SGDClassifier**: Linear classifiers (SVM, logistic regression, etc.) with SGD training. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning.
    
14. **OneVsRestClassifier**: This strategy, also known as one-vs-all, is implemented in `OneVsRestClassifier`. The strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes.
    
15. **KNeighborsClassifier**: This is a classifier implementing the k-nearest neighbors vote.
    
```python 
with open('test_score.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['#', 'text', 'answer', 'predict', result])

  

    for row in csv_array:

        writer.writerow(row)
```


for writing, creates a CSV writer with a semicolon as the delimiter and a double quote as the quote character. It then writes a header row to the CSV file. After that, it iterates over `csv_arr` and writes each row to the CSV file.

The `quoting=csv.QUOTE_MINIMAL` option in your code tells the `csv.writer` to only quote fields which contain special characters such as the delimiter, quotechar or any of the characters in the lineterminator. In this case, since you’re using a semicolon as the delimiter and a double quote as the quote character, any field containing either of these characters will be quoted. This is the default quoting mode in `csv.writer`. If you want to change the quoting behavior, you can use other options like `csv.QUOTE_ALL`, `csv.QUOTE_NONNUMERIC`, or `csv.QUOTE_NONE`

This line of code opens a file named `test_score.csv` in write mode. The `newline=''` argument ensures that universal newline support is enabled.

## Flask Code
 The `if __name__ == '__main__':` line checks if this script is being run directly by the user and not being imported as a module. If the script is being run directly, the code inside this if-block will be executed.

The `port = int(os.environ.get('PORT', 5000))` line is getting the ‘PORT’ environment variable, which is the port number on which the server should run. If ‘PORT’ is not set, it defaults to 5000.

The `app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)` line starts the server. The server will listen on all public IPs (`0.0.0.0`) and on the port number specified by the ‘PORT’ environment variable. The `debug=True` argument means the server will provide detailed error messages if something goes wrong, and `use_reloader=True` means the server will restart itself whenever it detects a change in the source files.
