# Heart-Disease-Prediction-App
HEART DISEASE PREDICTION

1	CHAPTER 

1.1.	INTRODUCTION

Machine learning is a branch of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to learn from data and make predictions or decision without being explicitly programmed. Machine learning algorithms can be divided into three main categories: supervised learning, unsupervised learning and reinforcement learning. In this report, the focus is mainly on the supervised learning category in which correct outputs and set of features associated with the outputs already exist. Some algorithms are therefore used to train them with the existing data and then try to predict the output of the new data with only features associated with them. The algorithms used in supervised learning include the K-Nearest Neighbour (KNN) and the Support Vector Machine (SVM). These are the two machine learning algorithms that have been used to predict the likelihood of a person having heart disease based on their medical history and other relevant factors (Kaggle, 2023).
KNN is a type of supervised learning algorithm that is used for classification tasks. It works by finding the K nearest data points in the training dataset to a given input data point, and then classifying the input data point based on the class labels of the K nearest neighbour and similarity index. SVM on the other hand works by finding the hyperplane that best separates the data points of different classes. In the case of heart disease prediction, the SVM finds the hyper plane that best separates the data points of the people who have heart disease from those who do not (Kaggle, 2023). The choice of the algorithm used depends on the specific characteristics of the dataset and the goal of the analysis. In this case, both algorithms were used to compare their performances and accuracies.

1.2.	RELATED WORK

1.2.1. Research on heart disease prediction models.

Previous studies have explored the use of various machine learning models for predicting heart disease. For example, a study conducted by Rajagopal et al. (2019) used a deep learning approach to predict the risk of heart disease and it managed to achieve an accuracy of 92.47%. Singh et al. (2019) also conducted a study in which a decision tree algorithm was used to predict the risk of heart disease based on patient data, achieving an accuracy of 94.33% These studies demonstrate the potential of machine learning techniques for heart disease prediction.

 1.2.2. Comparison of SVM and KNN for heart disease prediction.

Several studies have compared the performance of SVM and KNN for heart disease prediction. For example, a study by Li et al. (2019) compared SVM, KNN, and logistic regression algorithms for predicting the risk of heart disease and found that SVM achieved the highest accuracy of 89.4%. Another study by Chavan and Gawali (2020) compared SVM and KNN algorithms for predicting heart disease and found that SVM outperformed KNN in terms of accuracy and sensitivity.

1.2.3.	Flask application development for healthcare.

Flask is now a popular web framework for building web applications in Python. Several studies and articles have discussed the use of Flask for developing medical applications. For example, in a study by Bhattacharjee et al. (2021) used Flask to develop a web-based platform for managing patient data and appointments. Flask can also be used to develop a telemedicine platform for consultations among healthcare providers.

1.2.4.	Privacy and security in healthcare applications.

 It is important to consider Machine learning principle for privacy and security concerns when developing healthcare applications because patient data are sensitive. Several studies and articles have discussed best practices for ensuring the privacy and security of patient data in healthcare applications. For example, in a study by Kaur et al. (2021), a privacy-preserving framework was proposed for healthcare data sharing.


2	CHAPTER 2

2.1	METHODOLOGY

For training building the model we used the following 7 steps
1.	Collecting data
2.	Data preprocessing
3.	Choosing model
4.	Training model
5.	Hyper parameter tuning
6.	Evaluation
7.	Prediction

2.1.1	DATA COLLECTION

The data that was used for this machine learning model was already prodided.
2.1.2	DATA PREPROCESSING

The LabelEncoder data preprocessor from the scikit-learn Python library was used for data preprocessing. To convert categorical variables into numerical variables, the LabelEncoder class was used. Using the LabelEncoder's fit transform method, categorical variables such as 'Sex,' 'ChestPainType,' 'RestingECG,' 'ExerciseAngina,' and 'ST Slope' were converted into numerical values. The label Enoder accomplished this by assigning a numeric value from 0 to N-1 to each distinct value of the feature, where N is the total number of distinct values (Kumar, 2021). Each of the categorical variable's numerical values were then stored in separate variables.
The categorical variables were then removed from the input data frame using the drop method. The numerical values obtained earlier were then added to the input data frame as new columns using the pandas data frame's 'assign' method. Finally, the numpy library's column stack method was used to combine all of the input variables and output variable (Heart Disease) into a single array. This array was used to train machine learning models.

2.1.3	CHOOSING MODEL

SVM is an acronym for Support Vector Machines. It is a type of supervised, nonlinear machine learning algorithm developed for binary classification. In our project model we are using SVM algorithms because of its ability to handle high dimensional data and non-linear relationships between the input features and output variables(Brownlee,2016). SVM is selected for its robustness to noisy and incomplete data as well as its high performance. Our goal is to accurately make heart failure prediction. We have trained SVM classifier model using 80% of the input data. This classifier has been used for diagnosis based on input features to distinguish if the patient is at a higher risk of heart failure.
On the other hand, for educational purposes, we tried using the KNN classification method to compare with the SVM classifications performance. The k-Nearest Neighbors algorithm (or KNN) uses a distance metric to find the k most similar instances in the training data for a new instance and takes the mean outcome of the neighbors as the prediction (Brownlee, 2016).

2.1.4	HYPERPARAMETER TUNING

This is the process of finding the best values for the hyper parameters of a machine leaning models in order to optimize the performance on a given dataset. It involves trying out different combinations of hyper parameters and evaluating their performance of the model on a validation set or through cross-validation. The goal is to find the combination of hyper parameters that yield the best performance on the validation set, while avoiding overfitting to the training set (Sabastain Raschka, 2019).
When using KNN, there are several hyper parameters that are to be considered, which include:

1.	The number of neighbours (K):

This determines the number of nearest neighbours to consider when making a prediction. The value significantly impacts the model’s accuracy as choosing too few neighbours may result in overfitting and choosing too many neighbours may result in under fitting.

2.	Distance metrics:

These are used to measure the similarity between the samples selected. The most commonly used is the Euclidean distance, but other metrics like the Manhattan and Chebyshev distance can also be used.

3.	Weight function

The weight function determines the importance of each neighbour in the prediction. The commonly used weights include the uniform based weights in which all neighbours are equally important and the distance based in which closer neighbours have higher weight.
Tuning these parameters uses different techniques such as grid-search, random search and Bayesian optimization. The models being explained in this report used the grid-search technique which specifies a range of values for each hyper parameter and evaluates the performance of the model for all possible combinations of hyper parameter values (Alpaydin, 2010).
 

Similarly, SVM model also has hyper parameters that need to be tuned to optimize its performance.
 Important parameters in the SVM model include:

4.	Kernel type

This is used to transform the input data into higher dimensional feature space. Most commonly used kernel functions are linear, polynomial and radial basis functions. In case of this SVM model, the linear function was used.

5.	Regularization parameter (C):

This parameter controls the trade-off between maximizing margin and maximizing the classification errors. Large values of C results to smaller margins but lead to better accuracy on the training data.

6.	Kernel coefficient (gamma):

This controls the shape of the decision boundary. Large values of gamma result to more complex decision boundaries that can better fit the training data but may over fit on the testing data. Conversely, smaller gamma values result in simpler decision boundary but may under fit the training data.

7.	Class weights:

These are used to adjust the importance of each class during training, which can help to balance the models performance (Sabastain Raschka, 2019) (Alpaydin, 2010).
The grid-search technique was used as well to tune these parameters.

 
	

2.2. EVALUATION

In KNN we used the accuracy score function to print the training and testing accuracies. These accuracies are also printed using the ‘best_score’ attribute and the ‘accuracy_score()’ function, respectively. (Muller, 2017)
 


2.3. FLASK APPLICATION

Flask is a Python-based microweb platform that allows users to add application functionality as if they were built into the framework itself. Below are some of the program modules that were used in the development of the webpage.
•	svm.pkl- This contains the machine learning model to predict the likelihood of a  heart disease. SVM provided the accuracy of 85% training accuracy and 84% testing accuracy with all the features, we integrated this as a predictive model in the svm.pkl file.
•	Knn.pkl. This also contains the knn classifier model which has 75% training accuracy and 69% testing accuracy. 
•	heartdisease.py- This package includes Flask Application Programming Interface(API) that receives patients attributes information through Graphical User Interface or API calls, computes the predicted value using our model, and returns it.
•	Templates- The HTML form (input.html & results.html) this folder allows the user to enter patients attributes information and show the results on the prediction outcome page.
•	Static- This folder contains the css file which has the styling required for our HTML form.

2.4. LIBRARIES USED

The code is importing packages such as flask, pickle, numpy, and scikit-learn’s KNeighborsClassifier. After importing the necessary packages, the code loads a saved KNeighborsClassifier model using pickle. The model was saved in a pickle file format with the extension ".pkl". Initialization of a Flask application instance is then done using the Flask instructor with an argument ‘name’. 
The Pandas and Numpy libraries are imported to handle data manipulation and analysis. The K Neighbors Classifier class from sklearn-neighbors is used to implement the KNN algorithm for classification. Finally, the pickle library is exporting the trained model.

2.5.	 OPERATION 

The first route ('/') specifies the home page and uses the render template () function to render the 'input.html' template. The second path ('/results') directs to the results page and accepts form submissions via the POST method. The patient input() function is called when the form is submitted. Using the request object, this function reads in the form data (patient information) and stores the input values in variables. The input values are then combined into a list and the machine learning model (model) is called to make a prediction based on the input data. The prediction variable stores the prediction. The status variable is set to "Negative" or "Positive" depending on the prediction value. Finally, the function calls the'results.html' template, passing the status variable as a parameter. The if __name__ == '__main__': statement ensures that the Flask application runs only if this file is running as the main program. The 'input.html' file is an HTML template that defines a form where patients' information can be entered. The'results.html' file is another HTML template that displays to the user the predicted status (Positive or Negative).

2.6.	 METHOD POST

8.	Method = “post “
A method = “post” attribute value specifies that the data will be sent to the server by storing it in an HTTP request body. This method is used to transfer data securely using HTTP headers. The method is also used when data is added or modified on the server.

9.	Request method==post 

This is the request-response protocol between a client (browser) and server. The browser sends a request to the server, then server returns a response to the client. The response contains status information about the request and may also contain the requested content.

2.7.	 RESULTS

When the user runs the application, results are displayed. The application checks for valid input in the field. When the user enters an invalid information for the provided parameters, the page reloads. When valid information is entered by the user, the application will display whether the user has the problem or not.


 

2.8.	FUTURE WORK

The dataset that we worked on was just for a small population. In the future, there is a need to work with a larger population that can help to investigate a more detailed dataset. For example, considering the total population of Malawian patients with additional features in the future and it should aim at increasing accuracy of the model.

3.	CONCLUSION

In this project, machine learning models are trained using SVM and KNN algorithms with an aim of predicting possible heart disease. When the results were compared with algorithm of KNN, SVM was found to be better classifier than KNN. The training and testing accuracy of SVM is higher than that of KNN. SVM is used for implementing the gradient boosting framework and it improves performance of the model. From this project, it is also observed that KNN supports various objective functions and is extensible that one can define his or her own objectives easily.

REFERENCES

1.	Alpaydin, E. (2010). Introduction to machine Learning. In E. Alpaydin, Introduction to Machine Learning (p. 584).
2.	Kaggle. (2023). Retrieved from kaggle: https://www.kaggle.com
3.	Sabastain Raschka, V. M. (2019). Python Mchine Learning. In V. M. Sabastain Raschka, Python Machine Learning (p. 772). Birmigham, UK: Packt Publishing.
4.	Chavan, D. D., & Gawali, B. V. (2020). Comparative study of SVM and KNN algorithms for heart disease prediction. International Journal of Computer Sciences and Engineering, 8(1), 60-64.
5.	Li, L., Li, H., Zhang, Y., & Yang, J. (2019). A comparative study of machine learning algorithms for predicting heart disease. Journal of Healthcare Engineering, 2019, 1-9.
6.	Singh, A., Singh, R., & Singh, A. (2019). Predictive modeling for heart disease using decision tree approach. International Journal of Computer Applications, 181(44), 26-30.
7.	Rajagopal, H., Gangadhar, K., & Manjusha, K. (2019). Deep learning based prediction of heart disease using feature selection and parameter tuning. 3rd International Conference on Inventive Systems and Control, Coimbatore, India, pp. 912-916.
8.	Kumar, S., Singh, D., & Rana, D. (2021). Principal component analysis based feature selection for heart disease prediction. 2nd International Conference on Inventive Research in Computing Applications, Coimbatore, India, pp. 1-5.
9.	Bhattacharjee, D., Datta, S., & Chakraborty, D. (2021). Development of a web-based healthcare platform for effective patient data management. Journal of King Saud University-Computer and Information Sciences, 33(1), 45-50.
10.	Kumar, V. (2021, September 10). Categorical Data Encoding with Sklearn LabelEncoder and OneHotEncoder. MLK - Machine Learning Knowledge.
 https://machinelearningknowledge.ai/categorical-data-encoding-with-sklearn-labelencoder- and-onehotencoder/#Label_Encoding


