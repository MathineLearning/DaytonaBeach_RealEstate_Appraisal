# -*- coding: utf-8 -*-
"""
Created on 7/19/2020

Decision tree of Daytona real estate market. Goes through Gini and entropy with  info gain. 
At the end is a visualization of it.

@author: nikre
"""
  
# Importing the required packages 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#Vis imports
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

#SOURCE: https://www.geeksforgeeks.org/decision-tree-implementation-python/    
#importing Data
col_names = ['Beds', 'Baths', 'SquareFeet','Price','Acre','Result']
House_data = pd.read_csv( r"~/Documents/Semester5/Semester 5.1/MA 440 Proj/wholedata.csv", 
    sep= ',', header = None, names = col_names)
House_data.head()

#Feature variables and target variables
feature_cols = ['Beds', 'Baths', 'SquareFeet','Price','Acre']
X1 = House_data[feature_cols] # Feature Variables
Y1 = House_data.Result # Target variable

# Split Data 
def splitdata(House_data): 

    # Train and Test 
    X1_train, X1_test, y1_train, y1_test = train_test_split(  
    X1, Y1, test_size = 0.33, random_state = 100) 
      
    return X1, Y1, X1_train, X1_test, y1_train, y1_test 
      
#Gini Index. 
def train_gini(X1_train, X1_test, y1_train): 
  
    # classifier With Gini
    classifier_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    #Training 
    classifier_gini.fit(X1_train, y1_train) 
    
    #Visualization conversion and output
    dot_data = StringIO()
    export_graphviz(classifier_gini, out_file=dot_data, filled=True, rounded=True,
    special_characters=True,feature_names = feature_cols,class_names=['Beach','Halifax','Inland'])#class_names=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('GitHub_Gini_Report.png')
    Image(graph.create_png())
    
    return classifier_gini 
      
#Entropy. 
def train_entropy(X1_train, X1_test, y1_train): 
  
    # classifier With Entropy
    classifier_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    #Training 
    classifier_entropy.fit(X1_train, y1_train) 
    
    #Visualization conversion and output
    dot_data = StringIO()
    export_graphviz(classifier_entropy, out_file=dot_data, filled=True, rounded=True,
    special_characters=True,feature_names = feature_cols,class_names=['Beach','Halifax','Inland'])#class_names=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('GitHub_Entropy_Report.png')
    Image(graph.create_png())

    return classifier_entropy 
  
  
#Predictions 
def prediction(X1_test, clf_object): 
  
    # Predicton with giniIndex 
    y1_prediction = clf_object.predict(X1_test)  
    return y1_prediction 
      
#Accuracy 
def calculate_accuracy(y1_test, y1_prediction): 
      
    print("Confusion Matrix: \n", 
        confusion_matrix(y1_test, y1_prediction)) 
      
    print ("Accuracy (%): ", 
    accuracy_score(y1_test,y1_prediction)*100) 
      
    print("Report : ", 
    classification_report(y1_test, y1_prediction)) 
  
def central(): 
 
    X1, Y1, X1_train, X1_test, y1_train, y1_test = splitdata(House_data) 
    classifier_gini = train_gini(X1_train, X1_test, y1_train) 
    classifier_entropy = train_entropy(X1_train, X1_test, y1_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_prediction_gini = prediction(X1_test, classifier_gini) 
    calculate_accuracy(y1_test, y_prediction_gini) 
    print("===================================================================")  
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_prediction_entropy = prediction(X1_test, classifier_entropy) 
    calculate_accuracy(y1_test, y_prediction_entropy) 
      
      
# Calling main function 
if __name__=="__main__": 
    central()

#================================================================================================    
#================================================================================================
#       VISUALIZATION
#================================================================================================
#================================================================================================    
"""
print('\n')
print('\n')


#SOURCE: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#Import Data For Visualization

#Train and Test
X2_train, X2_test, y2_train, y2_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 100)


DT_classifier = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5)
DT_classifier2 = DT_classifier.fit(X2_train,y2_train)

#clftest = clf.fit(X_test,y_test)

y2_prediction = DT_classifier.predict(X2_test)

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Results Using Sci-Kit Learn:") 
result = confusion_matrix(y2_test, y2_prediction)
print("Confusion Matrix:")
print(result)
result2 = accuracy_score(y2_test,y2_prediction)
print("Accuracy (%): ", result2 * 100)
result1 = classification_report(y2_test, y2_prediction)
print("Classification Report:",)
print (result1)


#Visualization conversion and output
dot_data = StringIO()
export_graphviz(DT_classifier2, out_file=dot_data, filled=True, rounded=True,
   special_characters=True,feature_names = feature_cols,class_names=['Beach','Halifax','Inland'])#class_names=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('GitHub_Report.png')
Image(graph.create_png())
"""