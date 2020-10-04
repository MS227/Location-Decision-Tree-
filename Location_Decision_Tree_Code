#Store the dataset in a DataFrame
import pandas as pd
location_data=pd.DataFrame(
[[20,22,3,"Yes"],[30,17,8,"No"],[10,30,2,"Yes"],
                [50,15,3,"Yes"],[60,16,5,"Yes"],[5,20,4,"No"],
                [30,12,3,"Yes"],[20,10,1,"Yes"],[70,11,5,"Yes"],
                [21,20,12,"No"],[8,10,5,"No"],[25,10,2,"Yes"],
                [10,12,0,"Yes"],[25,12,9,"Yes"],[3,30,1,"Yes"]])


#Adding headers to columns
location_data.columns=[["Person","Income","Shops","Successful"]]
print(location_data)

#Store the data in a csv file
location_data.to_csv("location.csv")

#Read the data from the csv file
import pandas as pd
location_data=pd.read_csv("location.csv", index_col=0)
print("\n\nThe dataset:\n", location_data)



#Split the features and the label columns
location_ftr = location_data[["Person","Income","Shops"]]
location_label = location_data[["Successful"]]
print("\n\nThe features data:\n", location_ftr)
print("\n\nThe label data:\n", location_label)




#Import the train_test_split function
from sklearn.model_selection import train_test_split
#Split the data into 75% training and 25% for testing
ftr_train,ftr_test,label_train,label_test=train_test_split(location_ftr,location_label,test_size=0.25)




#Import tree sub-package
from sklearn import tree 
#Create the decision tree model and set the max depth
decision_tree = tree.DecisionTreeClassifier(max_depth = 2)
#Train the model
decision_tree.fit(ftr_train,label_train)



#Predict the label of the testing features
prediction=decision_tree.predict(ftr_test)
print("\n\nThe predicition of the model are:\n", prediction)


#Import accuracy_score
from sklearn.metrics import accuracy_score

#Calculate accuracy
accuracy=accuracy_score(prediction,label_test)
print("\n\nAccuracy =",accuracy)
