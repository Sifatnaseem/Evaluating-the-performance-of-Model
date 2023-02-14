# Importing the metrics package from sklearn library
from sklearn import metrics
import pandas as pd
# Creating the confusion matrix
y1_true=[1,0,0,0,1,0,0,1,0,1]
y1_pred=[1,0,0,1,1,0,1,1,0,1]
y2_true=[1,0,0,0,1,0,0,1,0,1]
y2_pred=[1,0,0,0,0,0,1,0,1,1]
cm1 = metrics.confusion_matrix(y1_true, y1_pred)
cm2=metrics.confusion_matrix (y2_true,y2_pred)
# Assigning columns names
cm_df1 = pd.DataFrame(cm1,
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
cm_df2 = pd.DataFrame(cm2,
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
# Showing the confusion matrix
print (cm_df1)
print (cm_df2)
 # Creating a function to report confusion metrics
def confusion_metrics (con_matrix):
# save confusion matrix and slice into four pieces
    TP = con_matrix[1][1]
    TN = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))  
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    #calculate the positive predictive value
    conf_PPV = (TP / float(TP + FP))
    #calculate the negative predictive value
    conf_NPV = (TN / float(TN + FN))
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}')
    print(f'positive predictive value:{round(conf_PPV,2)}')
    print(f'negative predictive value:{round(conf_NPV,2)}')
    return conf_accuracy,conf_sensitivity, conf_specificity,conf_PPV,conf_NPV
# comparing performance of Model
conf_accuracy1, conf_sensitivity1, conf_specificity1,conf_PPV1,conf_NPV1 = confusion_metrics(cm1)
conf_accuracy2, conf_sensitivity2, conf_specificity2,conf_PPV2,conf_NPV2 = confusion_metrics(cm2)
if conf_accuracy1 > conf_accuracy2:
    print("Test Method A has higher accuracy than Test Method B")
else:
    print("Test Method B has higher accuracy than Test Method A")

if conf_sensitivity1 > conf_sensitivity2:
    print("Test Method A has higher sensitivity than Test Method B")
else:
    print("Test Method B has higher sensitivity than Test Method A")

if conf_specificity1 > conf_specificity2:
    print("Test Method A has higher specificity than Test Method B")
elif conf_specificity1 == conf_specificity2:
    print("Both have same specificity ")
else:        
    print("Test Method B has higher specificity than Test Method A")





