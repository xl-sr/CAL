import numpy as np
from sklearn.metrics import confusion_matrix
    
def get_intersection_union_per_class(confusion_matrix):
    number_of_labels = confusion_matrix.shape[0]
    matrix_diagonal = [confusion_matrix[i][i] for i in range(number_of_labels)]
    errors_summed_by_row = [0] * number_of_labels
    
    for row in range(number_of_labels):
        for column in range(number_of_labels):
            if row != column:
                errors_summed_by_row[row] += confusion_matrix[row][column]
    errors_summed_by_column = [0] * number_of_labels
    
    for column in range(number_of_labels):
        for row in range(number_of_labels):
            if row != column:
                errors_summed_by_column[column] += confusion_matrix[row][column]
        
    divisor = [0] * number_of_labels
    for i in range(number_of_labels):
        divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
        if matrix_diagonal[i] == 0:
            divisor[i] = 1
               
    return [float(matrix_diagonal[i]) / divisor[i] for i in range(number_of_labels)]
    
def calculate_scores(cl, pred_cl):
    cm = confusion_matrix(cl, pred_cl)
    precision, recall = [], []
    for i in range(1,len(cm)):
        p1 = 100*(cm[i,i]/(np.sum(cm[:,i])))
        r1 = 100*(cm[i,i]/(np.sum(cm[i,:])))
        precision.append('{:.2f}'.format(p1))
        recall.append('{:.2f}'.format(r1))     
    val_acc = 100*float(cm.trace())/np.sum(cm)
    IoUs = get_intersection_union_per_class(cm)
    IoU_mean = 100*np.mean(IoUs)
        
    return val_acc, IoU_mean
    
def labels2classes(predictions):
    classes = np.argmax(predictions, axis=1)
    return classes.reshape((-1,1))
    
def calc_metrics(preds, labels):  
    scores = {}

    ### Classification
    classification_labels = ['red_light', 'hazard_stop', 'speed_sign']
    for k in classification_labels:
        cl, pred_cl = labels2classes(labels[k]), labels2classes(preds[k])
        scores[k + '_val_acc'], scores[k + '_IoU'] = calculate_scores(cl, pred_cl)
        
    #### Regression
    regression_labels = ['relative_angle', 'center_distance', 'veh_distance']
    for k in regression_labels: 
        scores[k + '_MAE_mean'] = mae = np.mean(abs(labels[k] - preds[k]))
    return scores
