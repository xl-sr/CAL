from model_functions import *
import numpy as np
import os
from PIL import Image

# the outputs of the network are normalized to a range of -1 to 1
# therefore we need to multiply it by the normalization constants
# (which where calculated on the training set)
NORM_DICT = {'center_distance': 1.6511945645500001,
             'veh_distance': 50.0,
             'relative_angle': 0.759452569632}
             
# set up for the model rebuild
last_layer_keys = ['0_red_light', '1_hazard_stop', '2_speed_sign',\
                   '3_relative_angle', '4_center_distance', '5_veh_distance']
[RED_LIGHT, HAZARD_STOP, SPEED_SIGN, RELATIVE_ANGLE, CENTER_DISTANCE, VEH_DISTANCE] = last_layer_keys

class ModelSingle(object):
    def __init__(self):
        print("Building model.")
        tup = reload_model_from_episode('full_model_ep_3936')
        self.model = tup[0]   
        self.max_input_shape = self.model.input_shape[0][1]
        
    def predict(self, im, meta):
        preds = self.model.predict([im] + meta, batch_size=1, verbose=0)  
        return preds      
               
class TaskBlockEnsemble(object):
     """
     enhance the predictions of the main model with the prediction
     of the specialized task block models
     """
     def __init__(self):
        print("Building model 1.")
        self.model1 = reload_model_from_episode('full_model_ep_3936')[0]
        self.inp_shape1 = self.model1.input_shape[0][1]
        
        print("Building model 2: red light")
        self.model2 = reload_model_from_episode(RED_LIGHT)[0] 
        self.inp_shape2 = self.model2.input_shape[0][1]
        
        print("Building model 3: hazard stop") 
        self.model3 = reload_model_from_episode(HAZARD_STOP)[0]     
        self.inp_shape3 = self.model3.input_shape[0][1]    
        
        print("Building model 4: relative angle")
        self.model4 = reload_model_from_episode(RELATIVE_ANGLE)[0] 
        self.inp_shape4 = self.model4.input_shape[0][1]
        
        # get input shape
        self.max_input_shape = self.inp_shape1
        
     def predict(self, im, meta):
        preds1 = self.model1.predict([im] + meta, batch_size=1, verbose=0)
        # the sequence is 14 frames long (= max input shape)
        # to use the task block models, the input sequence needs to be adjusted accordingly
        preds2 = self.model2.predict([im[:,(14 - self.inp_shape2):,:,:,:]] + meta, batch_size=1, verbose=0)
        preds3 = self.model3.predict([im[:,(14 - self.inp_shape3):,:,:,:]] + meta, batch_size=1, verbose=0)
        preds4 = self.model4.predict([im[:,(14 - self.inp_shape4):,:,:,:]] + meta, batch_size=1, verbose=0)

        preds = []
        # average the prediction of all 6 affordances
        # speed sign and veh_distance do not need an additional 
        # prediction -> reduces the comp. overhead
        preds.append((preds1[0] + preds2[0])/2)
        preds.append((preds1[1] + preds3[1])/2)
        preds.append(preds1[2])
        preds.append((preds1[3] + preds4[3])/2)
        preds.append((preds1[4] + preds4[4])/2)    
        preds.append(preds1[5])

        return preds

class CAL_network(object):
    def __init__(self, ensemble=False):
        front_model, _, preprocessing = get_conv_model()
        self.conv_model = front_model
        self.preprocessing = preprocessing     
        if ensemble:   
            self.model = TaskBlockEnsemble()
        else:
            self.model = ModelSingle()        

    def predict(self, im, meta):
        """ 
        input: transformed image, meta input(i.e. direction) as a list
        Returns the predictions dictionary
        """
        if not isinstance(meta, list):
            raise TypeError('Meta needs to be a list')

        prediction = {}
        meta = [np.array([meta[i]]) for i in range(len(meta))]

        # predict the classes and the probabilites of the validation set
        preds = self.model.predict(im, meta)

        # CLASSIFICATION
        classification_labels = ['red_light', 'hazard_stop', 'speed_sign']
        classes = [[False, True],[False, True],[-1, 30, 60, 90]]
        for i, k in enumerate(classification_labels):
            prediction[k] = self.labels2classes(preds[i], classes[i])

        # REGRESSION
        regression_labels = ['relative_angle', 'center_distance', 'veh_distance']
        for i, k in enumerate(regression_labels):
            prediction[k] = preds[i+len(classification_labels)][0][0]
            prediction[k] = np.clip(prediction[k],-1,1)
            # undo normalization
            prediction[k] = prediction[k]*NORM_DICT[k]

        return prediction

    def preprocess_image(self, im, sequence=False):
        im = Image.fromarray(im)
        im = im.crop((0,120,800,480)).resize((222,100))
        
        # reshape and resize the image
        x = np.asarray(im, dtype=K.floatx())
        x = np.expand_dims(x,0)
        # preprocess image
        x = self.preprocessing(x)
        x = self.conv_model.predict(x, batch_size=1, verbose=0)

        if sequence: x = np.expand_dims(x,1)
        return x

    def labels2classes(self, prediction, c):
        # turns predicted probs to onehot idcs predictions
        # returns a tuple oft the predicted class and its probability
        # == predict classes
        max_idx = np.argmax(prediction)
        predicted_class = c[max_idx]
        predicted_proba = np.max(prediction)
        return (predicted_class, predicted_proba)
