import numpy as np
import pandas as pd

def statistics_match(df1, df2):
    matches = []
    bounds = {
        'red_light': {'type': 'discrete', 'eps': 0.005,},
        'hazard_stop': {'type': 'discrete', 'eps': 0.005,},
        'speed_sign': {'type': 'discrete', 'eps': 0.005,},
        'center_distance': {'type': 'cont', 'eps': 0.02,},
        'relative_angle': {'type': 'cont', 'eps': 0.01,},
        'veh_distance': {'type': 'cont', 'eps': 0.6,},  
    }
    
    def calc_rel_count(x):
        _, counts = np.unique(x, return_counts=True)
        return counts/sum(counts)
    
    def calc_mean(x):
        return np.array([np.mean(x)])
    
    def cmp(key):
        ### for the discrete variable we compare the rel count, for cont we compare mean 
        fun = calc_rel_count if bounds[key]['type']=='discrete' else calc_mean
        if not all(abs(fun(df1[key]) - fun(df2[key])) < bounds[key]['eps']):
            print(f"too different in {key}")
            return False
        else:
            return True
              
    for k in bounds.keys(): matches.append(cmp(k))    
    return all(matches)
    
def get_val_idcs(data_path, split=0.1):
    """
    split the dataset into train and validation set
    two things to do for good validation split
     - we have sequential data, so random splitting is bad, since
       we will have very similar frames in the train/valid sets
     - each episode is recorded from three different positions,
       so when splitting, this recording need to stay together
       
    input: 
     - data_path
     - split: how large the validation should be, percentage wise
    """
    df = pd.read_csv(data_path + 'annotations.csv')
    
    # get the episode names per image file
    ep_names_df = [im[:16] for im in df['im_name']]
    # get the unique episode names
    ep_names = np.unique(ep_names_df)
        
    def get_val_idcs():
        # get random episode indices and split into train and val set
        shuffled_idcs = np.random.permutation(np.arange(len(ep_names)))
        n_ep_val = int(len(ep_names)*split)
        return shuffled_idcs[:n_ep_val]
    
    # run the split until the statistics of both valid and train set match
    ID = 0
    while True:
        print(ID)
        val_idcs = get_val_idcs()
        is_val = np.array([ep in ep_names[val_idcs] for ep in ep_names_df])
        
        # compare the mean and average for regression va
        match = statistics_match(df[~is_val], df[is_val])
        
        ID +=1
        if match: break
    
    return is_val
