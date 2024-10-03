# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
import pandas as pd
import random
import matplotlib.pyplot as plt
import scipy.interpolate

from tqdm import tqdm
from sklearn import tree
import time
from functools import partial, reduce
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, ExtraTreeClassifier

from sklearn.tree._tree import TREE_LEAF

# true implementation of the honest tree; takes much longer
try:
    from decision_tree import Tree
except:
    pass

import os 

def _prune_tree(model, I, W_I, w_var, min_samples_leaf):
    """
    function that prunes the tree until there are min_samples_leaf samples from eachtreatment class 
    (from the I sample); one of the requirements of the honest tree
    """
    
    # let's build a dataframe with the information of how many samples from each treatm. class are in each leaf
    index_cols = list(I[[]].index.names)
    preds_W = I[[]].reset_index().merge(W_I.reset_index(), on=index_cols, how='left')
    
    
    def prune_index(tree, index, leaves_to_remove):
        '''
        function that removes a leaf and makes its nodes point to the original's parents
        '''
        
        def get_index(x, children):
            if x in children:
                return list(children).index(x)
            else:
                return None

        # parents of the leaves we are about to remove
        parents = ([get_index(x, tree.children_right) for x in leaves_to_remove] + 
                   [get_index(x, tree.children_left) for x in leaves_to_remove])
        
        # filtering the empty ones
        parents = [x for x in parents if x]
        parents_set = list(set(parents))
        
        for parent in parents_set:
            tree.children_left[parent] = TREE_LEAF
            tree.children_right[parent] = TREE_LEAF
            
    def get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children):
        """
        function that calculates which leaves must be pruned
        """
        preds_W['leaf'] = model.apply(I)
        
        # how many samples for each treatm. class
        leaves_to_remove = preds_W.groupby(['leaf'] + [w_var]).size().reset_index()
        
        # pivoting so that we capture the leaves with no samples of one of the classes
        leaves_to_remove = leaves_to_remove.pivot_table(index='leaf', columns=w_var, values=0).fillna(0).reset_index()
        
        # melting to its original form
        leaves_to_remove = leaves_to_remove.melt(id_vars='leaf', value_name='leaf_size')
        
        # finding the incomplete nodes
        leaves_to_remove = leaves_to_remove.query("leaf_size < @min_samples_leaf").leaf.unique().tolist()
        
        # removing the parent nodes
        leaves_to_remove = [x for x in leaves_to_remove if x not in first_children]
        
        return preds_W, leaves_to_remove
    
    
    # getting the first noees - we can't remove them in case they don't satisfy our conditions, the program loops
    first_children = [model.tree_.children_right[0], model.tree_.children_left[0], 0]
    
    preds_W, leaves_to_remove = get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children)
    
    # looping until ready
    while leaves_to_remove:
        # start from the root
        prune_index(model.tree_, 0, leaves_to_remove)
        
        # updating the nodes
        preds_W, leaves_to_remove = get_leaves_to_remove(I, preds_W, model, w_var, min_samples_leaf, first_children)
        
    return model

def _train_honest_tree(df, y_var, w_var, X_var,index_cols, min_samples_leaf,y_cf,y1,y0):
    """
    function that effectively trains each tree in the forest
    """
    
    
    df_sample = df
    s = 0.2
        
    # step 1 : splitting (J = train, I = predictions)
    J, I, tau_J, tau_I, W_J, W_I, CF_J, CF_I, y1_J, y1_I, y0_J, y0_I = train_test_split(
                            df_sample.set_index(index_cols)[X_var],
                            df_sample.set_index(index_cols)[y_var],
                            df_sample.set_index(index_cols)[w_var],
                            df_sample.set_index(index_cols)[y_cf],
                            df_sample.set_index(index_cols)[y1],
                            df_sample.set_index(index_cols)[y0],
                            test_size=s,
                            random_state=2023)
    
    # step 1.5 get true ATE
    ATE_True_0 = 0
    for w_i in W_J.index:
        #print(w_i)
        # print(W_J.index)
        if W_J[w_i] == 1:
            ATE_T_ = tau_J[w_i] - CF_J[w_i]
        else:
            ATE_T_ = CF_J[w_i] - tau_J[w_i]
        ATE_True_0 += ATE_T_
    ATE_True_0 = ATE_True_0/len(W_J)
    
    ATE_True_1 = np.mean(y1_J - y0_J)

    # print(sum(ATE_True))
    
    # step 2 : training the tree

    model = ExtraTreeClassifier(criterion='gini', min_samples_leaf=2*min_samples_leaf, splitter='best',random_state = 2018)
    
    # we use J for training, but this time the target is the treament class variable
    model.fit(J, W_J)
    
    # pruning and prediction in J
    model = _prune_tree(model, I, W_I, w_var, min_samples_leaf)
    X_prediction, tau_prediction, W_prediction = J, tau_J, W_J
        
    
    # creating a dataframe with the predictions by leaf
    leaves = X_prediction[[]].copy()
    leaves['leaf'] = model.apply(X_prediction)
    leaves['true'] = tau_prediction
    leaves[w_var] = W_prediction
    
    leaves_p = leaves.groupby(['leaf'])[w_var].mean().reset_index()
    
    leaves_y = leaves.groupby(['leaf']+[w_var]).true.mean().reset_index()
    leaves_count = leaves.groupby(['leaf']).count().reset_index()
    
    
    n = sum(leaves_count[w_var])
    ATE = 0
    for leaf_ in set(leaves['leaf']):
        C_data = leaves_y.loc[leaves_y['leaf'] == leaf_]
        CATE = C_data.loc[C_data[w_var] == 1,'true'].values - C_data.loc[C_data[w_var] == 0,'true'].values
        p = leaves_count.loc[leaves_count['leaf'] == leaf_,w_var].values / n
        ATE += p * CATE
    
    e_ATE_0 = np.abs(ATE - ATE_True_0)
    e_ATE_1 = np.abs(ATE - ATE_True_1)
    # predicting
    
    this_preds = X_prediction[[]].copy()
    this_preds['leaf'] = model.apply(X_prediction)
    # this_preds.reset_index(inplace=True)
    
    this_preds['ps'] = 0
    
    for i in this_preds.index:
        # print(i)
        this_preds.loc[i,'ps'] = leaves_p.loc[leaves_p['leaf'] == leaves.loc[i,'leaf'],w_var].values
    

    return e_ATE_0,e_ATE_1,this_preds,model,leaves_p,leaves_count

data = pd.read_csv('../data/data-discrete.csv')
data = data.reset_index().rename(columns={'index':'index1'})

X_var_C = [ 'A', 'I', 'M', 'Z', 'TO', 'YO']

X_var = ['C']  + ['Z']
e_ATE_0,e_ATE_1,this_preds,model,leaves_p,leaves_count = _train_honest_tree(df = data, 
                                               y_var = 'Y', 
                                               w_var = 'T', 
                                               X_var = X_var,
                                               index_cols = ['index1'], 
                                               min_samples_leaf = 50,
                                               y_cf = 'Y_cf',
                                               y1 = 'Y_1',
                                               y0 = 'Y_0')
print("============{}============".format('_'.join(X_var)))
print(e_ATE_0)
print(e_ATE_1)
print(len(leaves_p))

data_out = data.loc[this_preds.index,:]
data_out['ps'] = this_preds['ps']
data_out.to_csv('../data/data_out_{}.csv'.format('_'.join(X_var)),index = False)