###
import numpy  as np
import scipy  as sc

###
from flowImg import *

### sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score

### Keras Neural Network
from keras.utils  import to_categorical

### plotting
import matplotlib.colors as colors

#####################################################################################        
def set_label_onevsrest(label, target):
    label = label.copy()
    label[label != target] = 0
    label[label == target] = 1
    return label

#####################################################################################        
def leave_one_out_split(dat, k, n_samples, n_subsamples):
    """divide the data into two parts"""
    ### the samples index after subsampling    
    idx_samples = np.array([[idx] * n_subsamples for idx in range(n_samples)])
    idx_samples = idx_samples.ravel()

    ### get the overall index for train and validation
    idx_trn = np.where(idx_samples != k)[0]
    idx_val = np.where(idx_samples == k)[0]
    
    ### for each label, get the index for train and validation
    idx_trn = np.concatenate([
        dat.which_label(label)[idx_trn] 
        for label in dat.unique_label()
    ])
        
    idx_val = np.concatenate([
        dat.which_label(label)[idx_val] 
        for label in dat.unique_label()
    ])
        
    ### split matrix
    mat_trn = dat.get_matrix_original()[idx_trn]
    mat_val = dat.get_matrix_original()[idx_val]
    
    ### split labels
    lab_trn = dat.label[idx_trn]
    lab_val = dat.label[idx_val]
        
    ### get_coord_original()
    n_sample = dat.get_num_sample()
    coord = np.array([dat.get_coord(idx) for idx in range(n_sample)])
    
    ### split coordinates
    coord_trn = np.concatenate(coord[idx_trn])
    coord_val = np.concatenate(coord[idx_val])
       
    ### split images   
    img_trn = dat.img[idx_trn]
    img_val = dat.img[idx_val]
        
    ### wrap the train and validation data      
    wrap_trn = Data_Wrapper(mat_trn, lab_trn, coord=coord_trn, image=img_trn)
    wrap_val = Data_Wrapper(mat_val, lab_val, coord=coord_val, image=img_val)
    return (wrap_trn, wrap_val)    

#####################################################################################  
def kfold(dat_img, fun_model, dct_kargs, n_samples, n_subsamples, verbose = True):
    """..."""
    ### init
    fprs, tprs, roc_aucs = dict(), dict(), dict()
    n_classes = len(dat_img.unique_label())
    
    ###
    for idx in range(n_samples):
        if (verbose):
            print("kfold: k =", idx)
        dat_img_train, dat_img_test = leave_one_out_split(dat_img, idx, n_samples, n_subsamples)
        
        ##############################
        if (verbose):
            print("\tFit & Predict")
        model      = fun_model(dat_img_train, **dct_kargs)  
        classifier = model.fit(dat_img_train, verbose = 0)
        res_pred   = classifier.predict(dat_img_test)
        
        y_prob = res_pred['predicted_prob']
        y_test = dat_img_test.label
        y_test = to_categorical(y_test)
        ############################## 
    
        ### calculate fpr and tpr
        if (verbose):
            print("\tROC")
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        ### store the results
        fprs[idx]     = fpr
        tprs[idx]     = tpr
        roc_aucs[idx] = roc_auc
        
    return fprs, tprs, roc_aucs

#####################################################################################        
def pool_roc(fprs, tprs, idx_class):
    ###
    base_fpr  = np.linspace(0, 1, 50)
    base_tprs = list()
    
    ###
    for idx_fold in fprs.keys():
        ###
        fpr      = fprs[idx_fold][idx_class]
        tpr      = tprs[idx_fold][idx_class]
        
        ###
        base_tpr = sc.interp(base_fpr, fpr, tpr) 
        base_tpr[0] = 0.0 
        base_tprs.append(base_tpr)

    ###
    base_tprs       = np.array(base_tprs)
    base_tprs_mu    = base_tprs.mean(axis=0)
    base_tprs_sd    = base_tprs.std(axis=0)    
    base_tprs_upper = np.minimum(base_tprs_mu + base_tprs_sd, 1)
    base_tprs_lower = base_tprs_mu - base_tprs_sd
    base_auc        = auc(base_fpr, base_tprs_mu)
    
    ### 
    dct = dict()
    dct["fpr"]        = base_fpr
    dct["tprs_mu"]    = base_tprs_mu
    dct["tprs_upper"] = base_tprs_upper
    dct["tprs_lower"] = base_tprs_lower
    dct["auc"]        = base_auc
    return dct

#####################################################################################        
def train_test_split_datawrapper(dat, n_train, n_subsample):
        """divide the data into two parts"""
        ### index for train and test
        #n_train = np.ceil(n_sample * 0.8).astype(int) # 11 -> 9
        n_train *= n_subsample                        # 9  -> 90
        for label in dat.unique_label():
            idx = dat.which_label(label) 
            assert n_train < len(idx)
        
        ids_train = np.concatenate([
            dat.which_label(label)[:n_train] 
            for label in dat.unique_label()
        ])
        
        ids_test  = np.concatenate([
            dat.which_label(label)[n_train:] 
            for label in dat.unique_label()
        ])
        
        ### split matrix, label, coordinates, image
        mat_train = dat.get_matrix_original()[ids_train]
        mat_test  = dat.get_matrix_original()[ids_test]
        
        lab_train = dat.label[ids_train]
        lab_test  = dat.label[ids_test]
        
        if dat.coord is not None:
            ### get_coord_original()
            n_sample = dat.get_num_sample()
            coord = np.array([dat.get_coord(idx) for idx in range(n_sample)])
            ### split
            coord_train = np.concatenate(coord[ids_train])
            coord_test  = np.concatenate(coord[ids_test])
        else:
            coord_train = None
            coord_test  = None
        
        if dat.img is not None:
            img = dat.img
            img_train = img[ids_train]
            img_test  = img[ids_test]
        else:
            img_train = None
            img_test  = None
        
        wrap_train = Data_Wrapper(mat_train, lab_train, coord=coord_train, image=img_train)
        wrap_test  = Data_Wrapper(mat_test, lab_test, coord=coord_test, image=img_test)
        return (wrap_train, wrap_test)
    
#####################################################################################
### https://matplotlib.org/users/colormapnorms.html
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))