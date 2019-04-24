### Basic tools
import numpy as np
from   numpy import random
from   numba import cuda, vectorize
from   collections import Counter
import copy
import random
import math

### Machine learning
from MulticoreTSNE           import MulticoreTSNE as TSNE
from sklearn.linear_model    import LogisticRegression
from sklearn.decomposition   import PCA
from sklearn.datasets        import make_blobs
from sklearn.pipeline        import Pipeline, FeatureUnion
from sklearn.base            import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics         import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.multiclass      import OneVsRestClassifier

### Keras Neural Network
from keras.utils  import to_categorical

### plotting
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D

### helper function
head = lambda x, n = 6: x[:n]
tail = lambda x, n = 6: x[-n:]



#########################################################################
class Data_Wrapper:
    """ A wrapper with data matrix (N, M_i, p) and data labels
    N   = number of samples
    M_i = number of observations / data points in the ith sample
    p   = number of variables for all sample
    
    Args:
        matrix (N, M_i, p):      data values
        label  (N,):             labels
        coord  (Sum(M_i), 2):    coordination after dim. reduct. data values
        img    (N, ngrid, ngrid):2D distribution from kernel smoothing of dim. reduct. 
    """
    def __init__(self, matrix, label, coord = None, image = None):
        ### convert data type
        matrix = np.array(matrix)
        label  = np.array(label)
        index  = self.create_index(matrix)
        
        ### check input: 
        ###     assure N is the same in matrix (N, M_i, p) and label (N,)
        assert matrix.shape[0] == label.shape[0]
        
        ### assign class fields
        self.label  = label
        self.index  = index
        self.matrix = np.vstack(matrix)
        self.coord  = coord
        self.img    = image
        self.ngrid  = None
        
        ### check input
        self.check_index(index)
        self.check_coord(coord)
        self.check_image(image)
        
    def __str__(self):
        out  = "===================================\n"
        out += ("Label: " + str(self.count_label()) + "\n")
        for label in self.unique_label():
            out += (" " * 4 + str(label) + ": " + str(self.which_label(label)) + "\n")
        
        out += "------------------\n"
        out += ("Data Matrix: "    + str(self.matrix.shape)       + "\n")
        out += ("    #Samples:   " + str(self.get_num_sample())   + "\n")
        out += ("    #Variables: " + str(self.get_num_variable()) + "\n")

        out += "------------------\n"
        out += "Coordinate: "
        out += (str(None) if self.coord is None else str(self.coord.shape))
        
        out += "\n------------------\n"
        out += "Image: "
        out += (str(None) if self.img is None else str(self.img.shape))
        out += "\n===================================\n"
        return out        
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Data_Wrapper):
            
            ### check label
            tmp1, tmp2 = self.label, other.label
            if not np.all(tmp1 == tmp2):
                return False
            
            ### check index
            tmp1, tmp2 = self.index, other.index
            if not np.all(tmp1 == tmp2):
                return False
            
            ### check matrix
            tmp1, tmp2 = self.matrix, other.matrix
            if not np.allclose(tmp1, tmp2):
                return False
            
            ### check coord
            ### not implemented yet
            
            ### check image
            ### not implemented yet
            
        return True
    
    def check_index(self, index):
        """Check if the index are has the correct dimension"""
        assert index.shape[0] == (self.label.shape[0] + 1)
        
    def check_coord(self, coord):
        """if exist, assure coord has the same observations with matrix (data values)"""
        if coord is not None:
            coord = np.array(coord)
            assert coord.shape[0] == self.matrix.shape[0]
        
    def check_image(self, img):
        """if exist, assure the number of images is the same with labels"""
        if img is not None:
            #img = np.array(coord)
            #ngrid = img.shape[1]
            #assert matrix.shape[0] == img.shape[0]
            assert img.shape[0] == self.label.shape[0]
    
    def create_index(self, matrix):
        """index indicate matrices from different samples"""
        idx = np.array([x.shape[0] for x in matrix])
        idx = np.r_[0, idx]
        idx = np.cumsum(idx)
        return idx
    
    def get_num_variable(self):
        """get the number of variables / markers / features"""
        return self.matrix.shape[1]    
    
    def get_num_sample(self):
        """get the total number of samples"""
        return len(self.label)
    
    def count_label(self):
        """count each label"""
        return Counter(self.label)
    
    def unique_label(self):
        """return the unique labels"""
        return np.unique(self.label)
    
    def which_label(self, label):
        """which sample contains the specify label"""
        return np.where(self.label == label)[0]

    def get_coord(self, k):
        """get the coordinate of sample specified"""
        if self.coord is None:
            return None
        return self.coord[self.index[k] : self.index[k + 1]]
    
    def get_sample(self, k):
        """get the matrix of sample specified"""
        return self.matrix[self.index[k] : self.index[k + 1]] 
    
    def get_sample_size(self, k):
        """get the size of sample specified"""
        return self.index[k + 1] - self.index[k] 
    
    def get_matrix_original(self):
        """get back the matrix with shape (N_SAMPLE, N_ROWS, N_COLS)"""
        n_sample = self.get_num_sample()
        mat = np.array([self.get_sample(idx) for idx in range(n_sample)])
        return mat
    
    def get_label(self, k):
        """get the label of sample specified"""
        return self.label[k]
    
    def get_index(self, label):
        """get the index of sample based on given label"""
        index = [idx for idx, lab in enumerate(self.label) if lab == label]
        return index
    
    def get_image(self, k):
        """get the image of sample"""
        if self.img is None:
            return None
        return self.img[k]
    
    def get_coord_grid(self, k, ngrid):
        """create grid from coordination of k samples"""
        ### Coordinate of k_th samples
        coord  = self.get_coord(k)
        x = coord[:, 0]
        y = coord[:, 1]
        
        ### set coordinate of grids
        xc = np.linspace(min(x), max(x), ngrid)
        yc = np.linspace(min(y), max(y), ngrid)
        xc, yc = np.meshgrid(xc, yc)
        
        ### the return grids coordinate with shape (ngrid x ngrid, 2)
        grids = np.array([xc.ravel(), yc.ravel()]).T
        assert grids.shape == (ngrid**2, 2)
        return grids
    
    def set_matrix(self, matrix):
        """set the matrix of data"""
        assert self.matrix.shape == matrix.shape
        self.matrix = matrix
    
    def set_image(self, image):
        """set the image of data"""
        # Check to make sure coord has the same number of samples
        assert self.label.shape[0]  == image.shape[0]
        assert self.matrix.shape[1] == image.shape[2]
        self.ngrid = int(image.shape[1]**0.5)
        self.img = image
    
    def set_coord(self, coord):
        """set the coordinate of data"""
        # Check to make sure coord has the same number of samples
        self.check_coord(coord)
        self.coord = coord
        
    def plot_sample(self, ax, k = 0, idx_x = 0, idx_y = 1, **kwargs):
        """plot the data matrix of a given sample, x variable and y variable"""
        matrix = self.get_sample(k)
        label  = self.get_label(k)
        x = matrix[:, idx_x]
        y = matrix[:, idx_y]
        ax.scatter(x, y, **kwargs)
        ax.set_title("Sample: " + str(k) + " | " + "Label: " + str(label))
        return ax
    
    def plot_coord(self, ax, k = 0, p = 0, **kwargs):
        """plot the coordinate of a given sample"""
        ### get sample matrix, coordinates, and label
        label  = self.get_label(k)  # for title
        matrix = self.get_sample(k) # original data
        coord  = self.get_coord(k)  # coordinates (ex: dimension reduction)
        
        ### set the scatter plot
        x =  coord[:, 0]
        y =  coord[:, 1]
        z = matrix[:, p]
        
        ### plot scatter plot and set title
        cax = ax.scatter(x, y, c = z, **kwargs)
        title  = ("Sample: "  + str(k)     + " | ")
        title += ("Label: "   + str(label) + " | ")
        title += ("Feature: " + str(p))
        ax.set_title(title)
        return cax
    
    def plot_img(self, ax, k = 0, p = 0, **kwargs):
        """plot the coordinate of a given sample"""
        ### get sample matrix, coordinates, and label
        label = self.get_label(k) # for title
        grids = self.get_coord_grid(k, self.ngrid)
        image = self.get_image(k) # get image 
        zc    = image[:, p]       # 1 dim. vector; length = ngrid x ngrid
        
        ### plot the image using scatter plot
        cax = ax.scatter(x = grids[:, 0], y = grids[:, 1], c = zc, **kwargs)
        title  = ("Sample: "  + str(k)     + " | ")
        title += ("Label: "   + str(label) + " | ")
        title += ("Feature: " + str(p))
        ax.set_title(title)
        return cax
    
    
#########################################################################
class Transform_StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize all the data based on the k sample
    Args:
        k (int) the k_th sample that the standardization of each sample is based on
    """
    def __init__(self, k = 0):
        """initialization"""
        self.mu  = 0
        self.std = 1
        self.k   = k
        
    def fit(self, dat, y = None, verbose = True):
        """Get the mean and std of the k_th sample"""
        x_selected = dat.get_sample(self.k)
        self.mu  = x_selected.mean(axis = 0)
        self.std = x_selected.std( axis = 0)
        
        if verbose:
            out =  ""
            out += "===================================\n"
            n_sample = dat.get_num_sample()
        
            out += "Transform_StandardScaler(k=" + str(self.k) + ")\n"
            out += ("Total Sample: " + str(n_sample) + "\n")
            out += ("Standardized based on sample: " + str(self.k) + "\n\n")
        
            out += ("***** k_th Sample: " + str(self.k) + " *****\n")
            out += "Mean:\n"
            out += str(self.mu) + "\n"
            out += "SD:\n"
            out += str(self.std) + "\n"
            out += "===================================\n"
            print(out)
            
        return self

    def transform(self, dat):
        """Standardize all the data based on the k sample"""
        dat_copy = copy.deepcopy(dat)
        dat_copy.set_matrix((dat.matrix - self.mu) / self.std)
        return dat_copy
    
#########################################################################
class Transform_Subsampling(BaseEstimator, TransformerMixin):
    def __init__(self, n_subsample = 10, n_size = 1000, random_state = 0):
        ### assign class fields
        self.indices = None
        self.n_subsample  = n_subsample
        self.n_size       = n_size
        self.random_state = random_state
        
    def fit(self, dat, y = None):
        return self

    def transform(self, dat):
        ### initialization
        lst_label  = [] # list of labels
        lst_matrix = [] # matrix of each sample
        
        # coordinates if exist
        if dat.get_coord(0) is not None:
            lst_coord  = []
        else:
            lst_coord  = None

        # image if exist
        # !!! not implemented yet
        
        ### for each sample, create subsamples
        np.random.seed(self.random_state)
        for idx_sample in range(dat.get_num_sample()):
            ### get each sample
            label  = dat.get_label(idx_sample)
            matrix = dat.get_sample(idx_sample)
            coord  = dat.get_coord(idx_sample)
            
            ### create subsamples
            for _ in range(self.n_subsample):
                # randomly select rows with replacement
                idx = np.random.randint(matrix.shape[0], size = self.n_size)
                
                # append the label
                # should be the same as original sample
                lst_label.append(label) # the label 
                
                # append the matrix
                # append the subsample selected from original sample
                lst_matrix.append(matrix[idx])  
                
                # append the coordinate if exist
                if coord is not None:
                    lst_coord.append(coord[idx])
                    
        ### Wrapping new data and check the result's dimension
        dat_new = Data_Wrapper(lst_matrix, lst_label, lst_coord)
        assert dat_new.get_num_sample() == (dat.get_num_sample() * self.n_subsample)
        assert dat_new.matrix.shape[0]  == (dat.get_num_sample() * self.n_subsample * self.n_size)
        return dat_new

#####################################################################################
class Transform_PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components = 2):
        self.k = n_components
        self.pca = PCA(n_components = self.k)
    
    def fit(self, dat, y = None):         
        return self

    def transform(self, dat):
        ### initialization
        dat_copy = copy.deepcopy(dat)
        
        ### transform each sample
        mat = [self.pca.fit_transform(dat.get_sample(idx)) for idx in range(dat.get_num_sample())]
        mat = np.array(mat)
        mat = np.vstack(mat)
        
        ### set coordinate
        dat_copy.set_coord(mat)
        return dat_copy
    
class Transform_TSNE(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_jobs=8, random_state=123):
        self.k    = n_components
        self.tsne = TSNE(n_jobs = n_jobs, random_state = random_state)
        self.random_state = random_state
        
    def fit(self, dat, y = None):         
        return self

    def transform(self, dat, verbose = True):
        dat_copy = copy.deepcopy(dat)
        
        ### transform each sample
        np.random.seed(self.random_state)
        mat = []
        for idx in range(dat.get_num_sample()):
            if verbose:
                print("Sample: " + str(idx))
            mat.append(self.tsne.fit_transform(dat.get_sample(idx)))
        
        ### set coordinate
        mat = np.array(mat)
        mat = np.vstack(mat)
        dat_copy.set_coord(mat)
        return dat_copy

#####################################################################################
### define a point type
point_dtype = np.dtype({
    'names':   ['x', 'y'],
    'formats': [np.float32, np.float32]})

### calculate squared distance
@cuda.jit(device = True)
def dist2_kernel(point1, point2):
    return math.pow(point1["x"] - point2["x"], 2) + math.pow(point1["y"] - point2["y"], 2)

### convert squared distance into exponential
@cuda.jit(device = True)
def dist2weight_kernel(dist2, sig2 = 1):
    #if (kernel == "standard normal"):
    return math.exp(-1 * dist2 / sig2)

### wrap function of distance and weight function
### signature: point_dtype[:], point_dtype[:], float[:, :]
@cuda.jit
def get_weights(grids, points, weights, sig2 = 1):
    # set index of cuda
    idx_grid, idx_point = cuda.grid(2)
    if (idx_grid < weights.shape[0]) & (idx_point < weights.shape[1]):
        # intialization
        grid  = grids[idx_grid]
        point = points[idx_point]
        
        # calculate distance and then convert to weights
        dist2 = dist2_kernel(grid, point)
        weights[idx_grid, idx_point] = dist2weight_kernel(dist2, sig2)
        
### helper function: convert coordinate into point
def set_points(dat_pts):
    """Convert data frame of coordinates to an array of points
    
    >>> tmp = np.arange(10).reshape(-1, 2)
    >>> set_points(tmp)
    ... array([(0., 1.), (2., 3.), (4., 5.), (6., 7.), (8., 9.)],
      dtype=[('x', '<f8'), ('y', '<f8')])
    >>> set_points(tmp)[0]['x'], set_points(tmp)[0]['y']
    ... (0.0, 1.0)
    """
    ### initialization
    num_pts = dat_pts.shape[0]
    points = np.empty((num_pts,), dtype = point_dtype)
    
    ### convert each row into a point
    for idx in range(num_pts):
        p = points[idx]
        p['x'], p['y'] = dat_pts[idx][0], dat_pts[idx][1]
    
    return points

#####################################################################################
class Transform_Density(BaseEstimator, TransformerMixin):
    """Class: Transform_Densty
    Args:
    Fields:
    Methods:
    """
    def __init__(self, sig2 = 1, ngrid = 128):
        self.ngrid = ngrid
        self.point_dtype = np.dtype({'names':   ['x', 'y'], 'formats': [np.float32, np.float32]})
        self.sig2 = sig2
        
    def fit(self, dat, y = None):
        return self

    def transform(self, dat, threadsperblock = (32, 32), verbose=True):
        # initialization
        lst_img = []
        dat_copy = copy.deepcopy(dat)
        
        # for each sample, create an image, which compose a set of frame
        for idx_sample in range(dat.get_num_sample()):
            if verbose:
                print("Sample: " + str(idx_sample))
                
            # Sample data (vector for each feature)
            matrix = dat.get_sample(idx_sample)
            
            # Coordinate from dimensional reduction
            coord  = dat.get_coord(idx_sample)
            points = set_points(coord)
            
            # generate a grid of a frame
            grids = dat.get_coord_grid(idx_sample, self.ngrid)
            grids = set_points(grids)
            
            # cuda memory blocks
            blockspergrid_x = math.ceil(grids.shape[0]  / threadsperblock[0])
            blockspergrid_y = math.ceil(points.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            # calculate the weights for smoothing
            weights = np.empty(
                shape = (grids.shape[0], points.shape[0]), 
                dtype = np.float32)
            get_weights[blockspergrid, threadsperblock](grids, points, weights, self.sig2)
            ################################
            
            # for each feature, generate each frame and store in an image
            img = []
            for idx_feature in range(dat.get_num_variable()):
        
                # interpolation
                z = matrix[:, idx_feature]
                zc = np.matmul(weights, z)
        
                # store into a list "img"
                img.append(zc)
        
            # collect the image of each sample
            img = np.array(img)
            lst_img.append(img.T)
        
        lst_img  = np.array(lst_img)
        dat_copy.set_image(lst_img)
        return dat_copy
        
#####################################################################################
class Classify_PCA_LogReg(BaseEstimator, ClassifierMixin):
    """Class: Classify_PCA_LogReg
    Args:
    Fields:
    Methods:
    """
    def __init__(self, dat, classifier=None, pca_explained_ratio = 0.9, random_state = 0):
        ### check dimension
        assert(dat.img.shape[0] == dat.label.shape[0])
        self.n_sample = dat.img.shape[0]
        
        ### PCA reduce the dimension of X
        ### If 0 < n_components < 1 and svd_solver == 'full', 
        ### select the number of components such that 
        ### the amount of variance that needs to be explained is 
        ### greater than the percentage specified by n_components.
        assert(pca_explained_ratio <= 1.0)
        self.pca = PCA(n_components = pca_explained_ratio, svd_solver = "full")
        
        ### declare the model
        if classifier is None:
            self.model = OneVsRestClassifier(LogisticRegression(random_state=random_state))
            #OneVsOneClassifier(LogisticRegression(random_state=0))
        else:
            self.model = classifier 
        
        
    def fit(self, dat, y = None, verbose = 1):         
        ### label y
        if y is None:
            self.y = dat.label
        
        ### data matrix X
        self.X = dat.img.reshape((self.n_sample, -1))
        
        ### reduce dimension using PCA and then fit the logistic regression model
        self.pca   = self.pca.fit(self.X)
        self.X_red = self.pca.transform(self.X)
        self.model.fit(self.X_red, self.y)
        
        ### if verbose: transform each sample and calculate accuracy
        if (verbose):
            num    = self.pca.explained_variance_ratio_
            y_pred = self.model.predict(self.X_red)
            acc    = accuracy_score(self.y, y_pred)       
            print("PCA explained ratio:   ", np.round(np.sum(num), 3))
            print("PCA num components:    ", self.pca.n_components_)
            print("Original    data shape:", self.X.shape)
            print("PCA reduced data shape:", self.X_red.shape)
            print("Log.Reg. Accuracy:     ", acc)
        
        return self

    def get_param(self):
        parameters = model.coef_
        return parameteres
    
    def predict(self, dat_new, y=None):
        ### check the dimension
        assert(dat_new.img.shape[0] == dat_new.label.shape[0])
        n_sample_new = dat_new.img.shape[0]
        
        ### label y
        if y is None:
            y_new = dat_new.label
        else:
            y_new = y
        
        ### data matrix X
        X_new = dat_new.img.reshape((n_sample_new, -1))
        assert(X_new.shape[1] == self.X.shape[1])
        
        ### reduce dimension using PCA and then fit the logistic regression model
        X_new_red = self.pca.transform(X_new)
        
        ### predict class and calcualte accuracy
        y_new_pred  = self.model.predict(X_new_red)
        y_new_prob  = self.model.predict_proba(X_new_red)
        y_new_score = self.model.decision_function(X_new_red)
        acc = accuracy_score(y_new, y_new_pred)
        
        ### output results
        return {"acc":             acc, 
                "predicted_class": y_new_pred,
                "predicted_prob":  y_new_prob,
                "predicted_score": y_new_score}

#####################################################################################
class Classify_CNN(BaseEstimator, ClassifierMixin):
    """comment"""
    
    def __init__(self, dat, fun_get_cnn, learn_rate = 5e-6):
        ### initialization
        img       = dat.img
        n_grid    = dat.ngrid
        n_sample  = dat.get_num_sample()
        n_feature = dat.get_num_variable()
        n_class   = len(dat.unique_label())
        
        ### check dimension
        input_shape = (n_grid, n_grid, n_feature)
        assert(img.shape[0] == n_sample)
        assert(img.shape[1] == n_grid**2)
        assert(img.shape[2] == n_feature)
        
        ### store the info
        self.input_shape = input_shape
        self.cnn_model = fun_get_cnn(input_shape, n_class, lr = learn_rate)
        
    def fit(self, dat_trn, y = None, dat_tst = None, epochs = 20, batch_size = 4, verbose = 1, random_state = 123):         
        ### initialization
        model       = self.cnn_model
        input_shape = self.input_shape
        ngrid       = input_shape[0]
        n_feature   = input_shape[-1]
        
        ### train
        x_trn = dat_trn.img.copy()
        x_trn = x_trn.reshape(-1, ngrid, ngrid, n_feature)
        y_trn = dat_trn.label.copy()
        y_trn = to_categorical(y_trn)
        
        ### test
        if dat_tst is not None:
            x_tst = dat_tst.img.copy()
            x_tst = x_tst.reshape(-1, ngrid, ngrid, n_feature)
            
            y_tst = dat_tst.label.copy()
            y_tst = to_categorical(y_tst)
        
        ### fit the model
        if dat_tst is not None:
            self.history = self.cnn_model.fit(
                x_trn, y_trn,
                batch_size = batch_size,
                epochs     = epochs,
                verbose    = verbose,
                validation_data = (x_tst, y_tst))
        else:
            self.history = self.model.fit(
                x_trn, y_trn,
                batch_size = batch_size,
                epochs     = epochs,
                verbose    = verbose)
        
        return self

    def predict(self, dat_new, y=None):
        ### check the dimension
        input_shape = self.input_shape
        ngrid       = input_shape[0]
        n_feature   = input_shape[-1]
        
        ### data X
        x_new = dat_new.img.copy()
        x_new = x_new.reshape(-1, ngrid, ngrid, n_feature)
            
        ### label y
        if y is None:
            y_new = dat_new.label.copy()
            y_new = to_categorical(y_new)
        else:
            y_new = y
        
        ### predict class and calcualte accuracy
        y_new_pred  = self.model.predict(X_new_red)
        y_new_prob  = self.model.predict_proba(X_new_red)
        y_new_score = self.model.decision_function(X_new_red)
        acc = accuracy_score(y_new, y_new_pred)
        
        ### output results
        return {"acc":             acc, 
                "predicted_class": y_new_pred,
                "predicted_prob":  y_new_prob,
                "predicted_score": y_new_score}
    
