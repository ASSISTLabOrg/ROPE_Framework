"""
Contains transformation objects for reducing and expanding state space.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
import pickle as pkl
from typing import Union
import utils
from sklearn.decomposition import PCA as skdPCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

#===================================== Principal Component Analysis =====================================#

def transformer_factory(config):
    """
    Returns the correct transformer type from the ini file.

    """

    #### PCA loads direct from pickle file
    if config["transformer"]["type"].lower() == "pca":
    
        with open(config["transformer"]["file"], "rb") as f:
            transformer = pkl.load(f)

    #### COAE built from weight file
    # elif config["type"].lower() == "coae":
    #     transformer = COAE(**vars)
    
    else:
        raise Exception("Transformer type not supported. Currently supports: [PCA, ]")
    
    return transformer

#===================================== Principal Component Analysis =====================================#

class PCA(skdPCA):

    def __init__(self, **kwargs):

        super().__init__(n_components=kwargs["n_components"])
        self.__dict__.update(**kwargs)
        
        self.PhysicsGrid = utils.PhysicsGrid(
            kwargs["physics_grid"]["model"],
            kwargs["physics_grid"]["ndim"],
            kwargs["physics_grid"]["dims"]
        )
        
        self.tree = KDTree(
            np.column_stack(
                self.PhysicsGrid.dims
            )
        )

        self.scaler = StandardScaler(
            
        )

        if not self._init_check():
            raise Exception("Model initialized incorrectly.")
        
    def _init_check(self):
        return True

    def reduce(self):
        pass

    def expand(self,
               X : np.ndarray,
               Y : np.ndarray,
               full=False):

        if not full:

            return self.interpolate(
                X, Y
            )
        
        else:

            ### inverse PCA transformation 
            Xp = np.squeeze(
                np.matmul(
                    Y.reshape((1,-1)), 
                    self.components_
                )
            )

            #### inverse scaling transformation
            return self.scaler.mean_ + self.scaler.scale_ * Xp

    def interpolate(self,
                    xn : utils.ArrayLikeType, 
                    y : np.ndarray,
                    method : str = "knn",
                    k : int = 8,
                    *args) -> float:
        """
        Method for interpolated PCA-space data without needing to do the full xform.

        Arguments:
            xn   : [array-like, 1D] point in configuration space to interpolate to
            y    : [array-like 1D] PCA component amplitude, already at correct time
            k    : [int] number of points to retrieve
            args : additional arguments for different interpolation methods

        Returns:
            X_itp : [float] interpolated datum
        
        TODO: currently only supports KNN. Would like to add trilinear, tricubic, whatever...
        """

        #### query the tree for best indices
        d, indices = self.tree.query(
            xn, 
            k=k
        )
        
        #### if requested point is on grid, no interpolation is required
        if d[0] == 0:

            #### inverse PCA transformation in the reduced index space
            Xp = np.squeeze(
                np.matmul(
                    y.reshape((1,-1)), 
                    self.components_[:,indices[0]]
                )
            )

            #### inverse scaling transformation in the reduced index space
            #### and this is the final result
            X_itp = self.scaler.mean_[indices[0]] + self.scaler.scale_[indices[0]] * Xp

        else:

            #### inverse PCA transformation in the reduced index space
            Xp = np.squeeze(
                np.matmul(
                    y.reshape((1,-1)), 
                    self.components_[:,indices]
                )
            )

            #### inverse scaling transformation in the reduced index space
            X = self.scaler.mean_[indices] + self.scaler.scale_[indices] * Xp

            #### interpolate based on requested method
            if method.lower()=="knn":
                X_itp = np.sum(X / d) / np.sum(1 / d) # inverse distance weighting
            
            else:
                raise Exception("Only supports the following interpolation methods: [knn]")
            
        return X_itp
    
#===================================== Autoencoder =====================================#

class COAE:

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

#===================================== Convenience Functions & Typing =====================================#

TransformerType = Union[PCA, COAE]