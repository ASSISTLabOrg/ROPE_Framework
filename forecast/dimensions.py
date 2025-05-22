"""

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#

import numpy as np
import utils
from sklearn.decomposition import PCA as skPCA

#===================================== Principal Component Analysis =====================================#

class PCA:

    def __init__(self, **kwargs):
        __dict__.update(**kwargs)

    def get_singular_vectors(snapshot):
        return None

    def interpolate(self,
                    xn : utils._array_like, 
                    y : np.ndarray,
                    method : str = "knn",
                    k : int = 8,
                    *args) -> float:
        """
        Method for interpolated PCA-space data without needing to do the full xform.

        Arguments:
            xn : Array-like of length(dims) point in configuration space to interpolate to
            y : ndarray of length(components) PCA component amplitude, already at correct time
            k : number of points to retrieve
            args : additional arguments for different interpolation methods.

        Returns:
            Interpolated datum.
        
        TODO: currently only supports KNN. Would like to add trilinear, tricubic, whatever...
        """

        #### query the tree for best indices
        d, indices = self.tree.query(xn, k=k)
        
        #### if requested point is on grid, no interpolation is required
        if d[0] == 0:

            #### inverse PCA transformation in the reduced index space
            Xp = np.squeeze(np.matmul(y.reshape((1,-1)), self.pca.components_[:,indices[0]]))

            #### inverse scaling transformation in the reduced index space
            #### and this is the final result
            X_itp = self.scaler.mean_[indices[0]] + self.scaler.scale_[indices[0]] * Xp

        else:

            #### inverse PCA transformation in the reduced index space
            Xp = np.squeeze(np.matmul(y.reshape((1,-1)), self.pca.components_[:,indices]))

            #### inverse scaling transformation in the reduced index space
            X = self.scaler.mean_[indices] + self.scaler.scale_[indices] * Xp

            #### interpolate based on requested method
            if method.lower()=="knn":
                X_itp = np.sum(X / d) / np.sum(1 / d) # inverse distance weighting
            
            else:
                raise Exception("Only supports the following interpolation methods: [knn]")
            
        return X_itp