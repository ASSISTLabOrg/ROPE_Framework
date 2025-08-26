"""
Contains reduced-order models for forecasting space weather.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#
import os
import sys
import numpy as np
import pickle as pkl
import fnmatch
from typing import Union
from pyrope import utils
from datetime import date
import h5py

#### add parent to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#===================================== Factory Methods =====================================#

def model_factory(config : dict):
    """
    Returns the correct model type from the ini file.

    Arguments:
        config : [dict] contains the configuration data for the model
    """

    #### load SINDYc model
    if config["model"]["type"].lower() == "sindyc":

        #### unpickle
        try:
            with open(config["model"]["file"], "rb") as f:
                model = pkl.load(f)
        except:
            pass
        else:
            return model

        #### load raw object from hdf5 file
        try:
            model = SINDYc(
                **utils.hdf5_to_dict(config["model"]["file"])
                )
        except:
            raise Exception("Model file could not be loaded.")
        else:
            return model
    
    # elif config["type"].lower() == "lstm":
    #     return LSTM(**vars)
    
    else:
        raise Exception("Model type not supported. Currently supports: [SINDYc, ]")
    
    return model

#===================================== SINDYc =====================================#

class SINDYc:
    """
    Attributes :
        Xi : coefficient matrices. Currently expects list/tuple containining (Xi_1, Xi_2)
        feature_library : library of model functions
        feature_names : names of functions; purely for inspection
        interp_method : how to interpolate matrices

    Methods :
        forecast : accepts a Task and forecasts the result in reduced-space from that task
        init_check : verfies the model was initialized correctly.
        
    """

    def __init__(
            self, 
            **kwargs):

        self.__dict__.update(**kwargs)
        
        #### make sure all the necessary variables are in place
        if not self.init_check():
            raise Exception("SINDYc model not properly inialized, check inputs.")

    def forecast(self,
                 task : utils.Task) -> np.ndarray:

        #### interpolate drivers to half-step times
        drivers_itp = self.interpolate_drivers(
            task.parameters["solver_times_half"], 
            task.driver_times, 
            task.drivers,
        )

        #### set up RKO4 solver and solution container
        fcst = np.zeros(
            (
                len(task.parameters["solver_times_full"]),
                task.initial_state.size
            )
        )
        fcst[0,:] = task.initial_state
        
        #### Forecast!
        for i in range(1, len(task.parameters["solver_times_full"])):

            Xi = self.get_coeffs(
                task.drivers[i,task.parameters["kp_index"]]
            )

            fcst[i,:] = self.RKO4_step(
                self.ODE_function, 
                fcst[i-1,:].reshape((-1,1)),
                task.parameters["solver_times_full"][i],
                task.parameters["solver_timestep"],
                *[task.parameters["solver_times_half"], 
                  drivers_itp, 
                  Xi
                  ]
            ).squeeze()
            
        output = utils.interpolate_matrix(
            task.Trajectory.time, 
            task.parameters["solver_times_full"],
            fcst, 
            self.interp_method, 
            axis=0
        )

        return output

    def init_check(self):
        """
        Checks all expected variables in the SINDYc model setup.
        """
        loaded_attrs = [key for key in __dict__.keys()]
        check_attrs = ["Xi", 
                       "feature_library", 
                       "feature_names", 
                       "interp_method"]

        check = True
        for key in check_attrs:
            if key not in loaded_attrs:
                check = False
                break

        return check

    def save_model(self, 
                   pickle=True, 
                   file=None):
        """
        Pickles the model for re-opening.

        Arguments:
            file : relative path of file from ROPE_Framework [str]; if None, autofills as SINDYc_[year]_[month]_[day].pkl

        """
        
        if file is None:
            ext = "pkl" * pickle + "h5" * (-1 * pickle)
            file = os.path.join(
                "data", 
                f"SINDYc_{date.today().strftime("%Y_%m_%d")}.{ext}"
            )

        if pickle:
            with open(file, "wb") as f:
                pkl.dump(
                    self, 
                    f, 
                    protocol=pkl.HIGHEST_PROTOCOL
                )
        else:
            utils.dict_to_hdf5(self.__dict__, file)

    def ODE_function(self,
                     t : float,
                     y : np.ndarray,
                     t_rkh : utils.ArrayLikeType,
                     drivers_itp : np.ndarray,
                     Xi : np.ndarray) -> np.ndarray:
        """
        Method for the SINDY model system of equations, passsed to Runge-Kutta pusher.

        Arguments:
            t : time point
            y : Array of shape [N,1] containing the current state of the solution y (initial condition)
            t_rkh : half-step time axis, used to get interpolated drivers
            drivers_itp : interpolated drivers, half-step resolution
            Xi : coefficient matrix for the time step

        Returns:
            dy_dt : Array of shape [N,1] containing the differential element of y(t)
        """
        theta = self.basis_xform(y.squeeze(), drivers_itp[np.argmin(np.abs(t_rkh - t)),:])
        return np.matmul(Xi, theta)
    
    def basis_xform(self,
                    q : utils.ArrayLikeType,
                    u : utils.ArrayLikeType) -> np.ndarray:
        """
        Transforms the input data into the "theta" vector for the SINDYc model.

        Arguments:
            q : Contains solution-space data
            u : Contains drivers
            state : state variable

        Returns:
            the transformed data
        """

        return np.array([f(q,u) for f in self.feature_library]).reshape((len(self.feature_library), 1))

    def construct_feature_library(self,
                                   q : utils.ArrayLikeType, 
                                   u : utils.ArrayLikeType,
                                   custom_library : dict = None) -> tuple[list[utils.FunctionType], list[str]]:
        """
        Method that constructs 

        Arguments:
            q : Array-like, represents feature space. Doesn't need to hold actual data.
            u : Array-like, represents driver space. Doesn't need to hold actual data.
            basis_library : list[str], holds list of basis library function types.
            custom_library : Dictionary with keys ["functions", "names"] holding the custom functions. custom_library["functions"] should be a list of functions.

        Returns:
            funcs : list of functions
            names : names of each function/model feature
        """
        
        funcs, names = [], []

        poly_filter = fnmatch.filter(self.basis_library, "poly_*")
        if len(poly_filter) > 0:
            for _basis in poly_filter:
                order = int(_basis[5:])
                _f, _n = self.construct_polynomial_basis(q, u, order=order)
                funcs += _f
                names += _n

        sine_filter = fnmatch.filter(self.basis_library, "sinusoidal_*")
        if len(sine_filter) > 0:
            for _basis in sine_filter:
                order = int(_basis[11:])
                _f, _n = self.construct_sinusoidal_basis(q, u, order=order)
                funcs += _f
                names += _n

        if custom_library is not None:
            funcs += custom_library["functions"]
            names += custom_library["names"]

        if len(funcs) == 0:
            raise Exception("No valid library functions detected. Current options are [poly_*(1,2,3), sinusoidal_*(n)], or specify a custom library dictionary.")
        
        return funcs, names
    
    @staticmethod
    def construct_polynomial_basis(q : utils.ArrayLikeType, 
                                    u : utils.ArrayLikeType,
                                    order : int) -> tuple[list[utils.FunctionType], list[str]]:

        #### convenience
        Nq, Nu = len(q), len(u)
        
        if order == 1:
            funcs = [lambda q, u, i=i: q[i] for i in range(Nq)] + \
                    [lambda q, u, i=i: u[i] for i in range(Nu)]
            names = [f"q{i}" for i in range(Nq)] + \
                    [f"u{i}" for i in range(Nu)]
            
        elif order == 2:
            funcs = [lambda q, u, i=i, j=j: q[i]*q[j] for i in range(Nq) for j in range(Nq)] + \
                    [lambda q, u, i=i, j=j: u[i]*u[j] for i in range(Nu) for j in range(Nu)] + \
                    [lambda q, u, i=i, j=j: q[i]*u[j] for i in range(Nq) for j in range(Nu)]
            names = [f"q{i}*q{j}" for i in range(Nq) for j in range(Nq)] + \
                    [f"u{i}*u{j}" for i in range(Nu) for j in range(Nu)] + \
                    [f"q{i}*u{j}" for i in range(Nq) for j in range(Nu)]
            
        elif order == 3:
            funcs = [lambda q, u, i=i, j=j, k=k: q[i]*q[j]*q[k] for i in range(Nq) for j in range(Nq) for k in range(Nq)] + \
                    [lambda q, u, i=i, j=j, k=k: u[i]*u[j]*u[k] for i in range(Nu) for j in range(Nu) for k in range(Nu)] + \
                    [lambda q, u, i=i, j=j, k=k: q[i]*q[j]*u[k] for i in range(Nq) for j in range(Nq) for k in range(Nu)] + \
                    [lambda q, u, i=i, j=j, k=k: q[i]*u[j]*u[k] for i in range(Nq) for j in range(Nu) for k in range(Nu)]
            names = [f"q{i}*q{j}*q{k}" for i in range(Nq) for j in range(Nq) for k in range(Nq)] + \
                    [f"u{i}*u{j}*u{k}" for i in range(Nu) for j in range(Nu) for k in range(Nu)] + \
                    [f"q{i}*q{j}*u{k}" for i in range(Nq) for j in range(Nq) for k in range(Nu)] + \
                    [f"q{i}*u{j}*u{k}" for i in range(Nq) for j in range(Nu) for k in range(Nu)]
            
        else:
            raise Exception(f"Order parameter {order} not supported. Must be one of [1, 2, 3]")
        
        return funcs, names

    @staticmethod
    def construct_sinusoidal_basis(q : utils.ArrayLikeType, 
                                    u : utils.ArrayLikeType,
                                    order : int = 1) -> list[utils.FunctionType]:

        #### convenience
        Nq, Nu = len(q), len(u)
        
        funcs = [lambda q, u, i=i: np.sin(q[i])**order for i in range(Nq)] + \
                [lambda q, u, i=i: np.sin(u[i])**order for i in range(Nu)]
        names = [f"sin^{order}(q{i})" for i in range(Nq)] + \
                [f"sin^{order}(u{i})" for i in range(Nu)]
        
        return funcs, names
    
    def get_coeffs(self, kp, kp_switch=3):
        """
        Gets the split coefficients.
        """
        if kp >= kp_switch:
            return self.Xi[0]
        else:
            return self.Xi[1]

    def interpolate_drivers(self,
                            t : utils.ArrayLikeType, 
                            tp : utils.ArrayLikeType, 
                            drivers: np.ndarray) -> np.ndarray:
        """
        Pre-builds the interpolated drivers.
        
        Arguments:
            t : Contains interpolated time axis
            tp : Contains original time axis
            drivers : array of size [len(tp), number of drivers], contains driver data

        Returns:
            Interpolated drivers on the new time axis
        """

        return utils.interpolate_matrix(
            t, tp, drivers, self.interp_method, axis=0
        )
    
    def RKO4_step(self,
                   y0 : utils.Union[float, np.ndarray],
                   t0 : float, 
                   h : float, 
                   *args) -> utils.Union[float, np.ndarray]:
        """
        Takes a single step in a Runge-Kutta O(4) solver for equations of the form dy/dt = f(t,y).
        Makes four function calls per time step.

        Presumes you will do a good job keeping function output shapes consistent, do pay attention to that.

        Arguments:
            func : f(t,y) in the equation.
            y0 : Solution on previous step (or initial condition)
            t0 : Time at which y(t) = y0
            h : Time step (same units as t0)
            *args : any additional arguments to be passed to func
        
        Returns:
            Solution for y at time t0 + h
        """
        k1 = self.ODE_function(
            t0, y0, 
            *args
        )
        k2 = self.ODE_function(
            t0 + h / 2.0, y0 + k1 * h / 2.0, 
            *args
        )
        k3 = self.ODE_function(
            t0 + h / 2.0, y0 + k2 * h / 2.0, 
            *args
        )
        k4 = self.ODE_function(
            t0 + h, y0 + k3 * h, 
            *args
        )
        return y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
#===================================== LSTM =====================================#

class LSTM:

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

#===================================== Convenience Functions & Typing =====================================#

ModelType = Union[SINDYc, LSTM]