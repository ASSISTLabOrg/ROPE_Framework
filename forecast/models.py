#### generic python
import numpy as np
import fnmatch
from json import load
from typing import Union
import utils
from manager import Task

def build_model(settings):

    #### load the model variables
    model_vars = load(
        settings["input_file"]
    )

    if settings["model_type"].lower() is "sindyc":
        return SINDYc(**model_vars)
    
    else:
        raise Exception("Model type not supported. Currently supports: [SINDYc, ]")

class SINDYc:
    """
    Attributes :
        Xi : coefficient matrices. Currently expects list/tuple containining (Xi_1, Xi_2)
        feature_library : library of model functions
        feature_names : names of functions; purely for inspection
        kp_index : row index of kp data in drivers
        f10_index : row index of f10.7 data in drivers
        interp_method : how to interpolate matrices
        time_step : time step of runge-kutta method

    Methods :
        forecast : accepts a Task and forecasts the result in reduced-space from that task
    """

    def __init__(self, **kwargs):

        __dict__.update(**kwargs)
        
        #### make sure all the necessary variables are in place
        if not self._init_check():
            raise Exception("SINDYc model not properly inialized.")

    def forecast(self,
                 task : Task) -> np.ndarray:

        #### interpolate drivers to half-step times
        drivers_itp = self._interpolate_drivers(
            task.parameters["solver_times_half"], 
            task.parameters["driver_times"], 
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

            Xi = self._get_coeffs_SINDYc(
                task.drivers[i,self.kp_index]
            )

            fcst[i,:] = self._RKO4_step(
                self._ODE_function, 
                fcst[i-1,:].reshape((-1,1)),
                task.parameters["solver_times_full"][i],
                self.time_step,
                *[task.parameters["solver_times_half"], 
                  drivers_itp, 
                  Xi
                  ]
            ).squeeze()
            
        if task.parameters["forecast_times"] is not None:
            fcst = utils.interpolate_matrix(
                task.parameters["forecast_times"], 
                task.parameters["solver_times_full"], 
                fcst, 
                self.interp_method, 
                axis=0
            )

        return fcst

    def _init_check(self):
        """
        Checks all expected variables in the SINDYc model setup.
        """
        loaded_attrs = [key for key in __dict__.keys()]
        check_attrs = ["Xi", "feature_library", "feature_names", "kp_index", "f10_index", "interp_method", "time_step"]

        check = True
        for key in check_attrs:
            if key not in loaded_attrs:
                check = False
                break

        return check
    
    def _ODE_function(self,
                      t : float,
                      y : np.ndarray,
                      t_rkh : utils._array_like,
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
        theta = self._basis_xform(y.squeeze(), drivers_itp[np.argmin(np.abs(t_rkh - t)),:])
        return Xi @ theta
    
    def _basis_xform(self,
                     q : utils._array_like,
                     u : utils._array_like) -> np.ndarray:
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

    def _construct_feature_library(self,
                                   q : utils._array_like, 
                                   u : utils._array_like,
                                   custom_library : dict = None) -> tuple[list[utils.ftype], list[str]]:
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
                _f, _n = self._construct_polynomial_basis(q, u, order=order)
                funcs += _f
                names += _n

        sine_filter = fnmatch.filter(self.basis_library, "sinusoidal_*")
        if len(sine_filter) > 0:
            for _basis in sine_filter:
                order = int(_basis[11:])
                _f, _n = self._construct_sinusoidal_basis(q, u, order=order)
                funcs += _f
                names += _n

        if custom_library is not None:
            funcs += custom_library["functions"]
            names += custom_library["names"]

        if len(funcs) == 0:
            raise Exception("No valid library functions detected. Current options are [poly_*(1,2,3), sinusoidal_*(n)], or specify a custom library dictionary.")
        
        return funcs, names
 
    def _construct_polynomial_basis(q : utils._array_like, 
                                    u : utils._array_like,
                                    order : int) -> tuple[list[utils.ftype], list[str]]:

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

    def _construct_sinusoidal_basis(q : utils._array_like, 
                                    u : utils._array_like,
                                    order : int = 1) -> list[utils.ftype]:

        #### convenience
        Nq, Nu = len(q), len(u)
        
        funcs = [lambda q, u, i=i: np.sin(q[i])**order for i in range(Nq)] + \
                [lambda q, u, i=i: np.sin(u[i])**order for i in range(Nu)]
        names = [f"sin^{order}(q{i})" for i in range(Nq)] + \
                [f"sin^{order}(u{i})" for i in range(Nu)]
        
        return funcs, names
    
    def _interpolate_drivers(self,
                             t : utils._array_like, 
                             tp : utils._array_like, 
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
    
    def _RKO4_step(self,
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
    
_modeltype = Union[SINDYc]