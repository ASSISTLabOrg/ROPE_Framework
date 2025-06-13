"""
Contains miscellaneous operations and convenience functions.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

#===================================== Imports =====================================#
from configparser import ConfigParser
import numpy as np
import h5py
from typing import Union
import asyncio
from dataclasses import dataclass

#===================================== Typing and Dataclasses =====================================#

#### type decoration convencience
from types import FunctionType
ArrayLikeType = Union[np.ndarray, list, tuple]
NumericType = Union[int, float]

@dataclass
class Trajectory:
    """
    Dataclass wrapping a time-parameterized trajectory in the atmosphere.

    Attributes:
        timestamp : ISO-8601 time stamp representing t = 0 for the trajectory.
        time : Array-like, time of trajectory - usually starts at 0 [hours], maximum 72 hours.
        altitude : Array-like, len == len(time) [km]
        longitude : Array-like, len == len(time)[deg]
        latitude : Array-like, len == len(time) [deg]
    """

    timestamp : str
    time : ArrayLikeType
    altitude : ArrayLikeType
    longitude : ArrayLikeType
    latitude : ArrayLikeType
    #avg_window : float = 0.5 TODO: maybe include time-averaging feature

@dataclass
class Task:
    """
    Dataclass wrapping the information required to execute a single forecast run.

    Attributes:
        Trajectory : a Trajectory object
        parameters : contains misc parameters - model dependent!
        drivers : ndarray of space weather drivers [len(driver_times), len(num_drivers)]
        driver_times : array-like of driver times [hours]
        initial_state : array-like of initial state-space of atmosphere in reduced space
    """

    Trajectory : Trajectory
    parameters : dict # will generally be model-specific
    drivers : np.ndarray
    driver_times : ArrayLikeType
    initial_state : ArrayLikeType

@dataclass
class PhysicsGrid:
    """
    Dataclass wrapping the physics grid. Useful for building the KDTree for interpolation.

    Attributes:
        model : string, name of the physics model [TIE-GCM, WAM-IPE, etc.]
        ndim : number of dimensions in model state-space (almost always 3)
        dims : array-like (usually list) containing all dimensional axes
    """

    model : str
    ndim : int
    dims : ArrayLikeType

#===================================== Factory Methods =====================================#

def job_factory(config : dict, 
                *trajectories) -> list[Task]:
    """
    Factory method for producing a Job, which can be processed by a multiprocessing pool.

    Arguments :
        config : The configuration dictionary
        *trajectories : infinite-length argument for inputting trajectories

    Returns :
        job : An iterable of Task objects
    """

    tasks = []
    for i, traj in trajectories:
        sw_drivers, sw_times = get_sw_drivers(traj)
        tasks.append(
            Task(
                traj,
                get_params(config, traj),
                sw_drivers,
                sw_times,
                get_initial_state(traj)
            )
        )

    return tasks

#===================================== Task configuration =====================================#

def get_sw_drivers(trajectory : Trajectory) -> np.ndarray:
    """
    Gets the correct space-weather drivers given the timestamp.

    Arguments:
        trajectory : Trajectory object

    Returns:
        drv_time : ndarray of shape len(trajectory.time), contains sw driver times
        drivers : ndarray of shape [len(trajectory.time), num_drivers]
    """

    drv_time = np.array([])
    drivers = np.array([])

    return drv_time, drivers

def get_initial_state(trajectory : Trajectory) -> np.ndarray:
    """
    Gets the initial atmosphere state in reduced-space, given the timestamp.

    Arguments:
        trajectory : Trajectory object
    
    Returns:
        state : ndarray, contains the reduced-space state to initialize the model.
    """

    state = np.array([])

    return state

def get_params(config : dict, 
               trajectory : Trajectory) -> dict:
    """
    Gets the parameters required to run the solver, if any.

    Argumnents:
        config : dictionary containing configuration data
        trajectory : Trajectory object

    Returns:
        params : dictionary containing parameters required for model solver
    """

    params = {}

    if config["model"]["type"].lower() == "sindyc":

        #### get all Runge-Kutta solver times
        dt = float(config["solver"]["timestep"])

        t_solve = np.linspace(
            trajectory.times[0],
            trajectory.times[-1],
            int(np.abs(trajectory.times[-1] - trajectory.times[0]) / dt) + 1
        )

        t_half = np.linspace(
            trajectory.times[0],
            trajectory.times[-1] + (t_solve[1] - t_solve[0]) / 2.0, # push past by a half-step
            int(2 * len(t_solve) / dt)
        )

        #### update parameter dictionary
        params.update(
            {
                "solver_timestep": dt,
                "solver_times_full": t_solve,
                "solver_times_half": t_half,
                "kp_index": int(config["solver"]["kp_index"])
            }
        )

    return params

#===================================== File I/O =====================================#

def hdf5_to_dict(file : str) -> dict:
    """
    Converts HDF5 file to a dictionary with the same key/value pairs

    Arguments:
        file : string, full path of the file to convert

    Returns:
        output : constructed dictionary containing all data
    """

    #### internal function to add nodes + data ¬to dictionary
    def _build_dict(_, node, output):
        current_level = output
        if isinstance(node, h5py.Dataset): # node is a dataset
            keys = node.name[1:].split("/")
            for key in keys:
                if key == keys[-1]:
                    current_level[f"{key}"] = node[:]
                else:
                    current_level = current_level.setdefault(key, {})
        else: # node is a group
            pass
    
    #### recursively iterate through all groups and return output
    output = {}
    with h5py.File(file, "r") as file:
        file.visititems(
            lambda _, node : _build_dict(
                _, node, output
            )
        )

    return output

def read_config(file : str) -> dict:
    """
    Reads .ini files into a dictionary.

    Arguments:
        file : string, full path of the ini file

    Returns:
        config_dict : dictionary containing all config data
    """
    
    config = ConfigParser()
    config.read(file)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = dict(config.items(section))
    return config_dict

async def read_file_async(filename : str, 
                          filetype : str):
    """
    Asynchronously (thread-safe) reads data from a file.
    """
    content = await asyncio.to_thread(
        lambda file : read_file(
            file, 
            filetype
        ), 
        filename
    )

    return content

def read_file(filename : str, 
              filetype : str = "csv"):
    """
    Wraps multiple file reading methods.
    """

    if filetype not in ["csv"]:
        raise Exception("Filetype not recognized. Must be one of [csv, ]")
    
    try:
        if filetype == "csv":
            with open(filename, 'r') as f:
                return f.read()
            
    except:
        raise Exception("File not found.")
    
#===================================== Array Manipulation ====================================#

def interpolate_matrix(x : ArrayLikeType,
                       xp : ArrayLikeType, 
                       A : np.ndarray, 
                       method="constant", 
                       axis=-1) -> np.ndarray:
    """
    Interpolates a 2D matrix along the requested axis

    Arguments:
        x : new axis of interpolation
        xp : original axis (same units as x)
        A : 2D array
        method : str, [constant, linear] interpolation
        axis : which of two axes to interpolate along?

    Returns:
        A_itp : interpolated matrix, as requested.
    """

    ### array-ify
    x = np.array(x)
    xp = np.array(xp)

    ### build new matrix shape
    dim0, dim1 = A.shape
    if axis == 0:
        dim0 = len(x)
    else:
        dim1 = len(x)
    A_itp = np.zeros((dim0, dim1))

    #### iterate over other axis to interpolate
    for i in range(A_itp.shape[np.abs(axis) - 1]):
        if axis == 0:
            A_itp[:,i] = _interp_1D(x, xp, A[:,i], method)
        else:
            A_itp[i,:] = _interp_1D(x, xp, A[i,:], method)

    return A_itp

def _interp_1D(x : ArrayLikeType, 
               xp : ArrayLikeType,
               yp : ArrayLikeType, 
               method : str = "constant"):
    """
    Wraps multiple 1D interpolation methods.

    Arguments:
        x : array-like of new points to interpolate onto
        xp : array-like of original points
        yp : array-like of data, len(xp) == len(yp)
        method : determines which type of interpolation. Supported: [constant, linear, ]

    Returns:
        y_itp : data, interpolated onto x
    """

    if method.lower() == "constant":
        y_itp = np.zeros(len(x))
        for i in range(len(x)):
            y_itp[i] = yp[np.argmin(np.abs(xp - x[i]))]

    elif method.lower() == "linear":
        y_itp = np.interp(x, xp, yp)

    else:
        raise Exception("Choose a valid interpolation method: [constant, linear, ]")
    
    return y_itp

#===================================== Miscellany ====================================#

def vprint(statement : str, 
           verbose : bool = False):
    """
    Prints only if verbose.
    """
    
    if verbose:
        print(statement)