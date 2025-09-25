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
from datetime import datetime, timedelta

#===================================== Typing and Dataclasses =====================================#

#### type decoration convencience
from types import FunctionType
ArrayLikeType = Union[np.ndarray, list, tuple]
NumericType = Union[int, float]

@dataclass
class PointSet:
    """
    Dataclass for a set of point in the atmosphere.

    Attributes:
        timestamp  : [str] ISO-8601 time stamp, representing t = 0 for the set (YYYY-MM-DDTHH:MM:SS.SSSSSS)
        time       : [array-like, 1D, positive-valued] time points (minutes)
        altitude   : [array-like, 1D, len(time)] height (km)
        longitude  : [array-like, 1D, len(time)] longitude (deg)
        latitude   : [array-like, 1D, len(time)] latitude (deg)
        avg_window : ***NOT IN USE*** [numeric] length of rolling averaging window (minutes)
    """

    timestamp : str
    time : ArrayLikeType
    altitude : ArrayLikeType
    longitude : ArrayLikeType
    latitude : ArrayLikeType
    #avg_window : NumericType

@dataclass
class Trajectory:
    """
    Dataclass wrapping a time-parameterized trajectory in the atmosphere.

    Attributes:
        timestamp : [str] ISO-8601 time stamp string representing t = 0 for the trajectory (YYYY-MM-DDTHH:MM:SS.SSSSSS)
        time      : [array-like, 1D] time of trajectory (hours)
        altitude  : [array-like, 1D, len(time)] height of trajectory (km)
        longitude : [array-like, 1D, len(time)] longitude of trajectory (deg)
        latitude  : [array-like, 1D, len(time)] latitude of trajectory (deg)
        avg_window : ***NOT IN USE*** [numeric] length of rolling averaging window (minutes)
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
        Trajectory    : [Trajectory] contains trajectory data
        parameters    : [dict] contains misc parameters - model dependent!
        drivers       : [ndarray, 2D (len(time), len(number of drivers))] space weather drivers
        driver_times  : [array-like] driver times (hours)
        initial_state : [arary-like] initial state-space of atmosphere in reduced space
    """

    Trajectory : Trajectory
    parameters : dict
    drivers : np.ndarray
    driver_times : ArrayLikeType
    initial_state : ArrayLikeType

@dataclass
class PhysicsGrid:
    """
    Dataclass wrapping the physics grid. Useful for building the KDTree for interpolation.

    Attributes:
        model : [str] name of the physics model [TIE-GCM, WAM-IPE, etc.]
        ndim  : [int] number of dimensions in model state-space (almost always 3)
        dims  : [array-like, list preferred] containing all dimensional axes
    """

    model : str
    ndim : int
    dims : ArrayLikeType

#===================================== Factory Methods =====================================#

def job_factory(config : dict, 
                *trajectories) -> list[Task]:
    """
    Factory method for producing a Job, which can be processed by a multiprocessing pool.

    Arguments:
        config        : [dict] the configuration dictionary
        *trajectories : [Trajectory objects] infinite-length argument for inputting trajectories

    Returns:
        job : [list] set of Task objects
    """

    tasks = []
    for trajectory in trajectories:
        sw_drivers, sw_times = get_sw_drivers(trajectory)
        tasks.append(
            Task(
                trajectory,
                get_params(config, trajectory),
                sw_drivers,
                sw_times,
                get_initial_state(trajectory)
            )
        )

    return tasks

#===================================== Task configuration =====================================#

def get_sw_drivers(trajectory : Trajectory) -> np.ndarray:
    """
    Gets the correct space-weather drivers given the timestamp.

    Arguments:
        trajectory : [Trajectory] path data

    Returns:
        times   : [ndarray] contains space weather driver time points
        drivers : [ndarray, (len(times), number of drivers)] contains space weather drivers
    """

    #### get timestamps for start and end of trajectory
    t_0 = datetime.fromisoformat(trajectory.timestamp)
    t_1 = t_0 + timedelta(seconds=3600 * (trajectory.time[-1] - trajectory.time[0]))

    #### number of day files to open
    ndays = np.ceil((t_1 - t_0).total_seconds() / (24 * 3600)).astype(int)

    #### Open files and collect data
    times = []
    drivers = []
    for i in range(ndays):
        t_i = t_0 + timedelta(i)
        fname = f"SW_drivers_{t_i.isoformat()[:10]}.h5"
        try:
            with h5py.File(fname, "r") as file:
                times += list(timedelta(i).total_seconds() / 60 + np.array(file["time"]))
                drivers.append(np.array(file["drivers"]))
        except:
            raise Exception(f"Cannot open file {fname}")
    
    return np.array(times), np.vstack(drivers)

def get_initial_state(trajectory : Trajectory) -> np.ndarray:
    """
    Gets the initial atmosphere state in reduced-space, given the timestamp.

    Arguments:
        trajectory : [Trajectory] path data
    
    Returns:
        state : [ndarray] contains the reduced-space state to initialize the model.
    """

    fname = f"state_{trajectory.timestamp[:10]}.h5"

    state = np.array([])

    return state

def get_params(config : dict, 
               trajectory : Trajectory) -> dict:
    """
    Gets the parameters required to run the solver, if any.

    Argumnents:
        config     : [dict] configuration data
        trajectory : [Trajectory] path data

    Returns:
        params : [dict] parameters required for model solver
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
        file : [str] full path of the file to convert

    Returns:
        output : [dict] data from file
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

def dict_to_hdf5(data : dict,
                 file : str):
    """
    Converts dictionary to hdf5 file with the same structure

    Arguments:
        data : [dict] dictionary containing the data to save
        file : [str] full path of the output file
    """

    def _build_file(h5_group, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                new_group = h5_group.create_group(key)
                _build_file(new_group, value)
            else:
                h5_group.create_dataset(key, data=value)

    with h5py.File(file, 'w') as f:
        _build_file(f, data)

def read_config(file : str) -> dict:
    """
    Reads .ini files into a dictionary.

    Arguments:
        file : [str] full path of the ini file

    Returns:
        config_dict : [dict] configuration data
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
                       method : str = "constant",
                       axis : int = -1) -> np.ndarray:
    """
    Interpolates a 2D matrix along the requested axis

    Arguments:
        x      : [array-like, 1D] new axis of interpolation
        xp     : [array-like, 1D] original axis (same units as x)
        A      : [ndarray, 2D] matrix to interpolate
        method : [str] interpolation type, one of (constant, linear)
        axis   : [int] which of two axes to interpolate along?

    Returns:
        A_itp : [ndarray, 2D] interpolated matrix, as requested.
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
        x      : [array-like, 1D] new axis of interpolation
        xp     : [array-like, 1D] original axis (same units as x)
        yp     : [array-like, 1D, len(xp)] data to interpolate
        method : [str] determines which type of interpolation. Supported: constant, linear

    Returns:
        y : [array-like, 1D] data interpolated onto x
    """

    if method.lower() == "constant":
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = yp[np.argmin(np.abs(xp - x[i]))]

    elif method.lower() == "linear":
        y = np.interp(x, xp, yp)

    else:
        raise Exception("Choose a valid interpolation method: [constant, linear, ]")
    
    return y

#===================================== Miscellany ====================================#

def vprint(statement : str, 
           verbose : bool = True):
    """
    Prints only if verbose.
    """
    if verbose:
        print(statement)