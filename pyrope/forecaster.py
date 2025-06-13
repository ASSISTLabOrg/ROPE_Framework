"""
Contains the DensityForecaster class - this is a client-side convencience object that manages all the details using the Client.ini file.

>>> from ROPE.forecaster import DensityForecaster
>>> df = DensityForecaster()
>>> trajectory_1 = df.make_trajectory(iso_timestamp, time, altitude, latitude, longitude)
>>> trajectory_2 = df.make_trajectory(iso_timestamp_2, time_2, altitude_2, latitude_2, longitude_2) # and so on
>>> density = df.forecast(trajectory_1, trajectory_2) # returns density traces for each trajectory

Contact: Violet Player
Email: violet.player@noaa.gov
"""
import sys
import os
import pickle as pkl
from connector import Connector
from utils import Trajectory, ArrayLikeType
from utils import job_factory, read_config
from models import model_factory
from transformers import transformer_factory

def DensityForecaster():

    def __init__(self, 
                 config_file="ROPE_Framework/ClientDemo.ini", #TODO: better path handling, client probably connecting from another directory
                 verbose=False, 
                 **kwargs):
        
        
        #### load config file
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # TODO: Fix this, terrible practice
        config = read_config(
            config_file
        )

        #### assign attributes
        self.connect = bool(config["server"]["connect"])
        self.verbose = verbose
        self.__dict__.update(**kwargs)

        #### connect to remote, if possible
        if self.connect:
            self.HOST = config["server"]["remote_host"]
            self.PORT = config["server"]["port"]
            self.connection = Connector(self.HOST, self.PORT, self.verbose)

        #### otherwise, build local model. Currently runs in serial mode only.
        else:
            self.model = model_factory(config)
            self.transformer = transformer_factory(config)

    @staticmethod
    def make_trajectory(time : ArrayLikeType,
                        altitude : ArrayLikeType,
                        longitude : ArrayLikeType,
                        latitude : ArrayLikeType):

        return Trajectory(
            time,
            altitude,
            longitude,
            latitude
        )

    def forecast(self, *trajectories):
        """
        Predicts the density for the provided trajectories

        Arguments :
            *trajectories : as many Trajectory objects as you want!

        Returnts :
            Predicted density trace for each submitted trajectory
        """

        if self.connect:
            
            #### pickle trajectories
            job = pkl.dumps(
                trajectories,
                protocol=pkl.HIGHEST_PROTOCOL
            )

            #### sends data to server, awaits return and unpickles
            result = pkl.loads(
                self.connection.send_and_recv(
                    job
                )
            )

        else:
            job = job_factory(
                trajectories
            )
            result = [
                self.model.forecast(task) for task in job.tasks
            ]
        
        return result