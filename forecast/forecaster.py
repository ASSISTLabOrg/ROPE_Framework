"""
The DensityForecaster class provides a user-side interface that will manage connecting to an active server (or run locally if they don't).

The use can then send a density r

"""
from connector import Connector
from encoders import build_encoder
from models import build_model
from manager import make_tasks

def DensityForecaster():

    def __init__(self, settings, verbose=False, **kwargs):

        #### assign attributes
        self.connect = settings["connect_bool"]
        __dict__.update(**kwargs)

        #### connect to remote, if possible
        if self.connect:
            self.HOST = settings["remote_host"]
            self.PORT = settings["port"]
            self.conn = Connector(self.HOST, self.PORT, self.verbose)

        #### otherwise, build local model. Currently runs in serial mode only.
        else:
            self.model = build_model(settings["model"])
            self.encoder = build_encoder(settings["encoder"])

    def forecast(self, t, x):

        if self.connect:
            packet = (t, x) # adjust this
            result = self.conn.transmit(packet) # sends data to server, awaits return

        else:
            tasks = make_tasks(t, x)
            result = self.model.forecast(tasks)
        
        return result