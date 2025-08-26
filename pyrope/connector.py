"""
Contains Connector class, which enables easily connecting to an existing mproc server.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

import socket
from pyrope.utils import vprint

class Connector:
    """
    TODO: Documentation
    """

    def __init__(self, HOST, PORT, verbose=True, **kwargs):

        self.HOST = HOST
        self.PORT = PORT
        self.verbose = verbose
        self.__dict__.update(**kwargs)

        #### initialize connection
        self.connect()

    def connect(self):
        """
        Establishes a connection to the server and assigns the socket to the connection object.

        """

        self.socket = socket.socket(
            socket.AF_INET, 
            socket.SOCK_STREAM
        )

        try:
            self.socket.connect(
                (self.HOST, 
                 self.PORT)
            )
            vprint(f"Connected to {self.HOST}:{self.PORT}", self.verbose)

        except socket.error as e:
            vprint(f"Connection error: {e}", self.verbose)

    def send_and_recv(self, data):
        """
        Sends data to the server and returns the ouput.

        Arguments :
            data : pickled data object

        Returns :
            result : output from server
        """

        self.socket.sendall(data)
        result = self.socket.recv(4096)
        if not result:
            raise Exception("Server disconnected.")
        
        return result
    