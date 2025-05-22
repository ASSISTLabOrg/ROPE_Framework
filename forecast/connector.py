import socket
from utils import vprint

class Connector:

    def __init__(self, HOST, PORT, verbose=True, **kwargs):
        self.HOST = HOST
        self.PORT = PORT
        self.verbose = verbose
        __dict__.update(**kwargs)
        self.connect()

    def connect(self):
        """
        Establishes a connection to the server and assigns the socket to the connection object
        """

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.socket.connect((self.HOST, self.PORT))
            vprint(f"Connected to {self.HOST}:{self.PORT}", self.verbose)

        except socket.error as e:
            vprint(f"Connection error: {e}", self.verbose)

    def transmit(self, data):

        """
        Sends data to the server and returns the ouput.

        
        """

        self.socket.sendall(data)
        result = self.socket.recv(1024)
        if not result:
            raise Exception("Server disconnected.")
        
        return result
    