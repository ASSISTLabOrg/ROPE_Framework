import asyncio
import socket
import time
from utils import vprint

class Connector:

    def __init__(self, **kwargs):
        __dict__.update(**kwargs)
        self.connect()

    def connect(self):
        """
        Establishes a connection to the server and assigns the socket to the connection object
        """
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.conn.connect((self.HOST, self.PORT))
            vprint(f"Connected to {self.HOST}:{self.PORT}", self.verbose)
        except socket.error as e:
            vprint(f"Connection error: {e}", self.verbose)

    def maintain(self):
        """Keeps the connection alive by periodically sending a keep-alive message."""
        while True:
            try:
                self.conn.sendall(b"keep-alive")  # Send a keep-alive message
                data = self.conn.recv(1024)
                if not data:
                    vprint("Server disconnected.", self.verbose)
                    return False  # Connection lost
                #print(f"Received: {data.decode()}")
                time.sleep(60)  # Wait for 60 seconds
            except socket.error as e:
                vprint(f"Socket error: {e}", self.verbose)
                return False


async def read_file_async(filename, filetype):

    content = await asyncio.to_thread(
        lambda file : read_file_sync(file, filetype), filename
    )

    return content

def read_file_sync(filename, filetype="csv"):

    if filetype not in ["csv"]:
        raise Exception("Filetype not recognized. Must be one of [csv, ]")
    
    try:
        if filetype == "csv":
            with open(filename, 'r') as f:
                return f.read()
            
    except:
        raise Exception("File not found.")