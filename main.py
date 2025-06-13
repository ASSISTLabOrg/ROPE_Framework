"""
Main file. Launches server and handles client connections.

Contact: Violet Player
Email: violet.player@colorado.edu
"""
import asyncio
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor as Pool
from pyrope.utils import Task, job_factory, read_config, ArrayLikeType
from pyrope.models import model_factory, ModelType
from pyrope.transformers import transformer_factory, TransformerType

def intializer(config):
    #### will probably need to do something
    pass

def run(task : Task, 
        model : ModelType,
        transformer : TransformerType):
    
    #### forecast in reduced space, rho(t)
    fcst = model.forecast(
        task
    )

    #### transform + interpolate forecast
    result = transformer.expand(
        task.Trajectory,
        fcst,
        full=False
    )

    return result

async def handle_client(reader, 
                        writer, 
                        **kwargs):
    
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr!r}")
        
    while True:

        #### Read user input
        input = await reader.read(4096)
        if not input:
            break

        #### Unpickle data
        try:
            data = pkl.loads(
                input
            )
        except:
            raise Exception("Data must be unpickle-able.")

        #### quick fix for singleton Trajectory inputs
        if type(data) != ArrayLikeType:
            data = [data]

        #### Build iterable of tasks (or a "job") from recieved Trajectories
        job = job_factory(
            kwargs["config"],
            *data
        )
        
        ##### Offload CPU-bound task to the pool
        result = await asyncio.get_event_loop().run_in_executor(
            kwargs["pool"],
            lambda task : run(
                task,
                kwargs["model"],
                kwargs["transformer"]
            ),
            job,
        )
        
        writer.write(result)
        await writer.drain()
    
    print(f"Close the connection from {addr!r}")
    writer.close()

async def main(config):

    #### build model & transformer
    global model, transformer
    model = model_factory(
        config
    )
    transformer = transformer_factory(
        config
    )
    
    #### initialize multiprocessing pool
    pool = Pool(
        max_workers=int(config["pool"]["max_workers"]),
        mp_context=config["pool"]["mp_context"],
        initializer=intializer,
        initargs=config,
    )

    #### open server; anonymous client function to add args
    server = await asyncio.start_server(
        lambda x, y : handle_client(
            x, y,
            {
                "pool": pool,
                "model": model,
                "transformer": transformer,
                "config": config,
            }
        ),
        config["server"]["address"],
        int(config["server"]["port"])
    )

    async with server:
        print("Server started. Waiting for connections...")
        await server.serve_forever()

if __name__ == "__main__":

    #### first, parse CLI arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()

    #-c CONFIGURATION FILE
    parser.add_argument("-c", "--config", help="Configuration file")
    args = parser.parse_args()

    #### then launch server
    asyncio.run(
        main(
            read_config(
                args.config
            )
        )
    )