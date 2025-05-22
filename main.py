"""
Main file.

Contact: Violet Player
Email: violet.player@colorado.edu
"""
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor as Pool
from forecast import driver

async def handle_client(reader, writer, executor):
    
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr!r}")
        
    while True:

        #### read user input
        input = await reader.read(100)
        if not input:
            break

        #### Build iterable of tasks from data
        tasks = driver.build_tasks(input)
        
        ##### Offload CPU-bound task to a separate process
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            driver.run,
            tasks,
        )
        
        writer.write(result)
        await writer.drain()
    
    print(f"Close the connection from {addr!r}")
    writer.close()


async def main(settings_file):

    #### initial file load
    settings = json.load(
        settings_file
    )

    #### set up forecaster 
    model = driver.build_model(
        settings["model"]
    )
    
    #### initialize multiprocessing pool
    pool = Pool(
        mp_context="forkserver" # forks retain parent process variables
    )

    #### open server; anonymous client function to add args
    server = await asyncio.start_server(
        lambda x, y : handle_client(x, y, pool), 
        '127.0.0.1', 
        8888
    )

    async with server:
        print("Server started. Waiting for connections...")
        await server.serve_forever()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    #-s SETTINGS FIILE
    parser.add_argument("-s", "--settings", help="Settings file")
    args = parser.parse_args()
    
    asyncio.run(
        main(
            args.settings
        )
    )