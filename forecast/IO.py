import asyncio

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