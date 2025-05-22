import numpy as np
from dataclasses import dataclass
from models import _modeltype
from encoders import _encodertype

@dataclass
class Task:
    id : int
    time : float
    position : np.ndarray
    parameters : dict
    drivers : np.ndarray


def make_tasks(input) -> list[Task]:
    tasks = []
    return tasks

def run(task : Task, 
        model : _modeltype, 
        encoder : _encodertype):

    return encoder.decode(
        task.time,
        task.position,
        model.forecast(
            task
        ), 
        full=False
    )