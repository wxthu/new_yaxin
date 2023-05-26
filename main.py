
import json
import multiprocessing as mp
import torch
import yaml
import math
from multiprocessing import Queue, Manager, Barrier

from multi_task import MultiTasks
from runtime import TaskEngine
from models.common import DetectMultiBackend

def lcm_list(lst: list):
    lcm = lst[0]
    for i in range(1, len(lst)):
        lcm = lcm * lst[i] // math.gcd(lcm, lst[i])
        
    return lcm

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # manager = Manager()
    filePath = "/home/air/yolov5/our_config.yaml"
    with open(file=filePath, mode="rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    tasks = config['task']
    ddls = config['deadline']
    batch = config['streams']
    shared_barr = Barrier(len(tasks) + 1)
    # Track the execution time and task processing status for each process
    results = [Queue() for _ in tasks]  
    shared_queues = [Queue() for _ in tasks]
    processes = list()
    
    mocked_tasks = MultiTasks(queues=shared_queues,
                              barrier=shared_barr,
                              lifetime=lcm_list(ddls)*20/1000,
                              intervals=[d / 1000 for d in ddls],
                              )
    
    
    for j, task in enumerate(tasks):
        processes.append(TaskEngine(name='task_%d'%(j),
                                    barrier=shared_barr,
                                    result=results[j],
                                    img_queue=shared_queues[j],
                                    batch=batch,
                                    model=task,
                                    ddl=ddls[j],
                                    )
                         )
    mocked_tasks.start()
    for ps in processes:
        ps.start()
    
    mocked_tasks.join()
    for ps in processes:
        ps.join()
        
    total_violated = 0
    total_task = 0
    timing = list()
    for result in results:
        start, end, violated, task_num = json.loads(result.get())
        timing.append(start)
        timing.append(end)
        total_violated += violated
        total_task += task_num
        
    qos_violation = round(total_violated / total_task, 3)
    print("Overall time: {} second".format(max(timing) - min(timing)))
    print("The number of timed-out task is {} and QoS violation rate: {}%".format(total_violated, 100 * qos_violation))
    