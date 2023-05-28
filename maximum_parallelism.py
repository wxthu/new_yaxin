import torch
import time
import numpy as np
import multiprocessing as mp 
from multiprocessing import Process, Barrier, Manager

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 5

from models.common import DetectMultiBackend

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Task(Process):
    def __init__(self,
                 name: str,
                 barrier: Barrier,
                 private_time,
                 batch: int = 1):
        super().__init__(name=name)
        self._barrier = barrier
        self._model = DetectMultiBackend(weights='yolov5x.pt', fp16=True)
        self._bsize = batch
        self._private = private_time
    
    @torch.no_grad()   
    def run(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            self._model.eval().to('cuda')
            img = torch.rand(self._bsize, 3, 640, 640, dtype=torch.float16).cuda()
            for _ in range(50):
                y = self._model(img)
            torch.cuda.synchronize()
            
            # formal comparision
            self._barrier.wait()
            start = time_sync()
            for _ in range(50):
                y = self._model(img)
            end = time_sync()
            
            self._private.append(round(1000*(end - start)/50, 2))
      

def main():
    batch = 1
    parallelism = 6
    avg_latency = []
    for i in range(1, parallelism + 1):
        barrier = Barrier(i)
        timing = [Manager().list() for _ in range(i)]
        tasks = []
        for j in range(i):
            tasks.append(Task(name='Process_%d'%(j),
                              barrier=barrier,
                              private_time=timing[j],
                              batch=batch,
                              )
                         )
            tasks[-1].start()
            
        for j in range(i):
            tasks[j].join()
        
        avg_latency.append(np.mean([list(lst) for lst in timing]))
        
    # plotting
    fig, ax = plt.subplots(layout='constrained')
    ax.bar(range(1, parallelism+1), avg_latency, width=0.3, color='#0099CC')
    ax.set_ylabel('Latency/ms')
    ax.set_xlabel('Parallelism level')
    ax.set_xticks(range(1, parallelism+1))
    ax.set_title("Inference performance of the YOLOv5x model \n under different levels of parallelism")
    for x, y in zip(range(1, parallelism+1), avg_latency):
        ax.text(x, y+0.05, '%.1f' % y, ha='center', va='bottom')
    fig.savefig('parallelism_level.jpg')
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
    