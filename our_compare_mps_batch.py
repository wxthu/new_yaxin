from turtle import color
import torch
import time
import numpy as np
import multiprocessing as mp 
from multiprocessing import Process, Barrier, Manager

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 5

from models.common import DetectMultiBackend
from torchvision.models import resnet50, mobilenet_v2, inception_v3

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Task(Process):
    def __init__(self,
                 pname: str,
                 engine,
                 barrier: Barrier,
                 private_time,
                 batch: int = 1):
        super().__init__(name=pname)
        self._barrier = barrier
        self._model = engine
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
            
            self._private.append(round(1000*(end - start)/50, 1))
      

def main():
    
    parallelism = 6
    
    for name, engine in models.items():
        avg_latency = []
        for i in range(1, parallelism + 1):
            barrier = Barrier(i)
            timing = [Manager().list() for _ in range(i)]
            tasks = []
            for j in range(i):
                tasks.append(Task(pname='Process_%d'%(j),
                                  engine=engine,
                                  barrier=barrier,
                                  private_time=timing[j],
                                )
                            )
                tasks[-1].start()
                
            for j in range(i):
                tasks[j].join()
            
            avg_latency.append(np.mean([list(lst) for lst in timing]))
        
        batch_latency = []
        for j in range(1, parallelism+1):
            im = torch.zeros(j, 3, 640, 640, dtype=torch.float16).cuda()
            engine.to('cuda')
            for _ in range(50):
                engine(im)
                
            start = time_sync()
            for _ in range(50):
                out = engine(im)
            end = time_sync()
            batch_latency.append(round(1000*(end-start)/50, 1))
         
        # plotting
        fig, ax = plt.subplots(layout='constrained')
        
        ax.bar(range(1, parallelism+1), avg_latency, label='MPS', width=0.3, color='#0099CC')
        ax.set_ylabel('Latency/ms')
        ax.set_xlabel('Parallelism level')
        ax.set_xticks(range(1, parallelism+1))
        ax.set_title("{} performance comparion between MPS and Batch".format(name))
        for x, y in zip(range(1, parallelism+1), avg_latency):
            ax.text(x, y+0.05, '%.1f' % y, ha='center', va='bottom')
        
        ax.plot(range(1, parallelism+1), batch_latency, label='Batch', marker='o', ms=6, color='#EE9B00')
        ax.legend()
        fig.savefig('cmp_mps_batch_%s.jpg' % (name))
        print('model {} performance measurement finished !'.format(name))
        time.sleep(60)
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    models = {'yolov5x' : DetectMultiBackend(weights='yolov5x.pt', fp16=True),
              'yolov5m' : DetectMultiBackend(weights='yolov5m.pt', fp16=True),
              'yolov5s' : DetectMultiBackend(weights='yolov5s.pt', fp16=True),
              'resnet50' : resnet50().half().eval(),
              'mobilenet_v2' : mobilenet_v2().half().eval(),
              'inception_v3' : inception_v3(init_weights=True).half().eval(),
              }
    main()
    