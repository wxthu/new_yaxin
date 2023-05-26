
import torch
import torch.nn as nn
import multiprocessing as mp

import time
import json
from models.common import DetectMultiBackend

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class TaskEngine(mp.Process):
    def __init__(self, 
                 name: str,
                 barrier: mp.Barrier,
                 result: mp.Queue,
                 img_queue: mp.Queue,
                 batch: int,
                 model: str,
                 ddl: float,
                ):
        super().__init__(name=name)
        self._name = name
        self._barrier = barrier
        self._tracking = result
        self._img_queue = img_queue
        self._model = DetectMultiBackend(weights=model, fp16=True)
        self._bsize = batch 
        self._ddl = ddl # ms
        
    def warmup(self):
        img = torch.rand(1, 3, 640, 640, dtype=torch.float16).cuda()
        self._model.to('cuda')
        for _ in range(50):
            out = self._model(img)
        
        torch.cuda.synchronize()
        del img, out

    @torch.no_grad()
    def run(self):
        self.warmup()
        self._barrier.wait()
        
        violated = 0
        task_num = 0
        p_start = time.time()
        while True:
            signal = self._img_queue.get()
            if signal == 'exit': break

            start = time.time()
            ims = torch.rand((self._bsize, 3, 640, 640), dtype=torch.float16).cuda()
            y = self._model(ims)
            torch.cuda.synchronize()
            
            # len(y) == 2 and len(y[0]) == 1, len(y[1]) == 3
            del ims
            y[0] = y[0].cpu()
            for res in y[1]:
                res = res.cpu()  
            end  = time.time()
            task_num += self._bsize
            if (end - start) * 1000 > self._ddl:
                violated += self._bsize
            # print(f"{self._name} completes inference one time ...")
        
        p_end = time.time()    
        self._tracking.put(json.dumps((p_start, p_end, violated, task_num)))
        
        print("{} has finished successfully !!!".format(self._name))

