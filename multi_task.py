
import threading
from multiprocessing import Process, Queue, Barrier
import time
import logging
from typing import List

class MultiTasks(Process):
    def __init__(self,
                 queues: List[Queue],
                 barrier: Barrier, 
                 lifetime: int,
                 intervals: List[float]):
        
        super().__init__()
        self._lifetime = lifetime # denote the duration of video streams
        self._barrier =  barrier
        self._task_num = len(queues)
        self._queues = queues
        self._intervals = intervals
        
        assert len(queues) == len(intervals)
        
    def send(self, 
             barrier: threading.Barrier,
             task_id: int):
        
        barrier.wait()
        if self._intervals[task_id] == 0.0:
            logging.warning('No data processed for task {}'.format(task_id))
            return
        
        count = 0
        target_num = int(self._lifetime / self._intervals[task_id])
        while count < target_num:
            self._queues[task_id].put('img')
            count += 1
            time.sleep(self._intervals[task_id])
        
        self._queues[task_id].put('exit')
        print("#### Task_{} finished and image number: {} ####".format(task_id, target_num))
            
    def run(self):
        thread_barrier = threading.Barrier(self._task_num + 1)
        thread_pools = []
        for task_id in range(self._task_num):
            thread = threading.Thread(target=self.send,
                                      args=(thread_barrier, task_id), 
                                      name="Queue_%d" % (task_id))
            thread.start()
            thread_pools.append(thread)
            
        self._barrier.wait()
        thread_barrier.wait()
        
        for t in thread_pools:
            t.join()
        
        print("All image queue threads complete successfully !!!")
        