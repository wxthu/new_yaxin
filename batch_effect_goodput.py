from turtle import title
import torch
import time 
import matplotlib.pyplot as plt

from models.common import DetectMultiBackend

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

@torch.no_grad()
def main():
    model = DetectMultiBackend(weights='yolov5x.pt', fp16=True)
    model.eval().to('cuda')
    max_batch = 8
    for i in range(max_batch):
        im = torch.zeros(i+1, 3, 640, 640, dtype=torch.float16).cuda()
        for _ in range(50):
            out = model(im)
            
    throughput = []
    for i in range(1, max_batch+1):
        im = torch.zeros(i, 3, 640, 640, dtype=torch.float16).cuda()
        start = time_sync()
        for _ in range(50):
            out = model(im)
        end = time_sync()
        throughput.append(round(1/((end - start)/50/i), 1))
        
    fig, ax = plt.subplots()
    ax.set(title='batch size effect on throughput ')
    # ax.bar(range(1, max_batch+1), throughput, color="#DC564C", edgecolor="white", linewidth=0.5)
    ax.plot(range(1, max_batch+1), throughput, color='#DC564C', marker='o', ms=2.8, lw=1.2)
    ax.set(xlabel='batch size')
    ax.set(ylabel='throughput/s')
    fig.savefig("batch_vs_throughout_yolov5x_fp16.jpg", dpi=512)
    
if __name__ == '__main__':
    main()