import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx import Proxy, Graph, GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.profiler import profile, record_function, ProfilerActivity
@torch.compile
def f(x):
#   reorder_x =  empty_like_tensor[:,l_100,:]
  return x * x
@torch.compile()
def f_modify_without_reorder(x):
  empty_like_tensor = torch.ones((4096,100,64),device="cpu")
  result_x = x[0,:50,:] * x[0,:50,:]
  empty_like_tensor[:,:50,:] = result_x
  empty_like_tensor[:,50:,:] = x[0,50:,:] * x[0,50:,:]
#   reorder_x =  empty_like_tensor[:,l_100,:]
  return empty_like_tensor
f(torch.ones([4096,100,8],device = "cpu"))
total_time = []
t = torch.ones([4096,100,8],device="cpu")
for i in range(1000):
#   start_time = time.time()  # 开始计时
    with torch.no_grad():
        f(t)
        #   end_time = time.time()  # 结束计时
print(1)          
          # 计算并打印函数执行所需的时间
        #   elapsed_time = end_time - start_time
        #   total_time.append(elapsed_time * 1000)
    #   print(calculate_mean_and_variance_manual(total_time))
# l = [i for i  in range(100) if i != 1 and i != 2]
# l_100 = [i for i  in range(100)][::]
# @torch.compile(fullgraph=True)
# def f(x: torch.Tensor):
#       empty_like_tensor = torch.ones((4096,100,32),device='cpu')
#       result_x = x[0,(1,2),:] * x[0,(1,2),:]
#       empty_like_tensor[:,(1,2),:] = result_x
#       empty_like_tensor[:,l,:] = x[0,l,:] * x[0,l,:]
#     #   reorder_x =  empty_like_tensor[:,l_100,:]
#       return torch.sum(empty_like_tensor, dim = 1)
# f(torch.empty([4096,100,32],device='cpu'))
# import sys
# sys.path.append('../../')
# sys.path.append('../')

# import pnn


# pnn_loop = pnn.ProductNeuralNetworkModel([50 for i in range(100)],32,[100,100,100],0.1)
# model_compile = torch.compile(pnn_loop.cuda())
# pnn_loop = pnn_loop.cuda()
# t = torch.randint(low=0, high=20, size=(4096,100), dtype=torch.long).cuda()
# import time 

# def time_evaluation(origin, compiled, input, exec_func=None, exp_name: str = '', warmup_time: int = 5) -> None:
#     torch.cuda.synchronize()
#     s_t = time.time()
#     exec_func(origin, input) if exec_func else origin(input)
#     torch.cuda.synchronize()
#     start_t1 = time.time() - s_t
#     print(f"Normal firstly used time:{start_t1}s")

#     torch.cuda.synchronize()
#     s_t = time.time()
#     exec_func(compiled, input) if exec_func else compiled(input)
#     torch.cuda.synchronize()
#     start_t2 = time.time() - s_t
#     print(f"Compiled firstly used time:{start_t2}s")

#     assert warmup_time >= 1
#     for _ in range(warmup_time - 1):
#         exec_func(compiled, input) if exec_func else compiled(input)

#     t_1_total, t_2_total = 0., 0.
#     for i in range(10):
#         torch.cuda.synchronize()
#         s_t = time.time()
#         exec_func(origin, input) if exec_func else origin(input)
#         torch.cuda.synchronize()
#         t_1 = time.time() - s_t
#         t_1_total += t_1

#         torch.cuda.synchronize()
#         s_t = time.time()
#         exec_func(compiled, input) if exec_func else compiled(input)
#         torch.cuda.synchronize()
#         t_2 = time.time() - s_t
#         t_2_total += t_2

#         print(f"{i}:\n\tNormal used time:{t_1}s, \n\t"
#               f"Compiled used time:{t_2}s")

#     print(f"{exp_name}在编译前的首次运行时间为:{start_t1}秒")
#     print(f"{exp_name}在编译后的首次运行时间为:{start_t2}秒")
#     print(f"{exp_name}在后续运行过程中的加速比为:{t_1_total / t_2_total:.2f}")

# time_evaluation(pnn_loop,model_compile,t)