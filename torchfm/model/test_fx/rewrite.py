import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import torch.fx as fx
from torch.fx import Proxy, Graph, GraphModule
import operator


# def with_func(input_node,)



def reducesum_rewrite(traced,redency_part_slice,unredency_part_slice,input_node_name,target_node_name):
  graph = traced.graph
  env  = utils.get_env(traced)
  input_node = env[input_node_name]
  target_node = env[target_node_name]
  with graph.inserting_before(target_node):
    # 冗余数据提取
    new_redency_extract_node = graph.call_function(operator.getitem,(input_node,redency_part_slice))
    # 非冗余数据提取
    new_unredency_extract_node = graph.call_function(operator.getitem,(input_node,unredency_part_slice))
    
    # 冗余部分计算
    new_redency_sum_node = graph.call_function(torch.sum,(new_redency_extract_node,),{'dim':0})
    # 非冗余数据计算
    new_unredency_sum_node = graph.call_function(torch.sum,(new_unredency_extract_node,),{'dim':1})
    # 操作还原
    new_add_node = graph.call_function(torch.add,(new_redency_sum_node,new_unredency_sum_node))
    utils.replace_use_with(target_node,new_add_node)
  # map_dict = get_successors_map(traced)
  # successors = map_dict[target_node_name]
  # for successor in successors:
  #   successor_node = env[successor]
  #   arg_tuple = successor_node.args
  #   new_tuple = list(arg_tuple)
    
  # graph.erase_node(env['output'])
  # graph.output(new_add_node)
  graph.eliminate_dead_code()
  graph.lint() 
  return graph




def linear_rewrite(traced,redency_part_slice,unredency_part_slice,input_node_name,target_node_name):
  graph = traced.graph
  env = utils.get_env(traced)
  input_node = env[input_node_name]
  target_node = env[target_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node_name,'_')
  target_weight = target_node_mod.weight
  target_bias = target_node_mod.bias
  
  # redency_linear
  redency_weight = target_weight[:,redency_part_slice[1]]
  redency_linear = torch.nn.Linear(redency_weight.shape[0],redency_weight.shape[1])
  redency_linear.weight.data = redency_weight
  redency_linear.bias.data = target_bias
  
  # unredency_linear
  unredency_weight = target_weight[:,unredency_part_slice[1]]
  unredency_linear = torch.nn.Linear(unredency_weight.shape[0],unredency_weight.shape[1],bias=False)
  unredency_linear.weight.data = unredency_weight
  
  traced.register_module(f"{target_node_name}_redency",redency_linear)
  traced.register_module(f"{target_node_name}_unredency",unredency_linear)
  with graph.inserting_before(target_node):
    # 冗余数据提取
    new_redency_extract_node = graph.call_function(operator.getitem,(input_node,redency_part_slice))
    # 非冗余数据提取
    new_unredency_extract_node = graph.call_function(operator.getitem,(input_node,unredency_part_slice))

    
    # 冗余计算
    new_redency_compute_node = graph.call_module(f"{target_node_name}_redency",(new_redency_extract_node,))
    # 非冗余计算
    new_unrendency_compute_node = graph.call_module(f"{target_node_name}_unredency",(new_unredency_extract_node,))
    new_add_node = graph.call_function(torch.add,(new_redency_compute_node,new_unrendency_compute_node))
    utils.replace_use_with(target_node,new_add_node)
  graph.eliminate_dead_code()
  graph.lint() 
  return graph