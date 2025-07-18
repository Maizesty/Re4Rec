import sys
sys.path.append('../../')
sys.path.append('../')
import time
import onn
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.fx import subgraph_rewriter, symbolic_trace
import utils
from torch.fx import Proxy, Graph, GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.profiler import profile, record_function, ProfilerActivity
import time
import torch._dynamo as dynamo
def gen_pattern_replace_and_matcher_for_ffm(traced,
                                                  redundancy_part_slice,non_redundancy_part_slice,
                                                  embed_node_name,getitem_node_names,num_field,batch = 4096,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
  def _match(match,ori,pat):
    return True 
  class PatternClass(torch.nn.Module):
    def __init__(self,num_fields):
      super().__init__()
      self.num_fields = num_fields
      self.embeddings = torch.nn.ModuleList([
              torch.nn.Embedding(1, 1) for _ in range(self.num_fields-1)
          ])
    def field_aware_interaction(self,field_aware_emb_list):
      interaction = []
      for i in range(self.num_fields - 1):
          for j in range(i + 1, self.num_fields):
              v_ij = field_aware_emb_list[j - 1][:, i, :]
              v_ji = field_aware_emb_list[i][:, j, :]
              dot = torch.sum(v_ij * v_ji, dim=1, keepdim=True)
              interaction.append(dot)
      return torch.cat(interaction, dim=1)  
    def forward(self,x):
      field_aware_emb_list = [self.embeddings[i](x) for i in range(self.num_fields-1)]
      return self.field_aware_interaction(field_aware_emb_list)
  pattern = PatternClass(num_field)  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  embed_names = [f'embeddings_{i}' for i in range(num_field-1)]
  embed_node_list = [node_map[pattern_env[name]] for name in embed_names]
  embed_node_module_list = [utils.get_target_mod(traced,embed_node.target) for embed_node in embed_node_list]
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.embeddings = torch.nn.ModuleList(embed_node_module_list)
      self.num_fields = num_field
      self.num_prefix = redundancy_part_slice[1].stop
      self.num_sufix = self.num_fields - self.num_prefix
    def forward(self,x):
      redundancy_part = x[redundancy_part_slice] 
      non_redundancy_part = x[non_redundancy_part_slice] 
      redundancy_ffm_embed = [self.embeddings[i](redundancy_part) for i in range(self.num_fields-1)]
      non_redundancy_ffm_embed = [self.embeddings[i](non_redundancy_part) for i in range(self.num_fields-1)]
      redundancy_interaction = []
      for i in range(self.num_prefix - 1):
          for j in range(i + 1, self.num_prefix):
              v_ij = redundancy_ffm_embed[j - 1][ i, :]
              v_ji = redundancy_ffm_embed[i][ j, :]
              dot = torch.sum(v_ij * v_ji, dim=-1, keepdim=True)
              redundancy_interaction.append(dot)
              
      non_redundancy_interaction = []
      for i in range(self.num_sufix - 1):
          for j in range(i + 1, self.num_sufix):
              v_ij = non_redundancy_ffm_embed[j - 1+self.num_prefix][:, i, :]
              v_ji = non_redundancy_ffm_embed[i+self.num_prefix][:, j, :]
              dot = torch.sum(v_ij * v_ji, dim=-1, keepdim=True)
              non_redundancy_interaction.append(dot)
      mixed_interaction = []
      for i in range(self.num_prefix):
          for j in range(self.num_sufix):
              v_ij = redundancy_ffm_embed[j - 1+self.num_prefix][ i, :]
              v_ji = non_redundancy_ffm_embed[i - 1][ :,j, :]
              dot = torch.sum(v_ij * v_ji, dim=-1, keepdim=True)
              mixed_interaction.append(dot)
      mixed = torch.concat(mixed_interaction,dim = -1)
      non_redundancy = torch.concat(non_redundancy_interaction,dim = -1)
      redundancy = torch.concat(redundancy_interaction,dim = -1).repeat(batch,1)
      return torch.concat([redundancy,non_redundancy,mixed],dim = -1)
  return pattern,ReplacementClass(),_match    

def workload_onn(num_field, prefix,dim = 64,l = [1024,512,256],batch = 4096):
  print(f"now gen workload of ONN with config: dim: {dim}, num_field: {num_field}, prefix: {prefix}")
  onn_model = onn.ONN([100 for i in range(num_field)],dim,l,0.1)

  model_traced_ori = symbolic_trace(onn_model)
  
  onn_model_modify = onn.ONN([100 for i in range(num_field)],dim,l,0.1)
  onn_model_traced_modify = symbolic_trace(onn_model_modify)
  pattern,replace,match = gen_pattern_replace_and_matcher_for_ffm(onn_model_traced_modify,
                                                                      (0,slice(None,prefix,None)),(slice(None,None,None),slice(prefix,None,None)),
                                                                      embed_node_name = "embedding_embedding",
                                                                      getitem_node_names = ["getitem","getitem_1"],num_field=num_field,batch = batch)
  matches = subgraph_rewriter.replace_pattern_with_filters(onn_model_traced_modify, pattern, replace,[match])
  return model_traced_ori,onn_model_traced_modify

def genWorkload(num_field = 34 * 5,prefix = 29 * 5, batch = 4096, dim = 64):
  ori_model_name = f'/home/yssun/pytorch-fm/torchfm/model/test_fx/exp/model_repo/onn/onn_{batch}_{num_field}_{prefix}_{dim}_ori.onnx'
  modify_model_name = f'/home/yssun/pytorch-fm/torchfm/model/test_fx/exp/model_repo/onn/onn_{batch}_{num_field}_{prefix}_{dim}_modify.onnx'
  ori, modify = workload_onn(num_field,prefix,dim,l = [1024,512,256],batch = batch)
  torch.onnx.export(ori,               # 模型 being run
                  torch.randint(low=0, high=20, size=(batch,num_field), dtype=torch.long),                  # 模型输入 (or a tuple for multiple inputs)
                  ori_model_name,        # 导出文件的文件名
                  export_params=True, # 如果设置为True，则参数也会被导出。注意某些情况下参数可能无法被导出。
                  opset_version=10,   # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠以优化模型
                  input_names = ['input'],   # 输入的名称
                  output_names = ['output'], # 输出的名称
                  )
  torch.onnx.export(modify,               # 模型 being run
                  torch.randint(low=0, high=20, size=(batch,num_field), dtype=torch.long),                  # 模型输入 (or a tuple for multiple inputs)
                  modify_model_name,        # 导出文件的文件名
                  export_params=True, # 如果设置为True，则参数也会被导出。注意某些情况下参数可能无法被导出。
                  opset_version=10,   # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠以优化模型
                  input_names = ['input'],   # 输入的名称
                  output_names = ['output'], # 输出的名称
                  )
  
dims= [32]
batches = [1024,2048,4096]
num_field_and_prefixs = [(34 ,29),(22 ,10 )]


for dim in dims:
  for batch in batches:
    for num_field,prefix in num_field_and_prefixs:
      genWorkload(num_field=num_field,prefix=prefix,batch=batch,dim=dim)