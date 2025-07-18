import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.fx import subgraph_rewriter, symbolic_trace
import utils
from torch.fx import Proxy, Graph, GraphModule

import operator

op_map = {
  '<built-in function pow>':operator.pow
}



def gen_pattern_replace_and_matcher_for_single_op(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  target_node_name,
                                                  batch_size):
  env  = utils.get_env(traced)
  target_node = env[target_node_name]
  target_op = op_map[str(target_node.target)]
  
  def _match(match,ori,pat):
    target_node = None
    for node in pat.nodes:
      if node.op == "call_function" and node.target == target_op:
        target_node = node
    return match.nodes_map[target_node].name == target_node_name
  
  #匹配模板
  def pattern(x,y):
    return target_op(x,y)
  
  #匹配模板
  def replace(x,y):
    redency_part = x[redency_part_slice]
    unredency_part = x[unredency_part_slice]
    redency_part_result = torch.unsqueeze(target_op(redency_part,y),0).expand(batch_size,-1,-1)
    unredency_part_result = target_op(unredency_part,y)
    return torch.concat((redency_part_result,unredency_part_result),1)
  
  return pattern,replace,_match

def gen_pattern_replace_and_matcher_for_reduce_sum(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  target_node_name
                                                ):
  env  = utils.get_env(traced)
  target_node = env[target_node_name]
  target_args = target_node.args
  target_op = torch.sum
  def _match(match,ori,pat):
    target_node = None
    for node in pat.nodes:
      if node.op == "call_function" and node.target == target_op:
        target_node = node
    return match.nodes_map[target_node].name == target_node_name
  
  def pattern(x):
    return target_op(x,target_args[1])
  
  def replace(x):
    redency_part = x[redency_part_slice]
    unredency_part = x[unredency_part_slice]
    redency_part_result = target_op(torch.unsqueeze(redency_part,0),target_args[1])
    unredency_part_result = target_op(unredency_part,target_args[1])
    return redency_part_result + unredency_part_result
  
  return pattern,replace,_match

def gen_pattern_replace_and_matcher_for_linear(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  target_node_name
                                                ):
  env  = utils.get_env(traced)
  target_node = env[target_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node_name)
  def _match(match,ori,pat):
    selected_node = None
    for node in pat.nodes:
      if node.op == "call_module" :
        selected_node = node
    return match.nodes_map[selected_node].name == target_node_name
  
  class PatternClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = torch.nn.Linear(1, 1)
    def forward(self,x):
      return self.mlp(x)
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      target_weight = target_node_mod.weight
      target_bias = target_node_mod.bias
      redency_weight = target_weight[:,redency_part_slice[1]]
      self.redency_linear = torch.nn.Linear(redency_weight.shape[0],redency_weight.shape[1])
      self.redency_linear.weight.data = redency_weight
      self.redency_linear.bias.data = target_bias
      
      unredency_weight = target_weight[:,unredency_part_slice[1]]
      self.unredency_linear = torch.nn.Linear(unredency_weight.shape[0],unredency_weight.shape[1],bias=False)
      self.unredency_linear.weight.data = unredency_weight

    def forward(self,x):
      redency_part = x[redency_part_slice]
      unredency_part = x[unredency_part_slice]
      return self.redency_linear(redency_part) + self.unredency_linear(unredency_part)
    
  
  return PatternClass(),ReplacementClass(),_match

def gen_pattern_replace_and_matcher_for_LR(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  target_node_name,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

  class PatternClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.fc = torch.nn.Embedding(1, 1)
      self.bias1 = torch.nn.Parameter(torch.zeros((1,)))
      self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))

    def forward(self,x):
      x = x + self.offsets
      return torch.sum(self.fc(x), dim=1) + self.bias1 
  def _match(match,ori,pat):
    return True 
  env  = utils.get_env(traced)
  target_node = env[target_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node_name,"_")
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  fc_node = node_map[pattern_env['fc']]
  fc_node_module = utils.get_target_mod(traced,fc_node.target)
  
  bias_node = node_map[pattern_env['bias1']]
  bias_val = utils.get_target_mod(traced,bias_node.target)
  
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      embedding_config = fc_node_module.weight.data.shape
      self.fc = nn.Embedding(embedding_config[0],embedding_config[1])
      # self.fc = fc_node_module
      self.fc.weight.data.copy_(fc_node_module.weight.data)
      self.bias = bias_val

    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      redency_sum = torch.sum(self.fc(redency_part)) + self.bias
      unredency_sum = torch.sum(self.fc(unredency_part),dim=1)
      return redency_sum + unredency_sum
      # return unredency_sum
    
  
  return pattern,ReplacementClass(),_match



def gen_pattern_replace_and_matcher_for_FM(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  target_node_name,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

  class PatternClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.embed = torch.nn.Embedding(1, 1)
      self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))

    def forward(self,x):
      x = x + self.offsets
      x = self.embed(x)
      square_of_sum = torch.sum(x, dim=1) ** 2
      sum_of_square = torch.sum(x ** 2, dim=1)
      ix = square_of_sum - sum_of_square    
      ix = torch.sum(ix, dim=1, keepdim=True)  
      return 0.5 * ix
  def _match(match,ori,pat):
    return True 
  # env  = utils.get_env(traced)
  # target_node = env[target_node_name]
  # target_node_mod = utils.get_target_mod(traced,target_node_name,"_")
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  embed_node = node_map[pattern_env['embed']]
  embed_node_module = utils.get_target_mod(traced,embed_node.target)
  

  
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      embedding_config = embed_node_module.weight.data.shape
      self.embed = nn.Embedding(embedding_config[0],embedding_config[1])
      self.embed.weight.data.copy_(embed_node_module.weight.data)
      
      
    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      redency_embed = self.embed(redency_part)
      unredency_embed = self.embed(unredency_part)
      
      redency_embed_sum = torch.sum(redency_embed,dim=0)
      unredency_embed_sum = torch.sum(unredency_embed,dim=1)
      square_of_sum = (redency_embed_sum + unredency_embed_sum) ** 2
      
      redency_embed_square_sum = torch.sum(redency_embed ** 2,dim=0)
      unredency_embed_square_sum = torch.sum(unredency_embed ** 2,dim=1)
      sum_of_square = redency_embed_square_sum + unredency_embed_square_sum
      ix = square_of_sum - sum_of_square
      ix = torch.sum(ix,dim = 1,keepdim=True)
      
      
      return 0.5 * ix
    
  
  return pattern,ReplacementClass(),_match

def gen_pattern_replace_and_matcher_for_MLP(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  key_node_name,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


  def _match(match,ori,pat):
    return True 
  env  = utils.get_env(traced)
  target_node = env[key_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node.target)
  shape_info = target_node_mod.weight.data.shape
  class PatternClass(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.embed = torch.nn.Embedding(1, 1)
          self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))
          self.embed_output_dim = shape_info[1]
          self.mlp = nn.Linear(shape_info[0],shape_info[1])


      def forward(self,x):
          x = x + self.offsets
          x = self.embed(x).view(-1,self.embed_output_dim)
          return self.mlp(x)
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  embed_node = node_map[pattern_env['embed']]
  embed_node_module = utils.get_target_mod(traced,embed_node.target)
  
  linear_node = node_map[pattern_env['mlp']]
  linear_node_module = utils.get_target_mod(traced,linear_node.target)
  linear_node_weight = linear_node_module.weight.data
  linear_node_bias = linear_node_module.bias.data
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      self.embed = embed_node_module
      self.embed_dim = self.embed.weight.data.shape[1]
      self.redency_weight_len = self.embed_dim * redency_part_slice[1].stop
      redency_weight = linear_node_weight[:,:self.redency_weight_len]
      unredency_weight = linear_node_weight[:,self.redency_weight_len:]
      self.unredency_weight_len = unredency_weight.shape[1]
      self.redency_linear = nn.Linear(redency_weight.shape[1],redency_weight.shape[0])
      self.redency_linear.weight.data.copy_(redency_weight)
      self.redency_linear.bias.data.copy_(linear_node_bias)

      self.unredency_linear = nn.Linear(unredency_weight.shape[1],unredency_weight.shape[0],bias=False)
      self.unredency_linear.weight.data.copy_(unredency_weight)

      

    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      return self.redency_linear(self.embed(redency_part).view(-1,self.redency_weight_len)) + self.unredency_linear(self.embed(unredency_part).view(-1,self.unredency_weight_len))
      # return unredency_sum
    
  
  return pattern,ReplacementClass(),_match


def gen_pattern_replace_and_matcher_for_loop_pnn(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  embed_node_name,getitem_node_names,num_field,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


  def _match(match,ori,pat):
    return True 
  env  = utils.get_env(traced)
  target_node = env[embed_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node.target)
  shape_info = target_node_mod.weight.data.shape
  getitem_node_args = [env[i].args[1] for i in getitem_node_names]
  class PatternClass(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.embed = torch.nn.Embedding(1, 1)
          self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))
          self.embed_output_dim = shape_info[1] * num_field
          self.mlp = nn.Linear(shape_info[0],shape_info[1])

      def pn(self,x):
         return torch.sum(x[getitem_node_args[0]] * x[getitem_node_args[1]], dim = 2)

      def forward(self,x):
          x = self.embed(x + self.offsets)
          cross_term = self.pn(x)
          x = torch.cat([x.view(-1, self.embed_output_dim), cross_term], dim=1)
          return self.mlp(x)
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  embed_node = node_map[pattern_env['embed']]
  embed_node_module = utils.get_target_mod(traced,embed_node.target)
  
  linear_node = node_map[pattern_env['mlp']]
  linear_node_module = utils.get_target_mod(traced,linear_node.target)
  linear_node_weight = linear_node_module.weight.data
  linear_node_bias = linear_node_module.bias.data
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      self.num_fields = num_field
      self.num_prefix = redency_part_slice[1].stop
      self.ori_linear_shape = linear_node_weight.shape
      self.num_sufix = self.num_fields - self.num_prefix
      self.total = self.num_fields * (self.num_fields - 1) // 2
      self.redency_cross_part_total = self.num_prefix * (self.num_prefix - 1) // 2
      self.unredency_cross_part_total = self.num_sufix * (self.num_sufix - 1) // 2
      self.rest_cross_part_total = self.total - self.redency_cross_part_total - self.unredency_cross_part_total
      # 提取对应的权重参数逻辑有些复杂，先mock
      self.embed = embed_node_module
      self.embed_dim = self.embed.weight.data.shape[1]

      # cross part linear
      self.redency_cross_part_linear = nn.Linear(self.redency_cross_part_total,self.ori_linear_shape[0],bias = True)
      self.unredency_cross_part_linear = nn.Linear(self.unredency_cross_part_total,self.ori_linear_shape[0],bias = False)
      self.mixed_cross_part_linear = nn.Linear(self.rest_cross_part_total,self.ori_linear_shape[0],bias = False)
      
      # embed part linear 
      self.redency_linear = nn.Linear(self.embed_dim * self.num_prefix,self.ori_linear_shape[0],bias = False)
      self.unredency_linear = nn.Linear(self.embed_dim * self.num_sufix,self.ori_linear_shape[0],bias = False)

    def pn(self,reducey_x,unredency_x):
       redency_x_row, reducey_x_col = list(),list()
       unredency_x_row, unredency_x_col = list(), list()
       mixed_x_row, mixed_x_col = list(), list()
       prefix = self.num_prefix
       sufix = self.num_sufix
       for i in range(prefix - 1):
          for j in range(i+1,prefix):
             redency_x_row.append(i),reducey_x_col.append(j)
       for i in range(sufix - 1):
          for j in range(i + 1,sufix):
             unredency_x_row.append(i),unredency_x_col.append(j)

       for i in range(prefix):
          for j in range(sufix):
             mixed_x_row.append(i),mixed_x_col.append(j)
       return torch.sum((reducey_x[redency_x_row] * reducey_x[reducey_x_col]).unsqueeze(0),dim = 2),\
              torch.sum(unredency_x[:,unredency_x_row] * unredency_x[:,unredency_x_col],dim = 2),\
              torch.sum(reducey_x[mixed_x_row] * unredency_x[:,mixed_x_col],dim = 2)


    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      redency_part_embed = self.embed(redency_part)
      unredency_part_embed = self.embed(unredency_part)
      redency_part_pn, unredency_part_pn, mixed_part_pn = self.pn(redency_part_embed,unredency_part_embed)
      
      
      return self.redency_linear(self.embed(redency_part).view(-1,self.embed_dim * self.num_prefix)) + self.unredency_linear(self.embed(unredency_part).view(-1,self.embed_dim * self.num_sufix)) +\
         self.redency_cross_part_linear(redency_part_pn) + self.mixed_cross_part_linear(mixed_part_pn) + self.unredency_cross_part_linear(unredency_part_pn)
      # return unredency_sum
    
  
  return pattern,ReplacementClass(),_match



def gen_pattern_replace_and_matcher_for_outer_pnn(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  embed_node_name,getitem_node_names,num_field,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


  def _match(match,ori,pat):
    return True 
  env  = utils.get_env(traced)
  target_node = env[embed_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node.target)
  shape_info = target_node_mod.weight.data.shape
  getitem_node_args = [env[i].args[1] for i in getitem_node_names]
  class PatternClass(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.embed = torch.nn.Embedding(1, 1)
          self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))
          self.embed_output_dim = shape_info[1] * num_field
          self.mlp = nn.Linear(shape_info[0],shape_info[1])
          self.kernel = torch.nn.Parameter(torch.zeros((1,1,1)))

      def pn(self,x):
         p , q = x[getitem_node_args[0]] , x[getitem_node_args[1]]
         kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
         return torch.sum(kp * q, -1)

      def forward(self,x):
          x = self.embed(x + self.offsets)
          cross_term = self.pn(x)
          x = torch.cat([x.view(-1, self.embed_output_dim), cross_term], dim=1)
          return self.mlp(x)
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  embed_node = node_map[pattern_env['embed']]
  embed_node_module = utils.get_target_mod(traced,embed_node.target)
  
  kernel_node = node_map[pattern_env['kernel']]
  kernel_node_module = utils.get_target_mod(traced,kernel_node.target)
  
  linear_node = node_map[pattern_env['mlp']]
  linear_node_module = utils.get_target_mod(traced,linear_node.target)
  linear_node_weight = linear_node_module.weight.data
  linear_node_bias = linear_node_module.bias.data
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      self.num_fields = num_field
      self.num_prefix = redency_part_slice[1].stop
      self.ori_linear_shape = linear_node_weight.shape
      self.num_sufix = self.num_fields - self.num_prefix
      self.total = self.num_fields * (self.num_fields - 1) // 2
      self.redency_cross_part_total = self.num_prefix * (self.num_prefix - 1) // 2
      self.unredency_cross_part_total = self.num_sufix * (self.num_sufix - 1) // 2
      self.rest_cross_part_total = self.total - self.redency_cross_part_total - self.unredency_cross_part_total
      # 提取对应的权重参数逻辑有些复杂，先mock
      self.embed = embed_node_module
      self.embed_dim = self.embed.weight.data.shape[1]

      # cross part linear
      self.redency_cross_part_linear = nn.Linear(self.redency_cross_part_total,self.ori_linear_shape[0],bias = True)
      self.unredency_cross_part_linear = nn.Linear(self.unredency_cross_part_total,self.ori_linear_shape[0],bias = False)
      self.mixed_cross_part_linear = nn.Linear(self.rest_cross_part_total,self.ori_linear_shape[0],bias = False)
      
      # embed part linear 
      self.redency_linear = nn.Linear(self.embed_dim * self.num_prefix,self.ori_linear_shape[0],bias = False)
      self.unredency_linear = nn.Linear(self.embed_dim * self.num_sufix,self.ori_linear_shape[0],bias = False)

      # kernel 同理，先mock
      
      self.kernel = kernel_node_module
      self.redency_kernel = torch.nn.Parameter(self.kernel[:,:self.redency_cross_part_total])
      self.unredency_kernel = torch.nn.Parameter(self.kernel[:,:self.unredency_cross_part_total])
      self.mixed_kernel = torch.nn.Parameter(self.kernel[:,:self.rest_cross_part_total])
      

    def pn(self,reducey_x,unredency_x):
      redency_x_row, reducey_x_col = list(),list()
      unredency_x_row, unredency_x_col = list(), list()
      mixed_x_row, mixed_x_col = list(), list()
      prefix = self.num_prefix
      sufix = self.num_sufix
      for i in range(prefix - 1):
         for j in range(i+1,prefix):
            redency_x_row.append(i),reducey_x_col.append(j)
      for i in range(sufix - 1):
         for j in range(i + 1,sufix):
            unredency_x_row.append(i),unredency_x_col.append(j)

      for i in range(prefix):
         for j in range(sufix):
            mixed_x_row.append(i),mixed_x_col.append(j)
      kp_redency = torch.sum(reducey_x[redency_x_row].unsqueeze(0).unsqueeze(1)* self.redency_kernel, dim=-1).permute(0, 2, 1)
      kp_unredency = torch.sum(unredency_x[:,unredency_x_row].unsqueeze(1)* self.unredency_kernel, dim=-1).permute(0, 2, 1)
      kp_mixed = torch.sum(reducey_x[mixed_x_row].unsqueeze(0).unsqueeze(1)* self.mixed_kernel, dim=-1).permute(0, 2, 1)
      return torch.sum((kp_redency * reducey_x[reducey_x_col]).unsqueeze(0),dim = -1),\
            torch.sum(kp_unredency * unredency_x[:,unredency_x_col],dim = -1),\
            torch.sum(kp_mixed * unredency_x[:,mixed_x_col],dim = -1)


    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      redency_part_embed = self.embed(redency_part)
      unredency_part_embed = self.embed(unredency_part)
      redency_part_pn, unredency_part_pn, mixed_part_pn = self.pn(redency_part_embed,unredency_part_embed)
      
      
      return self.redency_linear(self.embed(redency_part).view(-1,self.embed_dim * self.num_prefix)) + self.unredency_linear(self.embed(unredency_part).view(-1,self.embed_dim * self.num_sufix)) +\
         self.redency_cross_part_linear(redency_part_pn) + self.mixed_cross_part_linear(mixed_part_pn) + self.unredency_cross_part_linear(unredency_part_pn)
      # return unredency_sum
    
  
  return pattern,ReplacementClass(),_match

def gen_pattern_replace_and_matcher_for_afm(traced,
                                                  redency_part_slice,unredency_part_slice,
                                                  embed_node_name,getitem_node_names,num_field,batch = 4096,match_func = None
                                                ):
  from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


  def _match(match,ori,pat):
    return True 
  env  = utils.get_env(traced)
  target_node = env[embed_node_name]
  target_node_mod = utils.get_target_mod(traced,target_node.target)
  shape_info = target_node_mod.weight.data.shape
  getitem_node_args = [env[i].args[1] for i in getitem_node_names]
  class PatternClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(1, 1)
        self.offsets = torch.as_tensor(np.array((0, *np.cumsum([10,10])[:-1]), dtype=np.int64))
        self.embed_output_dim = 176
        self.atten = nn.Linear(1,1)
        self.projection = torch.nn.Linear(1, 1)
        self.fc = torch.nn.Linear(1, 1)
        # self.num_fields = 22

    def inner_product(self,x):
      return x[getitem_node_args[0]] * x[getitem_node_args[1]]

    def forward(self,x):
        x = self.embed(x + self.offsets)
        inner_product = self.inner_product(x)
        # return inner_product
        attn_scores = F.relu(self.atten(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        return self.fc(torch.sum(attn_scores * inner_product, dim=1))
  pattern = PatternClass()  
  pattern_trace = symbolic_trace(pattern)
  pattern_graph = pattern_trace.graph
  original_graph = traced.graph
  matcher =  SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True)
  _matches = matcher.match(original_graph)
  match_filters = [_match if match_func is None else match_func]
  _matches = [
      m for m in _matches
      if all(match_filter(m, original_graph, pattern_graph)
              for match_filter in match_filters)
  ]  
  # 因为在过滤器中做了限制应该只有一个符合要求的
  _matched = _matches[0]
  pattern_env = utils.get_env(pattern_trace)
  node_map = _matched.nodes_map
  pn = pattern_env['offsets']
  offsets_node = node_map[pn]
  offsets_val = utils.get_target_mod(traced,offsets_node.target)
  
  embed_node = node_map[pattern_env['embed']]
  embed_node_module = utils.get_target_mod(traced,embed_node.target)

  
  atten_node = node_map[pattern_env['atten']]
  atten_node_module = utils.get_target_mod(traced,atten_node.target)
  
  projection_node = node_map[pattern_env['projection']]
  projection_node_module = utils.get_target_mod(traced,projection_node.target)
  
  fc_node = node_map[pattern_env['fc']]
  fc_node_module = utils.get_target_mod(traced,fc_node.target)
  
  
  class ReplacementClass(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.redency_offset = torch.as_tensor(np.array(offsets_val[redency_part_slice[1]], dtype=np.int64))
      self.unredency_offset = torch.as_tensor(np.array(offsets_val[unredency_part_slice[1]], dtype=np.int64))
      self.num_fields = num_field
      self.num_prefix = redency_part_slice[1].stop
      self.num_sufix = self.num_fields - self.num_prefix
      self.total = self.num_fields * (self.num_fields - 1) // 2
      self.redency_cross_part_total = self.num_prefix * (self.num_prefix - 1) // 2
      self.unredency_cross_part_total = self.num_sufix * (self.num_sufix - 1) // 2
      self.rest_cross_part_total = self.total - self.redency_cross_part_total - self.unredency_cross_part_total
      # 提取对应的权重参数逻辑有些复杂，先mock
      self.embed = embed_node_module
      self.embed_dim = self.embed.weight.data.shape[1]

      self.attention = atten_node_module
      self.projection = projection_node_module
      self.fc = fc_node_module

    def pn(self,reducey_x,unredency_x):
       redency_x_row, reducey_x_col = list(),list()
       unredency_x_row, unredency_x_col = list(), list()
       mixed_x_row, mixed_x_col = list(), list()
       prefix = self.num_prefix
       sufix = self.num_sufix
       for i in range(prefix - 1):
          for j in range(i+1,prefix):
             redency_x_row.append(i),reducey_x_col.append(j)
       for i in range(sufix - 1):
          for j in range(i + 1,sufix):
             unredency_x_row.append(i),unredency_x_col.append(j)

       for i in range(prefix):
          for j in range(sufix):
             mixed_x_row.append(i),mixed_x_col.append(j)
       return (reducey_x[redency_x_row] * reducey_x[reducey_x_col]).unsqueeze(0),\
              unredency_x[:,unredency_x_row] * unredency_x[:,unredency_x_col],\
              reducey_x[mixed_x_row] * unredency_x[:,mixed_x_col]


    def forward(self,x):
      redency_part = x[redency_part_slice] + self.redency_offset
      unredency_part = x[unredency_part_slice] + self.unredency_offset
      redency_part_embed = self.embed(redency_part)
      unredency_part_embed = self.embed(unredency_part)
      redency_part_pn, unredency_part_pn, mixed_part_pn = self.pn(redency_part_embed,unredency_part_embed)
      redency_attn_scores = self.projection(F.relu(self.attention(redency_part_pn)))
      unredency_attn_scores = self.projection(F.relu(self.attention(unredency_part_pn)))
      mixed_rest_attn_scores = self.projection(F.relu(self.attention(mixed_part_pn)))      
      redency_attn_scores = redency_attn_scores.repeat([batch,1,1])
      attn_scores  = torch.concat([redency_attn_scores,unredency_attn_scores,mixed_rest_attn_scores],dim = 1)
      attn_scores = F.softmax(attn_scores, dim=1)
      # return unredency_sum
      redency_attn_scores = attn_scores[0,:self.redency_cross_part_total,:]
      unredency_attn_scores = attn_scores[:,self.redency_cross_part_total:self.redency_cross_part_total+self.unredency_cross_part_total,:]
      mixed_rest_attn_scores = attn_scores[:,self.redency_cross_part_total+self.unredency_cross_part_total:,]
      attn_output = torch.sum(redency_attn_scores * redency_part_pn, dim=1) + torch.sum(unredency_attn_scores * unredency_part_pn, dim=1) + torch.sum(mixed_rest_attn_scores * mixed_part_pn, dim=1)
      return self.fc(attn_output)
  return pattern,ReplacementClass(),_match