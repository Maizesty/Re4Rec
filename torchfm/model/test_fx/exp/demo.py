import sys
sys.path.append('../../')
sys.path.append('../')
import time
import wd
from graphviz import Digraph
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fx as fx
name_list =[
  'embedding_embedding',
  'view',
  'mlp_mlp_0'
]
def draw(graph):
    dot = Digraph(comment='graph', graph_attr={'rankdir': 'LR'},format='png')
    root = graph._root
    cur = root._next
    while cur is not root:
        if not cur._erased and cur.op != "get_attr":
            dot.node(cur.name,cur.name,style='filled',
                    shape='box',
                    align='left',
                    fontsize='10',
                    ranksep='0.1',
                    height='0.2',
                    fontname='monospace',fillcolor='White' if cur.name not in name_list else 'Orange')
            for arg in cur.args:
                if isinstance(arg,torch.fx.node.Node) and arg.op != "get_attr":
                    dot.edge(arg.name,cur.name)
                if isinstance(arg,list):
                    for item in arg:
                        dot.edge(item.name,cur.name)
        cur = cur._next
    return dot
  
wdl_model_ori = wd.WideAndDeepModel([100 for i in range(110)],32,[400,400,400],0.1)
wdl_model = fx.symbolic_trace(wdl_model_ori)
ga = wdl_model.graph
dot = draw(ga)
dot.render('round-table.gv',format='jpg', view=True)  
