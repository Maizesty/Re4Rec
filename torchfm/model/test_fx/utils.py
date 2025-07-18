import torch.fx as fx
import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter
from torch.fx.node import map_arg
from graphviz import Digraph
import torch
from tabulate import tabulate
from torch.fx import (
    Node,
    Graph,
)
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher


def new_nodes_are_equal(self, pn: Node, gn: Node) -> bool:
    # if exact match for placeholder is not required, then use placeholder as a wildcard
    if not self.match_placeholder and pn.op == "placeholder":
        return True

    if pn.op == gn.op:
        if pn.op == "placeholder" or pn.op == "output":
            return True
        elif pn.op == "get_attr":
            return self._match_attributes(pn, gn)
        elif pn.op == "call_module":
          pn_module = get_target_mod(pn.graph.owning_module,pn.target)
          gn_module = get_target_mod(gn.graph.owning_module,gn.target)
          return type(pn_module) == type(gn_module)
        return pn.target == gn.target
    return False



def draw(graph):
    dot = Digraph(comment='graph')
    root = graph._root
    cur = root._next
    while cur is not root:
        if not cur._erased:
            dot.node(cur.name,cur.name,style='filled',
                    shape='box',
                    align='left',
                    fontsize='10',
                    ranksep='0.1',
                    height='0.2',
                    fontname='monospace',fillcolor='Orange')
            for arg in cur.args:
                if isinstance(arg,torch.fx.node.Node):
                    dot.edge(arg.name,cur.name)
                if isinstance(arg,list):
                    for item in arg:
                        dot.edge(item.name,cur.name)
        cur = cur._next
    return dot
  
def get_successors_map(traced):
  graph = traced.graph
  map = dict()
  for node in graph.nodes:
    map[node.name] = []
  
  for node in graph.nodes:
    args = node.args
    for pre in args:
      if hasattr(pre,'name') and pre.name in map:
        map[pre.name].append(node.name)
  return map

def get_env(traced):
  graph = traced.graph
  env = {}
  for node in graph.nodes:
    env[node.name] = node
  return env


def get_target_mod(model,target_name,split_char='.'):
  target_name_list = target_name.split(split_char)
  mod = model
  for name in target_name_list:
    mod = getattr(mod,name)
  return mod


def _print_tabular(traced,mod):
  graph = traced.graph
  node_specs = []
  for n in graph.nodes:
    true_method = None
    if n.op == 'call_module':
      true_method = str(type(get_target_mod(mod,n.target)))
    node_specs.append([n.op,n.name,n.target,true_method,n.args,n.kwargs])
  print(tabulate(node_specs,
    headers=['opcode', 'name', 'target','true_method', 'args', 'kwargs']
  ))


def print_tabular(mod):
  traced = fx.symbolic_trace(mod)
  _print_tabular(traced,mod)


def replace_use_with(from_node,to_node):
  to_process = [n for n in from_node.users if n != to_node]
  for use_node in to_process:
    
    def maybe_replace_node(n):
      if n == from_node:
        return to_node
      else:
        return n
    new_args = map_arg(use_node.args, maybe_replace_node)
    new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
            # assert isinstance(new_args, tuple)
            # assert isinstance(new_kwargs, dict)
    use_node._Node__update_args_kwargs(new_args, new_kwargs)
  return [n for n in to_process]

class ProfilingInterpreter(Interpreter):
    def __init__(self, mod : torch.nn.Module):
        # Rather than have the user symbolically trace their model,
        # we're going to do it in the constructor. As a result, the
        # user can pass in any ``Module`` without having to worry about
        # symbolic tracing APIs
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # We are going to store away two things here:
        #
        # 1. A list of total runtimes for ``mod``. In other words, we are
        #    storing away the time ``mod(...)`` took each time this
        #    interpreter is called.
        self.total_runtime_sec : List[float] = []
        # 2. A map from ``Node`` to a list of times (in seconds) that
        #    node took to run. This can be seen as similar to (1) but
        #    for specific sub-parts of the model.
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # Next, let's override our first method: ``run()``. ``Interpreter``'s ``run``
    # method is the top-level entrypoint for execution of the model. We will
    # want to intercept this so that we can record the total runtime of the
    # model.

    def run(self, *args) -> Any:
        # Record the time we started running the model
        t_start = time.time()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        t_end = time.time()
        # Store the total elapsed time this model execution took in the
        # ProfilingInterpreter
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # Now, let's override ``run_node``. ``Interpreter`` calls ``run_node`` each
    # time it executes a single node. We will intercept this so that we
    # can measure and record the time taken for each individual call in
    # the model.

    def run_node(self, n : torch.fx.Node) -> Any:
        # Record the time we started running the op
        t_start = time.time()
        # Run the op by delegating back into Interpreter.run_node()
        return_val = super().run_node(n)
        # Record the time we finished running the op
        t_end = time.time()
        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(t_end - t_start)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort : bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries : List[List[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)
        total_time = 0
        for node, runtimes in self.runtimes_sec.items():
          total_time += statistics.mean(runtimes)
        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = statistics.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtim
            node_summaries.append(
                [node.op, str(node), mean_runtime * 1000, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers : List[str] = [
            'Op type', 'Op', 'Average runtime (ms)', 'Pct total runtime'
        ]
        print(f"total true time {total_time * 1000} ms")
        print(f"total time: {mean_total_runtime * 1000} ms")
        return tabulate(node_summaries, headers=headers)


# hook function, replace the subgraphMatcher
ori_func = SubgraphMatcher._nodes_are_equal

SubgraphMatcher._nodes_are_equal = new_nodes_are_equal
print("replace success!")