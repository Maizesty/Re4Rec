from abc import ABC, abstractmethod
import numpy as np
import onnx_graphsurgeon as gs
from collections import deque
from graphviz import Digraph
import pprint
from io import StringIO

class TensorAnnotation:
  def __init__(self,var,slices = None) -> None:
    self.var = var
    if isinstance(self.var,gs.ir.tensor.Constant):
      self.slices = [slice(None,None) for _ in var.shape]
    else:
      self.slices = slices
  
  @property
  def tensorLength(self):
    return len(self.slices)
  
  @property
  def startList(self):
    return [i.start for i in self.slices]
  
  @property
  def stopList(self):
    return [i.stop for i in self.slices]
  
  @property
  def tensorType(self):
    if isinstance(self.var,gs.ir.tensor.Variable):
      return "Variable"
    else:
      return "Constant"
  
  @property
  def hasRedundancy(self):
    if self.slices is None or len(self.slices) == 0:
      return False
    for i in range(len(self.var.shape)):
      if self.isFullRedundancyAtDim(i):
        return True
    return False
  def __eq__(self, other):
    if isinstance(other, TensorAnnotation):
      if self.slices is None or other.slices is None:
        return False
      if len(self.slices) == len(other.slices):
        return all(x == y for x, y in zip(self.slices, other.slices))
      else:
        return False
    return False
  
  def isFullRedundancyAtDim(self,dim):
    curSlice = self.slices[dim]
    start = curSlice.start
    stop = curSlice.stop
    shapeAtDim = self.var.shape[dim]
    # 判断一下是否在dim这个维度全冗余
    # 最简单的一种情况是 start stop都是None
    if start is None and stop is None:
      return True
    if isinstance(shapeAtDim, str): 
      if start == 0 and stop is None:
        return True
      else :
        return False
    return (start == 0 and (stop is None or stop == shapeAtDim)) or (start is None and stop == shapeAtDim)
    
    



class ENode:
  def __init__(self,node):
    self.node = node[0]
    # 节点类型 如果是0 是input， 1是正常节点， 2是output
    self._nodeType = node[1]
    # 记录前驱节点
    self.precursor = set()
    # 记录后继节点
    self.successor = set()
    # 这里存储一个节点需要的输入，
    self.inputs = dict()
    self._output = None
    # 冗余类型 0：未知，1：无冗余，2：有冗余可以传播，3：有冗余但是不能传播
    self.redundancyType = 0
  
  @property
  def nodeType(self):
    if self._nodeType == 0:
      return "Input"
    if self._nodeType == 1:
      return self.node.op
    if self._nodeType == 2:
      return "Output"
  
  @property
  def nodeTypeVal(self):
    return self._nodeType

  @property
  def attr(self):
    if self._nodeType != 1:
      return None
    return self.node.attrs
  
  @property
  def output(self):
    return self._output
  
  @output.setter
  def output(self, value):
    self._output = value
  
  @property
  def outputName(self):
    if self._nodeType != 1:
      return self.node.name
    else:
      return self.node.outputs[0].name
  
  @property
  def outputs(self):
    if self._nodeType == 1:
      return self.node.outputs
    else:
      return [self.node]
  
  @property
  def name(self):
    return self.node.name
  
  @property
  def isInputPrepare(self):
    if self._nodeType == 0:
      return len(self.inputs) == 1
    elif self._nodeType == 1:
      return len(self.inputs) == len(self.node.inputs)
    elif self._nodeType == 2:
      return len(self.inputs) == 1
  
  def addInput(self,name,annotation):
    self.inputs[name] = annotation
  
  def addPrecursor(self,nodeName):
    self.precursor.add(nodeName)
    
  def addSuccessor(self,nodeName):
    self.successor.add(nodeName)
    
    
  # 访问函数，目前用于计算冗余  
  def vis(self,func):
    return func(self)


class EGraph:
  def __init__(self,graph):
    self.graph = graph
    self.output_dict = dict() # 记录某个输出变量和其对应的消费者节点的映射
    self.producer_out_dict = dict() # 记录输出变量名字和其对应生产节点之间的映射
    self.nodes = dict() # 建立一个节点名和节点之间的映射，方便后面查找
    self.constants = dict() # 建立一个常量名和常量的映射，方便后面查找
    self.var = dict() # 建立一个变量名和变量之间的映射，方便后面查找
    self.inputs = dict()
    self.outputs = []
    self.Enodes =  dict() # 建立一个节点名和ENode节点之间的映射，方便后面查找
    self.transformToEGraph(graph = self.graph)

  def draw(self):
    dot = Digraph(comment='graph')
    for n,node in self.Enodes.items():
      color = 'Orange'
      node_name = '?'
      if node.nodeTypeVal == 0:
        node_name = f"input : {node.name}"
      elif node.nodeTypeVal == 1:
        node_name = node.name
      elif node.nodeTypeVal == 2:
        node_name = f"output : {node.name}"
      dot.node(n,node_name,style='filled',
                    shape='box',
                    align='left',
                    fontsize='10',
                    ranksep='0.1',
                    height='0.2',
                    fontname='monospace',fillcolor=color)
    for name,node in self.Enodes.items():
      for outputName,successorName in node.successor:
        dot.edge(name,successorName,outputName,fontsize='10')
    return dot
  
  def drawRedundancy(self):
    plot_dict = {0:'Grey',1:'lightcyan',2:'Orange',3:'Green'}
    dot = Digraph(comment='graph')
    for n,node in self.Enodes.items():
      color = plot_dict[node.redundancyType]
      node_name = '?'
      if node.nodeTypeVal == 0:
        node_name = f"input : {node.name}"
      elif node.nodeTypeVal == 1:
        node_name = node.name
      elif node.nodeTypeVal == 2:
        node_name = f"output : {node.name}"
      
      dot.node(n,node_name,style='filled',
                    shape='box',
                    align='left',
                    fontsize='10',
                    ranksep='0.1',
                    height='0.2',
                    fontname='monospace',fillcolor=color)
    for name,node in self.Enodes.items():
      for outputName,successorName in node.successor:
        dot.edge(name,successorName,outputName,fontsize='10')
    return dot    
      
      
  def transformToEGraph(self,graph):
    for i in  graph.inputs:
      # onnx的输入有时候可能有一些常量，这些常量的作用是给某些算子用的，并不是真实的输入，真实的输入应该是gs.ir.tensor.Variable类型的
      if isinstance(i,gs.ir.tensor.Variable):
        self.output_dict[i.name] = [] 
        self.producer_out_dict[i.name]  = i.name # 这里特殊处理，因为Onnx就没有专门的input node说法，这里就先假设一个节点他的名字和这个输入一模一样
        self.nodes[i.name] = (i,0) # 先将他标记为 (i,0)后续用于建立ENode用
        inputENode = ENode((i,0))
        self.inputs[i.name] = inputENode
        self.Enodes[i.name] = inputENode
        self.var[i.name] = i # 把变量先存起来
      else:
        self.constants[i.name] = i # 把常量先存起来
    # 先遍历找到所有输出变量，再构建输入变量
    for node in graph.nodes:
      curENode = ENode((node,1))
      self.Enodes[node.name] = curENode
      self.nodes[node.name] = (node,1) # 在Graph中的node就是真实的节点
      for i in node.outputs:
        self.output_dict[i.name] = []
        self.producer_out_dict[i.name] = node.name
        self.var[i.name] = i
    for node in graph.nodes:
      curENode = self.Enodes[node.name]
      for i in node.inputs:
        if i.name in self.output_dict:
          curENode.addPrecursor(((i.name,self.producer_out_dict[i.name])))
          precursorNode = self.Enodes[self.producer_out_dict[i.name]]
          precursorNode.addSuccessor((i.name,curENode.name))
          self.output_dict[i.name].append(node.name) # 构建输入变量和他下游算子的映射
        else:
          self.constants[i.name] = i
          curENode.addInput(i.name,TensorAnnotation(i))
    for i in graph.outputs:
      self.nodes[i.name] = (i,2)
      outputNode = ENode((i,2))
      self.outputs.append(outputNode)
      self.Enodes[i.name] = outputNode
      outputNode.addPrecursor((i.name,self.producer_out_dict[i.name]))
      precursorNode = self.Enodes[self.producer_out_dict[i.name]]
      precursorNode.addSuccessor((i.name,outputNode.name))
      self.output_dict[i.name].append(i.name)
    
  def redundancyCal(self,inputDescDict):
    
    queue = deque()
    vis = set()
    for inputName,inputNode in self.inputs.items():
      assert inputName in inputDescDict
      inputNode.addInput(inputName,TensorAnnotation(inputNode,inputDescDict[inputName]))
      queue.append(inputNode)
    
    while len(queue) > 0:
      size = len(queue)
      while size > 0:
        size = size - 1
        curNode = queue.popleft()
        if curNode.name not in vis:
          vis.add(curNode.name)
          self.redundancyPass(curNode)
          for _,childName in curNode.successor:
            childNode = self.Enodes[childName]
            if childNode.isInputPrepare:
              queue.append(childNode)
                
          
    
    
    
  def redundancyPass(self,node):
    # 找对应的函数
    # 计算
    # 将计算结果传递到子节点
    passMapper = {
      
      'Input' : self.inputRedundancyPass,
      'Relu' : self.unaryElementwiseOpRedundancyPass,
      'Pow' : self.unaryElementwiseOpRedundancyPass,
      'Add' : self.binaryElementwiseOpRedundancyPass,
    }
    func = self.defaultOpRedundancyPass
    if node.nodeType in passMapper:
      func = passMapper[node.nodeType]
    func(node)
    outputName = node.outputName
    for _,childName in node.successor:
      childNode = self.Enodes[childName]
      childNode.addInput(outputName,node.output)
      
      
  def inputRedundancyPass(self,node):
    inputs = tuple(node.inputs.values())
    a = inputs[0]
    outputVar = node.outputs[0]
    slices = a.slices
    node.output = TensorAnnotation(outputVar,slices=slices)
    if node.output.hasRedundancy:
      node.redundancyType = 2
    else :
      node.redundancyType = 1
      
    
  def unaryElementwiseOpRedundancyPass(self,node):
    
    # 单操作符的elementWise算子的冗余状态完全和其输入一致

    inputs = tuple(node.inputs.values())
    a = inputs[0]
    outputVar = node.outputs[0]
    slices = a.slices
    node.output = TensorAnnotation(outputVar,slices=slices)
    if node.output.hasRedundancy:
      node.redundancyType = 2
    else :
      node.redundancyType = 1
      
    
  def binaryElementwiseOpRedundancyPass(self,node):
    # 现在不考虑两个变量时有广播的情况
    inputs = tuple(node.inputs.values())
    a = inputs[0]
    b = inputs[1]
    outputVar = node.outputs[0]
    # 先只考虑 常数 + 变量的情况
    if a.tensorType == 'Constant' and b.tensorType == 'Variable':
      slices = b.slices
      node.output = TensorAnnotation(outputVar,slices=slices)
      if node.output.hasRedundancy:
        node.redundancyType = 2
      else :
        node.redundancyType = 1
      return 
    if b.tensorType == 'Constant' and a.tensorType == 'Variable':
      slices = a.slices
      node.output = TensorAnnotation(outputVar,slices=slices)
      if node.output.hasRedundancy:
        node.redundancyType = 2
      else :
        node.redundancyType = 1
      return 
    if a.tensorType == 'Variable' and b.tensorType == 'Variable':
      if a == b:
        node.output = TensorAnnotation(outputVar,slices=a.slices)
        if node.output.hasRedundancy:
          node.redundancyType = 2
        else :
          node.redundancyType = 1
      else :
        node.output =  TensorAnnotation(outputVar)
        node.redundancyType = 1
    
  
  # def gatherOpRedundancyPass(self,node):
    
  
  def defaultOpRedundancyPass(self,node):
    outputVar = node.outputs[0]
    node.output = TensorAnnotation(outputVar)
    node.redundancyType = 0