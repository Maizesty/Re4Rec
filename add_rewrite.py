import onnx_graphsurgeon as gs
import onnx
import numpy as np
import sys
import onnx_utils


@gs.Graph.register()
def redencyPartofAdd(self,input,start,end,axes,addData,repeat,name,GatherAxes = [0]):
  # 对一个张量[batch,m,n,...],先假定只在一个维度需要提取冗余，例如 对于输入[batch,m,n],在第二个维度 0~k存在冗余
  # 则提取冗余为 RedencyX = X[:,0:k,:] ,下述三个操作是从torch中前述操作翻译为onnx以后的操作，被分解为了三步
  # 输入的张量为 [batch,m,n] 先锁定一行截取出来，
  reGather = self.gather(name+'/reGather',input,np.array(GatherAxes))
  # 将0~k截取出来，由于gather会改变维度所以先判断一下axes在gather维度前后
  if axes > GatherAxes[0]:
    axes = axes - 1
  reSlice = self.slice(name+'/reSlice',*reGather,start,end,axes)
  # 截取出来的张量为[k,n]，需要给他再添加一个维度
  reUnsqueeze = self.unsqueeze(name+'/reUnsqueeze',*reSlice,[0])
  # 冗余部分计算只计算一次
  reAdd = self.op_with_const('Add',name+'/reAdd',*reUnsqueeze,addData)
  # 重新拓展到batch维度张量
  reTile = self.tile(*reAdd,np.array(repeat))
  return reTile

@gs.Graph.register()
def unRedencyPartOfAdd(self,input,start,end,axes,addData,name):
  # 非冗余部分直接截取就可以，并不需要经过上述复杂的操作
  sliceNode = self.slice(name+'/unSlice',input,start,end,axes)
  unAdd = self.op_with_const('Add',name+'/unAdd',*sliceNode,addData)
  return unAdd
# @gs.Graph.register()
# def addRedencyRewrite(self,input,name,
#                       unRedencyStart,unRedencyEnd,unRedencyAxes,unRedencyAddData,
#                       redencyStart,redencyEnd,redencyAxes,redencyRepeat,redencyAddData,redencyGatherAxes):
#   unRedencyPart = self.unRedencyPartOfAdd(input,unRedencyStart,unRedencyEnd,unRedencyAxes,unRedencyAddData,name)
#   redencyPart = self.redencyPartofAdd(input,redencyStart,redencyEnd,redencyAxes,redencyAddData,redencyRepeat,name,redencyGatherAxes)
#   concatOp = self.concat([*redencyPart,*unRedencyPart],1)
#   return concatOp

@gs.Graph.register()
def addRedencyRewrite(self,input,originalNode,inputConst,
                      unRedencyStart,unRedencyEnd,
                      redencyStart,redencyEnd):
  # 冗余重写部分，现在先假设输入张量为[batch,m]，且对一个维度为自由维度，在第二个维度出现冗余
  # TODO:现在只这样子考虑的原因一个是推荐系统里面性质决定的，其次是目前这种类似[:,0:9,0:32]的截取方法没有办法通过很灵活的拓展到n维度，
  # TODO:想到的一个方法是使用布尔索引矩阵进行截取，这样子可以编写成函数，就不会有这种情况
  # 基于这个假设，unRedencyPartOfAdd函数传递的axes = 1，
  unRedencyAxes = 1
  # redencyPartofAdd 传递的axes = 1，且GatherAxes = 0
  redencyAxes = 1
  redencyGatherAxes = [0]
  # 要指定名字，就拿原来的节点名指定
  name = originalNode.name
  # 这里涉及到子图替换所以要提取一下原本节点的outputs
  outputs = originalNode.outputs
  # for output in outputs:
  #   output.inputs.clear()
  # 需要原来加的常量的数据，传入一个gs的const变量，然后通过.values变成np的张量
  # 现在先假设是非广播加，所以常量和非常量的维度是一样的
  originalInput = inputConst.values
  unRedencyAddData = originalInput[:,unRedencyStart:unRedencyEnd]
  redencyAddData = originalInput[0,redencyStart:redencyEnd]
  redencyRepeat = [1 for _ in input.shape]
  redencyRepeat[0] = input.shape[0]
  unRedencyPart = self.unRedencyPartOfAdd(input,unRedencyStart,unRedencyEnd,unRedencyAxes,unRedencyAddData,name)
  redencyPart = self.redencyPartofAdd(input,redencyStart,redencyEnd,redencyAxes,redencyAddData,redencyRepeat,name,redencyGatherAxes)
  
  concatOp = self.concat([*redencyPart,*unRedencyPart],outputs,1)
  # 替换完了清空一下原本节点的output
  originalNode.outputs = []
  return concatOp

