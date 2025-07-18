import torch
import torch.nn.functional as F
from torch import nn
from torchfm.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
import onnx
import torch.fx as fx
from ReLayer import ReFeaturesEmbedding,ReFeaturesLinear,ReMultiLayerPerceptron
from einops import rearrange,repeat,reduce

from onnxsim import simplify
import onnxoptimizer
from onnx.shape_inference import infer_shapes

class ReMultiHeadSelfAttentionV2(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None,dropout = 0.1):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self._dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, self._dim * 3, bias=False)
        self.W_0 = nn.Linear( self._dim, dim, bias=True)
        self.scale_factor = self.dim_head ** 0.5

    def forward(self, x):
        assert x[0].dim() == 3
        assert x[1].dim() == 3
        field = x[1].shape[0] + x[0].shape[0]
        batch = x[1].shape[1]
        # Step 1
        qkv = (self.to_qvk(x[0]),self.to_qvk(x[1]))
        
        # Step 2
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        # q, k, v = tuple(rearrange(qkv, 'f b (k h d) -> k f (b h) d ', k=3, h=self.heads ))
        q_s = qkv[1][:,:,:self._dim].reshape(-1,batch * self.heads,self.dim_head).permute(1, 0, 2)
        k_s = qkv[1][:,:,self._dim : self._dim * 2].reshape(-1,batch * self.heads,self.dim_head).permute(1,2,0)
        v_s = qkv[1][:,:,self._dim * 2:].reshape(-1,batch * self.heads,self.dim_head).permute(1, 0, 2)

        q_p = qkv[0][:,:,:self._dim].reshape(-1,1 * self.heads,self.dim_head).permute(1, 0, 2)
        k_p = qkv[0][:,:,self._dim : self._dim * 2].reshape(-1,1 * self.heads,self.dim_head).permute(1,2,0)
        v_p = qkv[0][:,:,self._dim * 2:].reshape(-1,1 * self.heads,self.dim_head).permute(1, 0, 2)
        
        v = torch.concat([v_p.repeat(batch,1,1),v_s], dim = 1)
        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        # scaled_dot_prod = torch.bmm(q.permute(1, 0, 2), k.permute(1,2,0)) / self.scale_factor
        qs_ks = torch.bmm(q_s,k_s) / self.scale_factor
        qp_kp = torch.bmm(q_p,k_p) / self.scale_factor
        qp_ks = torch.bmm(q_p.repeat(batch,1,1),k_s) / self.scale_factor
        qs_kp = torch.bmm(q_s,k_p.repeat(batch,1,1)) / self.scale_factor
        
        qp_kp_qp_ks = torch.concat([qp_kp.repeat([batch,1,1]),qp_ks],dim = 2)
        qs_kp_qs_ks = torch.concat([qs_kp,qs_ks],dim = 2)
        scaled_dot_prod = torch.concat([qp_kp_qp_ks,qs_kp_qs_ks],dim = 1)
        
        # if mask is not None:
        #     assert mask.shape == scaled_dot_prod.shape[2:]
        #     scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 4. Calc result per batch and per head h
        out  = attention @ v
        out = out.permute(1, 0, 2)
        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "f (b h) d -> (f b) (h d)",h=self.heads,d=self.dim_head)

        # Step 6. Apply final linear transformation layer
        out = self.W_0(out)
        out = rearrange(out, "(f b) (d h)-> f b (d h)",h=self.heads,d=self.dim_head,f =field )
        return out





class MultiHeadSelfAttentionV2(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None,dropout = 0.1):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self._dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, self._dim * 3, bias=False)
        self.W_0 = nn.Linear( self._dim, dim, bias=True)
        self.scale_factor = self.dim_head ** 0.5

    def forward(self, x):
        assert x.dim() == 3
        field = x.shape[0]
        batch = x.shape[1]
        # Step 1
        qkv = self.to_qvk(x)  
        
        # Step 2
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        # q, k, v = tuple(rearrange(qkv, 'f b (k h d) -> k f (b h) d ', k=3, h=self.heads ))
        q = qkv[:,:,:self._dim].reshape(-1,batch * self.heads,self.dim_head)
        k = qkv[:,:,self._dim : self._dim * 2].reshape(-1,batch * self.heads,self.dim_head)
        v = qkv[:,:,self._dim * 2:].reshape(-1,batch * self.heads,self.dim_head)

        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        scaled_dot_prod = torch.bmm(q.permute(1, 0, 2), k.permute(1,2,0)) / self.scale_factor

        # if mask is not None:
        #     assert mask.shape == scaled_dot_prod.shape[2:]
        #     scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 4. Calc result per batch and per head h
        out  = attention @ v.permute(1, 0, 2)
        out = out.permute(1, 0, 2)
        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "f (b h) d -> (f b) (h d)",h=self.heads,d=self.dim_head)

        # Step 6. Apply final linear transformation layer
        out = self.W_0(out)
        out = rearrange(out, "(f b) (d h)-> f b (d h)",h=self.heads,d=self.dim_head,f =field )
        return out
    
    
    
class AutomaticFeatureInteractionModel(torch.nn.Module):
    """
    A pytorch implementation of AutoInt.

    Reference:
        W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
    """

    def __init__(self, field_dims, embed_dim, atten_embed_dim, num_heads, num_layers, mlp_dims, dropouts,prefix, has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.prefix = prefix
        self.embedding = ReFeaturesEmbedding(field_dims, embed_dim,prefix=self.prefix)
        self.linear = ReFeaturesLinear(field_dims,prefix=self.prefix)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = ReMultiLayerPerceptron(None,prefix * embed_dim,(self.num_fields - prefix) * embed_dim,mlp_dims,dropouts[1])
        self.self_attns = torch.nn.ModuleList([
            # torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
            MultiHeadSelfAttentionV2(atten_embed_dim, num_heads, dropout=dropouts[0]) if i > 0 else ReMultiHeadSelfAttentionV2(atten_embed_dim, num_heads, dropout=dropouts[0])   for i in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        batch = x[1].size(0)
        embed_x = self.embedding(x)
        atten_x = (self.atten_embedding(embed_x[0]),self.atten_embedding(embed_x[1]))
        cross_term = (atten_x[0].transpose(0, 1),atten_x[1].transpose(0, 1))
        for self_attn in self.self_attns:
            # cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            cross_term= self_attn(cross_term)
        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = (self.V_res_embedding(embed_x[0]),self.V_res_embedding(embed_x[1]))
            V_res = torch.concat([V_res[0].repeat(batch,1,1),V_res[1]],dim = 1)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(embed_x)
        return torch.sigmoid(x.squeeze(1))


def gen_avazu(batch,col,dim):
    l =[    241,       8,       8,    3697,    4614,      25,    5481,
           329,      31,  381763, 1611748,    6793,       6,       5,
          2509,       9,      10,     432,       5,      68,     169,
            61]
    prefix = 12
    print(f"gen onnx model,batch:{batch},col:{col},dim:{dim}\n")
    model = AutomaticFeatureInteractionModel(l,dim,dim,8,3,[400,400,400],[0.1,0.1],12)
    sample = [torch.zeros(prefix,dtype=torch.int64),torch.zeros((batch,col - prefix),dtype=torch.int64)]
    torch.onnx.export(model,sample,f'/home/yssun/pytorch-fm/models/afi/reafi_{batch}_{col}_{dim}_custom.onnx')
    onnx_model = onnx.load(f'/home/yssun/pytorch-fm/models/afi/reafi_{batch}_{col}_{dim}_custom.onnx')  # load onnx model
    onnx_model = onnxoptimizer.optimize(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = infer_shapes(onnx_model)
    onnx.save(onnx_model, f'/home/yssun/pytorch-fm/models/afi/reafi_{batch}_{col}_{dim}_custom_s.onnx')
def gen_avazuori(batch,col,dim,head=8):
    l =[    241,       8,       8,    3697,    4614,      25,    5481,
           329,      31,  381763, 1611748,    6793,       6,       5,
          2509,       9,      10,     432,       5,      68,     169,
            61]
    print(f"gen onnx model,batch:{batch},col:{col},dim:{dim}\n")
    model = AutomaticFeatureInteractionModel(l,dim,dim*3,head,3,[400,400,400],[0.1,0.1])
    sample = torch.zeros((batch,col),dtype=torch.int64)
    torch.onnx.export(model,sample,f'/home/yssun/pytorch-fm/models/afi/rafi_{batch}_{col}_{dim}_{head}.onnx')
    onnx_model = onnx.load(f'/home/yssun/pytorch-fm/models/afi/rafi_{batch}_{col}_{dim}_{head}.onnx')  # load onnx model
    onnx_model = onnxoptimizer.optimize(onnx_model)
    traced_model = torch.jit.trace(model, sample)
    # print(traced_model)
    # tracer = fx.Tracer()
    # graph = tracer.trace(model, sample)
    # print(graph)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = infer_shapes(onnx_model)
    onnx.save(onnx_model, f'/home/yssun/pytorch-fm/models/afi/rafi_{batch}_{col}_{dim}_{head}_s.onnx')


if __name__ == "__main__":

    for b in [1024]:
        for dim in [32]:
            # gen_avazuori(b,22,dim,16)
            gen_avazu(b,22,dim)