import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, asdict, field
from .mlp import ConditionedNorm
from ...utils.dataclass import shallow_asdict

############
# Config
############

@dataclass
class AttentionConfig:
    hidden_size:int = 256 # should be multiple of num_heads
    num_heads:int = 8
    num_kv_heads:int = 8
    use_conditional_norm:bool = False
    cond_norm_hidden_size:int = 4

@dataclass 
class FFNConfig:
    hidden_size:int = 256
    use_conditional_norm:bool = False
    cond_norm_hidden_size:int = 4

@dataclass
class TransformerConfig:
    hidden_size:int = 256
    use_attn_norm: bool = True
    use_ffn_norm: bool = True
    norm_eps: float = 1e-6
    num_layers: int = 3
    attn_config: AttentionConfig = field(default_factory=AttentionConfig)
    ffn_config: FFNConfig = field(default_factory=FFNConfig)

"""
Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py
"""

class GroupQueryAttention(nn.Module):
    def __init__(self, 
                 input_size:int,
                 output_size:int, 
                 hidden_size:int = 128, 
                 num_heads:int = 8,
                 num_kv_heads:int = 4,
                 use_conditional_norm:bool = False,
                 cond_norm_hidden_size:int = 4
                 ):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert num_heads % num_kv_heads == 0, f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        self.num_heads = num_heads 
        self.num_kv_heads = num_kv_heads
        self.num_repeat   = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads 

        kv_hidden_size = self.head_dim * num_kv_heads

        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(input_size, hidden_size,    bias=False)
        self.k_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, output_size,   bias=False)

        if use_conditional_norm:
            self.correction = ConditionedNorm(output_size, output_size,cond_norm_hidden_size)
        else:
            self.correction = None
        
        self.attn_dtype = torch.float16
        
    def forward(self, x, condition:Optional[float]=None):
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (..., seq_len, output_size)
        """
        if self.correction is not None:
            x = self.correction(x, condition = condition)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)) # (..., seq_len, num_heads, head_dim)
        k = k.view(k.shape[:-1] + (self.num_kv_heads, -1)) # (..., seq_len, num_kv_heads, head_dim)
        v = v.view(v.shape[:-1] + (self.num_kv_heads, -1)) # (..., seq_len, num_kv_heads, head_dim)

        # NOTE: should use triton or expand to lower mem cost
        k = k.repeat_interleave(self.num_repeat, dim=-2) # (..., seq_len, num_heads, head_dim)
        v = v.repeat_interleave(self.num_repeat, dim=-2) # (..., seq_len, num_heads, head_dim)

        q = q.transpose(-3,-2) # (... , num_heads, seq_len, head_dim)
        k = k.transpose(-3,-2) # (... , num_heads, seq_len, head_dim)
        v = v.transpose(-3,-2) # (... , num_heads, seq_len, head_dim)

        attn = (q @ k.mT) * self.scale # (..., num_heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1) 
        x = attn @ v # （..., num_heads, seq_len, head_dim)
        x = x.transpose(-3,-2) #  (..., seq_len, num_heads, head_dim)
        x = x.contiguous().view(x.shape[:-2] + (-1,)) # (..., seq_len, hidden_size)
        x = self.o_proj(x)

        return x

    @classmethod
    def from_config(cls, input_size:int, output_size:int, config:AttentionConfig):
        return cls(input_size, output_size, **asdict(config))

class GroupQueryFlashAttention(nn.Module):
    def __init__(self, 
                 input_size:int,
                 output_size:int, 
                 hidden_size:int = 128, 
                 num_heads:int = 8,
                 num_kv_heads:int = 4,
                 use_conditional_norm:bool = False,
                 cond_norm_hidden_size:int = 4
                 ):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert num_heads % num_kv_heads == 0, f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        self.num_heads = num_heads 
        self.num_kv_heads = num_kv_heads
        self.num_repeat   = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads 
    
        kv_hidden_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(input_size, hidden_size,    bias=False)
        self.k_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, output_size,   bias=False)

        if use_conditional_norm:
            self.correction = ConditionedNorm(1, output_size,cond_norm_hidden_size)
        else:
            self.correction = None
            
        self.attn_dtype = torch.float16  # or torch.bfloat16

    def forward(self, x, condition:Optional[float]=None):
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (..., seq_len, output_size)
        """
        # 
        #x = x.to(self.attn_dtype)
        if self.correction is not None:
            x = self.correction(c=condition, x=x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, _ = q.size()

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)
            v = v.repeat_interleave(self.num_repeat, dim=1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.o_proj(x)
        
        return x

    @classmethod
    def from_config(cls, input_size:int, output_size:int, config:AttentionConfig):
        return cls(input_size, output_size, **asdict(config))

class FFN(nn.Module):
    def __init__(self,
                input_size:int, 
                output_size:int, 
                hidden_size:int = 256,
                use_conditional_norm:bool = False, 
                cond_norm_hidden_size:int = 4
                ):
        super().__init__()
        self.w1 = nn.Linear(input_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, output_size,bias=False)
        self.w3 = nn.Linear(input_size, hidden_size, bias=False)

        if use_conditional_norm:
            self.correction = ConditionedNorm(1, output_size,cond_norm_hidden_size)
        else:
            self.correction = None

    def forward(self, x, condition:Optional[float]=None):
        x = self.w2(F.silu(self.w1(x))*self.w3(x))

        if self.correction is not None:
            x = self.correction(c = condition, x=x)

        return x

    @classmethod
    def from_config(cls, input_size:int, output_size:int, config:FFNConfig):
        return cls(input_size, output_size, **asdict(config))

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, 
                input_size:int, 
                output_size:int,
                use_attn_norm:bool = True,
                use_ffn_norm:bool = True,
                norm_eps:float = 1e-6,
                attn_config:AttentionConfig = AttentionConfig(),
                ffn_config:FFNConfig = FFNConfig()
                ):
        super().__init__()
                
        self.attn = GroupQueryFlashAttention.from_config(input_size, 
                                    attn_config.hidden_size, 
                                    config = attn_config)
        self.ffn  = FFN.from_config(attn_config.hidden_size, 
                                    output_size, 
                                    config = ffn_config)

        self.attn_norm = RMSNorm(input_size, eps=norm_eps) if use_attn_norm else None 
        self.ffn_norm  = RMSNorm(attn_config.hidden_size, eps=norm_eps) if use_ffn_norm else None 

    def forward(
        self,
        x: torch.Tensor,
        condition:Optional[float]=None
    )->torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor, shape (..., seq_len, input_size)
        condition:Optional[float]

        Returns
        -------
        torch.Tensor, shape (..., seq_len, output_size)
        """
        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, condition = condition)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + self.ffn(h, condition = condition)
        return out

    @classmethod 
    def from_config(cls,input_size:int, 
                        output_size:int, 
                        config:TransformerConfig = TransformerConfig()):
        kwargs = shallow_asdict(config)
        kwargs.pop("num_layers")
        kwargs.pop("hidden_size")
        return cls(input_size, output_size, **kwargs)

class Transformer(nn.Module):
    def __init__(self, 
                input_size:int, 
                output_size:int, 
                config:TransformerConfig = TransformerConfig()
                ):
        super().__init__()
        hidden_size:int = config.hidden_size
        num_layers:int  = config.num_layers
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = nn.Identity()

        if hidden_size != output_size:
            self.output_proj = nn.Linear(hidden_size, output_size)
        else:
            self.output_proj = nn.Identity()

        self.layers = nn.ModuleList([
            TransformerBlock.from_config(
                hidden_size,
                hidden_size,
                config
            ) for _ in range(config.num_layers)
        ])

    def forward(self, x:torch.Tensor, condition:Optional[float]=None)->torch.Tensor:
        """ 
        Parameters
        ----------
        x: torch.Tensor 
            [..., seq_len, input_size]
        
        Returns
        -------
        torch.Tensor
            [..., seq_len, output_size]
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, condition = condition)
        x = self.output_proj(x)
        return x
