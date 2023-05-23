# Libraries
import torch
from torch import nn
import einops
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
n_channels = 3
patch_size = 16
latent_size = 768
batch_size = 35
num_heads = 12
dropout = 0.1
num_encoders = 12
num_classes = 4


class InputEmbedding(nn.Module):
    def __init__(self,n_channels = n_channels,patch_size = patch_size,batch_size = batch_size,latent_size = latent_size,device = device):
        super(InputEmbedding,self).__init__()
        
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.device = device
        
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        self.linear_proj = nn.Linear(self.input_size,self.latent_size).to(self.device)
        self.cls_token = self._block(self.batch_size,self.latent_size).to(self.device)
        self.pos_embed = self._block(self.batch_size,self.latent_size).to(self.device)

    def _block(self,batch_size,latent_size):
        return nn.Parameter(torch.randn(batch_size,1,latent_size))
    
    def forward(self,data):
        data = data.to(self.device)
        patches = einops.rearrange( data,
                                  
                                     pattern = 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)',h1 = self.patch_size,w1 = self.patch_size # b*n*(h1*w1*c)
                                 )
        
        linear_projection = self.linear_proj(patches).to(self.device)
        b,n,_ = linear_projection.shape # b*n*(h1*w1*c)
        linear_projection = torch.cat((self.cls_token,linear_projection),dim = 1) # b*n+1*(h1*w1*c)
        position_embedding = einops.repeat(self.pos_embed,pattern = 'b 1 d -> b m d',m = n + 1)
        linear_projection += position_embedding
        
        return linear_projection
    

class TransformerEncoder(nn.Module):
    def __init__(self,latent_size = latent_size,num_heads = num_heads,dropout = dropout):
        super(TransformerEncoder,self).__init__()
        
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.norm = nn.LayerNorm(self.latent_size)
        self.mult_head_att = nn.MultiheadAttention(self.latent_size,self.num_heads,self.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size,self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4,self.latent_size),
            nn.Dropout(self.dropout)
        )
    
    def forward(self,embed):
        first_norm = self.norm(embed)
        multi_head_attention = self.mult_head_att(first_norm,first_norm,first_norm)[0]
        first_residual_conn = embed + multi_head_attention  
        
        second_norm = self.norm(first_residual_conn)
        multi_layer_perc = self.mlp(second_norm)
        second_residual_conn = first_residual_conn + multi_layer_perc 
        
        return second_residual_conn
    

class VitTransformer(nn.Module):
    def __init__(self,latent_size = latent_size,num_encoders = num_encoders,num_classes = num_classes,dropout = dropout,device = device):
        super(VitTransformer,self).__init__()
        
        self.latent_size = latent_size
        self.num_encoders = num_encoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        
        self.input_embed = InputEmbedding().to(self.device)
        self.encoders = nn.ModuleList([TransformerEncoder() for _ in range(self.num_encoders)]).to(self.device)
        self.mlp_head = nn.Sequential(
                                        nn.LayerNorm(self.latent_size),
                                        nn.Linear(self.latent_size,self.latent_size),
                                        nn.Linear(self.latent_size,self.num_classes).to(self.device)
                                     )
    
    def forward(self,input_data):
        input_embed_out = self.input_embed(input_data)
        
        encoder_out = input_embed_out
        for enc in self.encoders:
            encoder_out = enc.forward(encoder_out)
            
        cls_token_embed_out = encoder_out[:,0] 
        mlp_head = self.mlp_head(cls_token_embed_out)
        
        return mlp_head