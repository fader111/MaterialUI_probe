import torch
import torch.nn as nn
from torch import Tensor

class InitAutoencoder_4layers(nn.Module):
    def __init__(self, num_teeth: int = 28, num_points: int = 5, coord_dim: int = 3):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.dense_dim = num_teeth * num_points * coord_dim
        self.dim_code = self.dense_dim // 6
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim),
            nn.ReLU(),
            nn.Linear(self.dense_dim, self.dense_dim),
            nn.ReLU(),
            nn.Linear(self.dense_dim, self.dense_dim//2),
            nn.ReLU(),
            nn.Linear(self.dense_dim//2, self.dim_code)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_code, self.dense_dim//2),
            nn.ReLU(),
            nn.Linear(self.dense_dim//2, self.dense_dim),
            nn.ReLU(),
            nn.Linear(self.dense_dim, self.dense_dim),
            nn.ReLU(),
            nn.Linear(self.dense_dim, self.dense_dim)
        )

    def forward(self, x):
        # Pass input through encoder
        x = x.view(x.size(0), -1) # Flatten the input
        encoded = self.encoder(x)
        # Pass encoded representation through decoder
        decoded = self.decoder(encoded)
        # Reshape back to original dimensions
        decoded = decoded.view(-1, self.num_teeth, self.num_points, self.coord_dim)
        return decoded
    
class InitAutoencoder(nn.Module):
    def __init__(self, num_teeth: int = 28, num_points: int = 5, coord_dim: int = 3):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.dense_dim = num_teeth * num_points * coord_dim
        self.dense_dim2 = round(self.dense_dim//1.5) #1.5
        self.dense_dim3 = round(self.dense_dim2//2)
        self.dim_code = round(self.dense_dim3//2)
        # self.dim_code = self.dense_dim // 6
        
        # Encoder
        self.encoder = nn.Sequential(
            # nn.Linear(self.dense_dim, self.dense_dim),
            # nn.ELU(),
            nn.Linear(self.dense_dim, self.dense_dim2),
            nn.ELU(),
            nn.Linear(self.dense_dim2, self.dense_dim3),
            nn.ELU(),
            nn.Linear(self.dense_dim3, self.dim_code)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_code, self.dense_dim3),
            nn.ELU(),
            nn.Linear(self.dense_dim3, self.dense_dim2),
            nn.ELU(),
            nn.Linear(self.dense_dim2, self.dense_dim)
            # nn.ELU(),
            # nn.Linear(self.dense_dim, self.dense_dim)
        )

    def forward(self, x):
        # Pass input through encoder
        x = x.view(x.size(0), -1) # Flatten the input
        encoded = self.encoder(x)
        # Pass encoded representation through decoder
        decoded = self.decoder(encoded)
        # Reshape back to original dimensions
        decoded = decoded.view(-1, self.num_teeth, self.num_points, self.coord_dim)
        return decoded
 
class ArchFormRegressor(nn.Module):
    def __init__(self, num_teeth: int = 28, 
                 num_points: int = 5, 
                 coord_dim: int = 3, 
                 hidden_dim: int = 512,
                 num_layers: int = 3):
        super().__init__()
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.input_dim = num_teeth * num_points * coord_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.input_dim
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_dim * 2, hidden_dim))
        layers.append(nn.ELU())
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ae_pred, template):
        # ae_pred, template: [batch, num_teeth, num_points, coord_dim]
        ae_pred = ae_pred.view(ae_pred.size(0), -1)
        template = template.view(template.size(0), -1)
        x = torch.cat([ae_pred, template], dim=1)
        out = self.net(x)
        return out.view(-1, self.num_teeth, self.num_points, self.coord_dim)