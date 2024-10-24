import math
import torch

def gen_sineembed_for_position(pos_tensor, num_pos_feats=128, scale=2 * math.pi):
    """
    Generate sinusoidal embeddings for positional information.
    
    Args:
        pos_tensor (torch.Tensor): Tensor containing the positional information. 
                                   Expected shape is (n_query, batch_size, num_dims).
                                   `num_dims` can be 2 (x, y) or 4 (x, y, w, h).
        num_pos_feats (int): Number of positional features (default: 128).
        scale (float): Scaling factor for positional encoding (default: 2 * pi).
        
    Returns:
        torch.Tensor: Sinusoidal embeddings for the given positions.
    """
    # Calculate the dim_t scaling factor
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)

    # Scale x and y embeddings
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t

    # Compute sine and cosine embeddings for x and y
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    # Handle the cases for positional tensors with width and height dimensions
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        # Scale width and height embeddings
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unsupported number of dimensions: {pos_tensor.size(-1)}. Expected 2 or 4.")
    
    return pos
