import torch

from config import PARAM_RANGES, NUM_PARAMS


def random_frame_batch(batch_size=1, num_frames=11):
    B, T = batch_size, num_frames
    params = torch.zeros(B, T, NUM_PARAMS, dtype=torch.int64)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        params[:, :, i] = torch.randint(lo, hi + 1, (B, T))
    return params
