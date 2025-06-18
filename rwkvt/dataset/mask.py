import torch
import numpy as np

# def create_mask(seq, token1, token2, min_len):
#     # 找到所有特殊标记的索引
#     indices1 = []
#     for i in range(min_len - len(token1) + 1):
#         if np.array_equal(seq[i:i + len(token1)], token1):
#             indices1.append(i)
#     indices2 = []

#     for i in range(min_len - len(token2) + 1):
#         if np.array_equal(seq[i:i + len(token2)], token2):
#             indices2.append(i)
#     mask = torch.zeros(seq.shape)
#     # assert len(indices2)!=0 and len(indices1)!=0
#     select = 0
#     for i in range(min_len):
#         if i in indices1:
#             select = 0
#         elif i in indices2:
#             select = 1
#         mask[i] = select
#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]

# def create_mask(seq, start, end, min_len):
#     # 找到所有特殊标记的索引
#     for token in seq:
        
#     indices1 = []
#     for i in range(min_len - len(token1) + 1):
#         if np.array_equal(seq[i:i + len(token1)], token1):
#             indices1.append(i)
#     indices2 = []

#     for i in range(min_len - len(token2) + 1):
#         if np.array_equal(seq[i:i + len(token2)], token2):
#             indices2.append(i)
#     mask = torch.zeros(seq.shape)
#     # assert len(indices2)!=0 and len(indices1)!=0
#     select = 0
#     for i in range(min_len):
#         if i in indices1:
#             select = 0
#         elif i in indices2:
#             select = 1
#         mask[i] = select
#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]

# def generate_mask(seq, token1, token2, min_len):
#     mask = torch.zeros(seq.shape)  # 初始化mask列表，默认全为0
#     current_mask_value = 0  # 初始状态下，所有位置的mask值为0

#     i = 0
#     while i < min_len:
#         if seq[i:i + len(token1)] == token1:
#             current_mask_value = 0
#             for j in range(len(token1)):
#                 mask[i + j] = current_mask_value
#             i += len(token1)
#         elif seq[i:i + len(token2)] == token2:
#             current_mask_value = 1
#             for j in range(len(token2)):
#                 mask[i + j] = current_mask_value
#             i += len(token2)
#         else:
#             mask[i] = current_mask_value
#             i += 1

#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]
def create_mask():
    return

def generate_mask():
    return
mask_fn_dict = {
    "qa": create_mask,
    "se": generate_mask
}