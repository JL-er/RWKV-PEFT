import os
import torch

if os.environ["RWKV_TRAIN_TYPE"] == 'infctx' and os.environ.get("L2WRAP_SPARSE", "0") == "1":
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, token_amount):
            ctx.token_amount = token_amount
            max_vals, max_ids = torch.max(y, -1)
            ctx.save_for_backward(max_vals, max_ids)
            ctx.y_shape = y.shape
            ctx.y_dtype = y.dtype
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            max_vals, max_ids = ctx.saved_tensors
            y_shape = ctx.y_shape
            factor = 1e-4 / ctx.token_amount

            batch_size, seq_len, vocab_size = y_shape
            batch_indices = torch.arange(batch_size, device=max_ids.device).repeat_interleave(seq_len)
            seq_indices = torch.arange(seq_len, device=max_ids.device).repeat(batch_size)

            if os.environ.get("WN_FIX_L2WRAP"):
                # mask = max_vals >= 3.
                # batch_indices = batch_indices[mask.flatten()]
                # seq_indices = seq_indices[mask.flatten()]
                # max_ids = max_ids[mask]
                # max_vals = max_vals[mask]
                values = max_vals.flatten() * factor * grad_output.flatten()
            else:
                values = max_vals.flatten() * factor
            values = values.to(ctx.y_dtype)

            indices = torch.stack([batch_indices, seq_indices, max_ids.flatten()])
            grad_y = torch.sparse_coo_tensor(indices, values, y_shape, dtype=ctx.y_dtype, device=grad_output.device)
            return grad_output, grad_y, None
elif os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, token_amount):
            ctx.save_for_backward(y)
            ctx.token_amount = token_amount
            return loss

        @staticmethod
        def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            if ctx.token_amount == 0:
                return (grad_output, None, None)
            factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
                # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
                gy.scatter_(-1, ids, maxx * factor * grad_output)
            else:
                gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy, None)
elif os.environ.get("L2WRAP_SPARSE", "0") == "1":
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            # 只保存必要的信息
            max_vals, max_ids = torch.max(y, -1)
            ctx.save_for_backward(max_vals, max_ids)
            ctx.y_shape = y.shape
            ctx.y_dtype = y.dtype
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            max_vals, max_ids = ctx.saved_tensors
            y_shape = ctx.y_shape
            factor = 1e-4 / (y_shape[0] * y_shape[1]) # may have problem when padding
            batch_size = y_shape[0]
            seq_len = y_shape[1]
            batch_indices = torch.arange(batch_size, device=max_ids.device).repeat_interleave(seq_len)
            seq_indices = torch.arange(seq_len, device=max_ids.device).repeat(batch_size)

            indices = torch.stack([
                batch_indices,
                seq_indices,
                max_ids.flatten()
            ])
            values = (max_vals.flatten() * factor).to(ctx.y_dtype)
            grad_y = torch.sparse_coo_tensor(indices, values, y_shape, dtype=ctx.y_dtype, device=grad_output.device)
            return grad_output, grad_y
else:
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            ctx.save_for_backward(y)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy)

