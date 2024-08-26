import torch
import os
import unittest
from unittest.mock import patch

# 原始版本的 L2Wrap
class L2WrapOriginal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        if ctx.token_amount == 0:
            return (grad_output, None, None)
        factor = 1e-4 / ctx.token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        if os.environ.get("WN_FIX_L2WRAP"):
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)

# 新版本的 L2Wrap
class L2WrapNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
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
        return grad_output, grad_y.to_dense(), None

class TestL2Wrap(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 5
        self.token_amount = self.batch_size * self.seq_len
        self.y = torch.randn(self.batch_size, self.seq_len, self.vocab_size, requires_grad=True)
        self.loss = torch.tensor(1.0, requires_grad=True)
        self.grad_output = torch.tensor(1.0)

    def test_l2wrap_original(self):
        with patch.dict(os.environ, {"WN_FIX_L2WRAP": "1"}):
            loss_original = L2WrapOriginal.apply(self.loss, self.y, self.token_amount)
            loss_original.backward(self.grad_output)
            grad_original = self.y.grad.clone()
            self.y.grad.zero_()

        return grad_original

    def test_l2wrap_new(self):
        with patch.dict(os.environ, {"WN_FIX_L2WRAP": "1"}):
            loss_new = L2WrapNew.apply(self.loss, self.y, self.token_amount)
            loss_new.backward(self.grad_output)
            grad_new = self.y.grad.clone()
            self.y.grad.zero_()

        return grad_new

    def test_consistency(self):
        grad_original = self.test_l2wrap_original()
        grad_new = self.test_l2wrap_new()

        self.assertTrue(torch.allclose(grad_original, grad_new, atol=1e-6),
                        "Gradients from original and new implementations are not close enough")

if __name__ == '__main__':
    unittest.main()