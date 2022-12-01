import os
import unittest
import random
import tempfile
import torch
from types import SimpleNamespace
from megatron.mpu import initialize_model_parallel
from metaseq.model_parallel.modules import ModelParallelTransformerDecoderLayer


def reset_seeds():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


def _distributed_init():
    backend = "nccl"
    local_rank = None
    rank = 0
    world_size = 1
    device = 0
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, init_method=init_method
    )


# TODO: add dtype
class TestParity(unittest.TestCase):
    def test_xformers_parity(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")

        _distributed_init()
        tensor_model_parallel_size_ = 1
        initialize_model_parallel(tensor_model_parallel_size_)

        args = SimpleNamespace(
            sequence_parallel=True,
            decoder_embed_dim=64,
            dropout=0.0,
            decoder_attention_heads=1,
            decoder_ffn_embed_dim=64,
            decoder_layers=1,
            attention_dropout=0.0,
        )
        S, B, E = 128, 2, 64
        x = torch.rand((S, B, E), device="cuda", requires_grad=False)
        x_ = x.clone()
        x.requires_grad = True
        x_.requires_grad = True

        xf_attn_variant = "xformers_default"
        std_attn_variant = "default"

        # xformers
        args.attn_variant = xf_attn_variant
        reset_seeds()
        xf_decoder = ModelParallelTransformerDecoderLayer(args).cuda()
        xf_result = xf_decoder(x)

        # std attn
        args.attn_variant = std_attn_variant
        reset_seeds()
        decoder = TransformerDecoderLayer(args).cuda()
        result = decoder(x_)

        assert torch.allclose(xf_result, result)

        loss_xf = torch.norm(xf_result)
        loss_xf.backward()

        loss = torch.norm(result)
        loss.backward()

        assert torch.allclose(x.grad, x_.grad)


if __name__ == "__main__":
    unittest.main()
