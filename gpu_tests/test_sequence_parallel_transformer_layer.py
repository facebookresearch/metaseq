import unittest
import random
import sys
import torch
from types import SimpleNamespace
from megatron.mpu import destroy_model_parallel, initialize_model_parallel
from metaseq.model_parallel.modules import ModelParallelTransformerDecoderLayer


def reset_seeds():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


def _distributed_init():
    backend = "nccl"
    rank = 0
    world_size = 1
    device = 0
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = "localhost"
    master_port = "6000"
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, init_method=init_method
    )


def _assert_allclose(out, ref, atol, rtol, msg="failed"):
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    max_diff = flatten_diff[max_pos]
    num_different = torch.count_nonzero(flatten_diff > 0)
    percentage = num_different / flatten_diff.numel()
    del flatten_diff
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={max_diff} > 0)"
        f"/ atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different}, percentage={percentage}"
    )


class TestParity(unittest.TestCase):
    def test_xformers_parity(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if "xformers" not in sys.modules:
            raise unittest.SkipTest("xformers not available, skipping test")

        fw_atol = 4e-3
        fw_rtol = 4e-4

        bw_atol = 9e-2
        bw_rtol = 2e-2

        _distributed_init()
        tensor_model_parallel_size_ = 1
        initialize_model_parallel(tensor_model_parallel_size_)

        S, B, E = 8, 16, 64
        H = 2
        args = SimpleNamespace(
            sequence_parallel=True,
            decoder_embed_dim=E,
            dropout=0.0,
            decoder_attention_heads=H,
            decoder_ffn_embed_dim=64,
            decoder_layers=1,
            attention_dropout=0.0,
            memory_efficient_fp16=True,
            bf16=False,
        )
        S, B, E = 64, 128, 64
        x = torch.rand(
            (S, B, E), device="cuda", dtype=torch.float16, requires_grad=False
        )
        x_ = x.clone()
        x.requires_grad = True
        x_.requires_grad = True

        xf_attn_variant = "xformers_default"
        std_attn_variant = "default"

        # xformers
        args.attn_variant = xf_attn_variant
        reset_seeds()
        xf_decoder = ModelParallelTransformerDecoderLayer(args).cuda()
        reset_seeds()
        xf_result = xf_decoder(x)

        # std attn
        args.attn_variant = std_attn_variant
        reset_seeds()
        decoder = ModelParallelTransformerDecoderLayer(args).cuda()
        reset_seeds()
        result = decoder(x_)

        torch.distributed.barrier()
        _assert_allclose(xf_result, result, atol=fw_atol, rtol=fw_rtol)

        # Test Backwards
        reset_seeds()
        xf_result.backward(torch.ones_like(x))
        reset_seeds()
        result.backward(torch.ones_like(x_))

        torch.distributed.barrier()
        _assert_allclose(x.grad, x_.grad, atol=bw_atol, rtol=bw_rtol)

        # Reset groups
        destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(">> passed the test")


if __name__ == "__main__":
    unittest.main()
