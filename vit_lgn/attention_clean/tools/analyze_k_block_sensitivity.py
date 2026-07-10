import sys
import torch

sys.path.insert(0, "/home/pan/ViT_LGN/attention-clean")

from vit_tiny_attention_logic import LogicTransformerBlock


def build_block(k: int, attention_only: bool) -> LogicTransformerBlock:
    return LogicTransformerBlock(
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path=0.0,
        attention_only=attention_only,
        attention_k=k,
        validate_input=False,
        use_thermometer_encoding=True,
        n_thresholds=3,
        encoding_scale=10.0,
        apply_sigmoid_before_encoding=True,
        decode_output=True,
        boundary_surrogate_temperature=1.0,
        topk_impl="torch-topk",
        topk_surrogate_mode="kth",
        topk_surrogate_proxy_source="soft-thermometer",
        topk_kth_detach_value=False,
        majority_train_temperature=0.5,
        fast_vote=True,
        majority_surrogate_mode="fraction",
        thermometer_decode_use_weights=False,
    )


def run_block(block: LogicTransformerBlock, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = block.norm1(x)
    attn = block.attn(x1)
    after_attn = x + attn
    x2 = block.norm2(after_attn)
    mlp = block.mlp(x2)
    out = after_attn + mlp
    return attn, out, mlp


def compare(ks: list[int], attention_only: bool) -> None:
    base = build_block(25, attention_only=attention_only)
    state = base.state_dict()
    models = {}
    for k in ks:
        model = build_block(k, attention_only=attention_only)
        model.load_state_dict(state, strict=False)
        model.eval()
        models[k] = model

    x = torch.randn(8, 65, 192)
    ref_attn, ref_out, ref_mlp = run_block(models[25], x)
    print("attention_only" if attention_only else "with_ffn")
    for k in ks:
        attn, out, mlp = run_block(models[k], x)
        attn_diff = (attn - ref_attn).norm().item() / (ref_attn.norm().item() + 1e-12)
        out_diff = (out - ref_out).norm().item() / (ref_out.norm().item() + 1e-12)
        if attention_only:
            print(f"k={k:2d} attn_diff={attn_diff:.4f} out_diff={out_diff:.4f}")
        else:
            attn_to_mlp = attn.norm().item() / (mlp.norm().item() + 1e-12)
            print(f"k={k:2d} attn_diff={attn_diff:.4f} out_diff={out_diff:.4f} attn_to_mlp={attn_to_mlp:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    ks = [1, 9, 25, 63]
    compare(ks, attention_only=False)
    compare(ks, attention_only=True)
