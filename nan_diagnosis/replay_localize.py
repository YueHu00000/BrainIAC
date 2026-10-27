"""Localize saved BrainIAC forward faults; never update parameters or source files."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
from types import MethodType

import torch
from torch.nn import functional as F

import diagnose_nan as diag


def summary_v2(value):
    value = value.detach().as_subclass(torch.Tensor)
    bad = ~torch.isfinite(value)
    result = {"shape": list(value.shape), "dtype": str(value.dtype),
              "finite": not bool(bad.any()), "nonfinite": int(bad.sum())}
    if result["finite"] and value.numel():
        result["max_abs"] = float(value.abs().max())
    elif not result["finite"]:
        result.update(nan_count=int(torch.isnan(value).sum()),
                      inf_count=int(torch.isinf(value).sum()))
    return result


class ForwardTraceV2:
    """Observe original coarse forward, or instrument one selected MONAI 1.3.2 scope."""

    def __init__(self, unique_ids, detailed_target=None):
        self.unique_ids = list(unique_ids)
        self.detailed_target = detailed_target
        self.stages = []
        self.first_nonfinite = None
        self.view = -1
        self.handles = []
        self.replaced = []

    def observe(self, stage, value, reference=None):
        value = value.detach().as_subclass(torch.Tensor)
        entry = {"stage": f"view{self.view}.{stage}", **summary_v2(value)}
        if not entry["finite"]:
            # Only batch-first forward tensors reach this observer. Q/K/V are
            # observed separately, never the rearranged [3,B,H,N,D] container.
            if value.ndim and value.shape[0] == len(self.unique_ids):
                counts = (~torch.isfinite(value)).reshape(value.shape[0], -1).sum(1)
                indices = (counts > 0).nonzero().flatten().cpu().tolist()
                entry.update(sample_indices=indices,
                             unique_ids=[self.unique_ids[i] for i in indices],
                             nonfinite_per_sample=[int(counts[i]) for i in indices])
                if reference is not None:
                    # One offending sample is enough for the local operator
                    # comparison, without retaining another full B160 attention.
                    index = indices[0]
                    with torch.no_grad(), torch.autocast(value.device.type, enabled=False):
                        ref = reference(index)
                    entry["float32_reference"] = {
                        "scope": "same_operand_values_cast_to_float32; saved TF32 policy",
                        "sample_indices": [index], "unique_ids": [self.unique_ids[index]],
                        "omitted_bad_sample_count": len(indices) - 1,
                        "stats": summary_v2(ref),
                        "cast_back_stats": summary_v2(ref.to(value.dtype))}
            self.first_nonfinite = entry["stage"]
        self.stages.append(entry)
        if self.first_nonfinite is not None:
            raise diag.NonFiniteError(self.first_nonfinite)

    def hook_module(self, name, module):
        def before(_module, inputs):
            self.observe(name + ".input", inputs[0])

        def after(_module, inputs, output):
            reference = None
            if isinstance(module, torch.nn.Linear):
                def reference(index):
                    return F.linear(inputs[0][index:index + 1].detach().float(),
                                    module.weight.detach().float(),
                                    module.bias.detach().float() if module.bias is not None else None)
            elif isinstance(module, torch.nn.LayerNorm):
                def reference(index):
                    return F.layer_norm(inputs[0][index:index + 1].detach().float(),
                                        module.normalized_shape,
                                        module.weight.detach().float() if module.weight is not None else None,
                                        module.bias.detach().float() if module.bias is not None else None,
                                        module.eps)
            if name == "backbone":
                output = output[0]
            self.observe(name + ".output", output, reference)

        self.handles.extend([module.register_forward_pre_hook(before),
                             module.register_forward_hook(after)])

    def instrument_attention(self, name, module):
        import monai
        from monai.networks.blocks.selfattention import SABlock
        if monai.__version__ != "1.3.2" or type(module) is not SABlock:
            raise ValueError("Detailed attention requires the original MONAI 1.3.2 SABlock")
        had_forward = "forward" in module.__dict__
        old_forward = module.__dict__.get("forward")
        self.replaced.append((module, had_forward, old_forward))
        trace = self

        # Literal MONAI 1.3.2 operation order; observations only detach tensors.
        def forward_v2(attn, x):
            output = attn.input_rearrange(attn.qkv(x))
            q, k, v = output[0], output[1], output[2]
            for label, value in (("q", q), ("k", k), ("v", v)):
                trace.observe(name + "." + label, value)
            raw = torch.einsum("blxd,blyd->blxy", q, k)
            trace.observe(name + ".qk_raw", raw,
                          lambda i: torch.einsum("blxd,blyd->blxy",
                                                q[i:i + 1].detach().float(),
                                                k[i:i + 1].detach().float()))
            scores = raw * attn.scale
            trace.observe(name + ".scaled_scores", scores,
                          lambda i: raw[i:i + 1].detach().float() * attn.scale)
            att_mat = scores.softmax(dim=-1)
            trace.observe(name + ".softmax", att_mat,
                          lambda i: scores[i:i + 1].detach().float().softmax(dim=-1))
            if attn.save_attn:
                attn.att_mat = att_mat.detach()
            att_mat = attn.drop_weights(att_mat)
            x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
            trace.observe(name + ".weighted_values", x,
                          lambda i: torch.einsum("bhxy,bhyd->bhxd",
                                                att_mat[i:i + 1].detach().float(),
                                                v[i:i + 1].detach().float()))
            x = attn.out_rearrange(x)
            x = attn.out_proj(x)
            x = attn.drop_output(x)
            return x

        module.forward = MethodType(forward_v2, module)

    def install(self, model):
        def start_view(_module, _inputs):
            self.view += 1

        self.handles.append(model.backbone.register_forward_pre_hook(start_view))
        coarse = {"backbone", "backbone.patch_embedding", "backbone.norm", "projection_head"}
        coarse.update(f"backbone.blocks.{i}" for i in range(len(model.backbone.blocks)))
        for name, module in model.named_modules():
            detailed = self.detailed_target and name.startswith(self.detailed_target + ".")
            if name in coarse or (detailed and not list(module.children())
                                  and "rearrange" not in name):
                self.hook_module(name, module)
        if self.detailed_target and self.detailed_target.startswith("backbone.blocks."):
            block = model.get_submodule(self.detailed_target)
            self.instrument_attention(self.detailed_target + ".attn", block.attn)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, had_forward, old_forward in self.replaced:
            if had_forward:
                module.forward = old_forward
            else:
                del module.forward
        self.replaced.clear()


def batch_ids_v2(batch):
    first = list(batch[0]["unique_id"])
    for view in batch:
        if len(first) != view["image"].shape[0] or list(view["unique_id"]) != first:
            raise ValueError("Saved views must have matching batch-first images and unique_id order")
    return first


def run_trial_v2(state, batch, precision, device, forward_only=False,
                 detailed_target=None, model_factory=diag.make_model):
    trace = ForwardTraceV2(batch_ids_v2(batch), detailed_target)

    def factory(hparams):
        model = model_factory(hparams)
        trace.install(model)
        return model

    try:
        result = diag.replay_trial(state, batch, precision, device,
                                   model_factory=factory, forward_only=forward_only)
    finally:
        trace.close()
    if trace.first_nonfinite is not None:
        result.update(finite=False, first_nonfinite=trace.first_nonfinite)
    result.update(trace_stages=trace.stages,
                  backward_requested=not forward_only,
                  backward_ran=result.get("backward_ran", False),
                  backward_completed="gradients" in result,
                  optimizer_updated=False)
    return result


def locate_trial_v2(state, batch, precision, device, forward_only=False,
                    model_factory=diag.make_model):
    coarse = run_trial_v2(state, batch, precision, device, forward_only,
                          model_factory=model_factory)
    first = coarse.get("first_nonfinite", "")
    match = re.match(r"view\d+\.(backbone\.blocks\.\d+|backbone\.patch_embedding|projection_head)\.", first)
    target = match.group(1) if match else None
    detail = (run_trial_v2(state, batch, precision, device, forward_only,
                           detailed_target=target, model_factory=model_factory) if target else None)
    return {"precision": precision, "coarse": coarse, "detail": detail,
            "detailed_target": target,
            "detail_reproduced_nonfinite": not detail["finite"] if detail is not None else None}


def parse_args_v2(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precisions", nargs="+", default=["16-mixed", "32-true"],
                        choices=("16-mixed", "32-true", "bf16-mixed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forward-only", action="store_true")
    return parser.parse_args(argv)


def main_v2(argv=None):
    args = parse_args_v2(argv)
    output = diag.fresh_output(args.output_dir)
    summary = json.loads((args.scene / "summary.json").read_text(encoding="utf-8"))
    if summary["kind"] == "data_nonfinite":
        raise ValueError("Localization needs a full saved model scene, not a worker data fault")
    state = torch.load(args.scene / "state.pt", map_location="cpu", weights_only=False)
    batch = torch.load(args.scene / "batch.pt", map_location="cpu", weights_only=False)
    batch_ids_v2(batch)
    report = {"scene": str(args.scene), "source_summary": summary,
              "source_environment": state["environment"], "trials": [],
              "forward_only": args.forward_only, "optimizer_update_requested": False,
              "note": "Each pass reloads state/buffers/RNG; FP32 keeps saved TF32 policy. "
                      "First observed bad tensor is not automatically the root cause."}
    for precision in args.precisions:
        if precision != "32-true" and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
            trial = {"precision": precision, "status": "not_run_requires_cuda"}
        elif precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            trial = {"precision": precision, "status": "not_run_bf16_unsupported"}
        else:
            try:
                trial = locate_trial_v2(state, batch, precision, args.device, args.forward_only)
                trial["status"] = "completed"
            except Exception as error:
                report["trials"].append({"precision": precision, "status": "error",
                                         "error_type": type(error).__name__, "error": str(error)})
                report["environment"] = diag.environment()
                diag.write_json(output / "localization_report.json", report)
                raise
        report["trials"].append(trial)
        report["environment"] = diag.environment()
        diag.write_json(output / "localization_report.json", report)
        print(json.dumps({"precision": precision, "status": trial["status"],
                          "first_nonfinite": trial.get("coarse", {}).get("first_nonfinite"),
                          "detailed_first_nonfinite": (trial.get("detail") or {}).get("first_nonfinite")}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_v2())
