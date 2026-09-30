"""Extract one corrected study with the existing BrainIAC embedding format."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainiac_embedding import extract_embeddings as extraction
from brainiac_embedding._resume import check_separate_roots, validate_study_id


def embed_study(study_id, processed_root, output_root, *, model_kind, checkpoint,
                brainiac_checkpoint=None, device="cpu", brainiac_root=None):
    """Reuse the loader, transform, pooling and writer without a cohort scan."""
    validate_study_id(study_id)
    processed_root, output_root = Path(processed_root).resolve(), Path(output_root).resolve()
    check_separate_roots(processed_root, output_root)
    output = output_root / f"{study_id}.npz"
    for path in (output, output.with_name(output.name + ".tmp")):
        if path.exists():
            raise FileExistsError(f"Use a new output root; already exists: {path}")
    images = extraction.load_study_images(study_id, processed_root, brainiac_root=brainiac_root)
    encoder = extraction.load_encoder(
        model_kind, checkpoint, brainiac_checkpoint=brainiac_checkpoint,
        device=device, brainiac_root=brainiac_root,
    )
    block, final = extraction.extract_selected_embeddings(encoder, images, device=device)
    return extraction.save_embedding_npz(output, block, final)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="R01_Study_002904")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-kind", choices=extraction.MODEL_KINDS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--brainiac-checkpoint", type=Path)
    parser.add_argument("--brainiac-root", type=Path)
    parser.add_argument("--device", default="cuda:0" if extraction.torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    output = embed_study(
        args.study_id, args.processed_root, args.output_root, model_kind=args.model_kind,
        checkpoint=args.checkpoint, brainiac_checkpoint=args.brainiac_checkpoint,
        device=args.device, brainiac_root=args.brainiac_root,
    )
    print(f"Saved {args.model_kind}: {output}", flush=True)


if __name__ == "__main__":
    main()
