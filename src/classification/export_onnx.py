"""Exporta um checkpoint do classificador para ONNX, com batch dinâmico.

Confere a paridade numérica entre a saída do PyTorch e a do onnxruntime antes de
considerar o export válido (mesmo princípio do src/export/test_model_output.py do
detector, mas embutido aqui em vez de manual).

Uso:
    uv run python src/classification/export_onnx.py --checkpoint models/classifier/runs/resnet50-30epochs/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnxruntime as ort
import torch
from evaluate import load_checkpoint


def export(checkpoint_path: Path, image_size: int = 224, atol: float = 1e-3) -> Path:
    device = torch.device("cpu")
    model = load_checkpoint(checkpoint_path, device)

    out_path = checkpoint_path.with_suffix(".onnx")
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    if not np.allclose(torch_out, onnx_out, atol=atol):
        raise RuntimeError(f"Saída do ONNX diverge do PyTorch (diferença máxima: {max_diff})")

    print(f"Export OK: {out_path} (diferença máxima PyTorch x ONNX: {max_diff:.2e})")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    export(args.checkpoint, image_size=args.image_size)
