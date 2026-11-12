"""Quantização INT8 dinâmica dos dois modelos ONNX (detector e classificador).

Quantização dinâmica: os pesos são guardados em INT8 e as ativações são quantizadas em tempo de
execução, sem conjunto de calibração. Reduz o arquivo para cerca de 1/4 do FP32. Antes de usar o
modelo quantizado, conferir com `evaluate_onnx.py --detector ... --classifier ...` que a Tabela 5
se mantém.

Resultado (20 metáfases de teste, CPU, contra o FP32): o classificador INT8 manteve o resultado
(22,6 MB, contra 89,8 MB), mas o detector INT8 caiu de 80% para 40% de metáfases com o total correto,
com ou sem `per_channel`. Na CPU, os dois INT8 ficaram mais lentos que o FP32 (ConvInteger). Por isso
o site usa os dois modelos em FP32.

Uso:
    uv run --no-active --group classifier python src/inference/quantize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_pipeline import CLASSIFIER_ONNX, DETECTOR_ONNX
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic


def quantize(model_path: Path) -> Path:
    prepared = model_path.with_name(f"{model_path.stem}.prep.onnx")
    out_path = model_path.with_name(f"{model_path.stem}.int8.onnx")
    # otimiza o grafo e infere formas antes de quantizar (recomendado pelo onnxruntime)
    quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
    quantize_dynamic(str(prepared), str(out_path), weight_type=QuantType.QInt8)
    prepared.unlink()
    size_in, size_out = model_path.stat().st_size / 2**20, out_path.stat().st_size / 2**20
    print(f"{model_path.name}: {size_in:.1f} MB -> {out_path.name}: {size_out:.1f} MB")
    return out_path


if __name__ == "__main__":
    for path in (DETECTOR_ONNX, CLASSIFIER_ONNX):
        quantize(path)
