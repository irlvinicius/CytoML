"""Converte anotações Pascal VOC XML (dataset Kaggle karyotype) para o formato YOLO.

O dataset original anota 24 classes reais por par homólogo cromossômico
(nomenclatura de grupo de Denver), mas o `dataset.yaml` atual usa `nc: 1`
(detecção de classe única "chromosome") — decisão do TCC1 para focar em
segmentação/isolamento. `DENVER_CLASSES` preserva o mapa das 24 classes reais
para reuso futuro (Fase 4: classificação), mas não é usado para gerar labels
aqui.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

DENVER_CLASSES = [
    "A1", "A2", "A3",
    "B4", "B5",
    "C6", "C7", "C8", "C9", "C10", "C11", "C12",
    "D13", "D14", "D15",
    "E16", "E17", "E18",
    "F19", "F20",
    "G21", "G22",
    "X", "Y",
]
DENVER_CLASS_TO_ID = {name: idx for idx, name in enumerate(DENVER_CLASSES)}


def parse_annotation(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        objects.append(
            {
                "name": name,
                "xmin": float(bndbox.find("xmin").text),
                "ymin": float(bndbox.find("ymin").text),
                "xmax": float(bndbox.find("xmax").text),
                "ymax": float(bndbox.find("ymax").text),
            }
        )

    return {"width": width, "height": height, "objects": objects}


def to_yolo_line(obj: dict, img_w: int, img_h: int, class_id: int = 0) -> str:
    x_center = (obj["xmin"] + obj["xmax"]) / 2 / img_w
    y_center = (obj["ymin"] + obj["ymax"]) / 2 / img_h
    box_w = (obj["xmax"] - obj["xmin"]) / img_w
    box_h = (obj["ymax"] - obj["ymin"]) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def convert_annotation(xml_path: Path, out_txt_path: Path, class_id: int = 0) -> None:
    annotation = parse_annotation(xml_path)
    lines = [
        to_yolo_line(obj, annotation["width"], annotation["height"], class_id)
        for obj in annotation["objects"]
    ]
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines) + "\n")
