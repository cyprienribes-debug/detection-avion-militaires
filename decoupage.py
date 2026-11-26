import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
import cv2

DOSSIER_IMAGES = "images"
DOSSIER_XML = "annotations"
DOSSIER_SORTIE = "dataset_classification"
TAILLE_VIGNETTE = 128
MARGE = 0.15

random.seed(42)

def extraire_vignette_carree(image, xmin, ymin, xmax, ymax, taille=128, marge=0.15):
    h_img, w_img, _ = image.shape
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cote = max(xmax - xmin, ymax - ymin) * (1.0 + 2.0 * marge)
    demi = max(8, int(round(cote / 2.0)))

    pad = demi + 4
    img_pad = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    pcx = int(round(cx)) + pad
    pcy = int(round(cy)) + pad

    crop = img_pad[pcy - demi : pcy + demi, pcx - demi : pcx + demi]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (taille, taille), interpolation=cv2.INTER_AREA)

fichiers_xml = sorted([
    f for f in os.listdir(DOSSIER_XML)
    if f.endswith(".xml") and os.path.exists(os.path.join(DOSSIER_IMAGES, f.replace(".xml", ".jpg")))
])

random.shuffle(fichiers_xml)
split_idx = int(0.8 * len(fichiers_xml))
split_data = {
    "train": fichiers_xml[:split_idx],
    "val": fichiers_xml[split_idx:]
}

with open("split_images.json", "w", encoding="utf-8") as f:
    json.dump(split_data, f, indent=2)

if os.path.exists(DOSSIER_SORTIE):
    shutil.rmtree(DOSSIER_SORTIE)

compteurs = {"train": 0, "val": 0}

for partie in ["train", "val"]:
    for f_xml in split_data[partie]:
        nom_base = f_xml.replace(".xml", "")
        img = cv2.imread(os.path.join(DOSSIER_IMAGES, f"{nom_base}.jpg"))
        if img is None:
            continue

        arbre = ET.parse(os.path.join(DOSSIER_XML, f_xml))
        for k, obj in enumerate(arbre.getroot().findall("object")):
            cat = obj.find("name").text.strip()
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            if (xmax - xmin) <= 6 or (ymax - ymin) <= 6:
                continue

            vignette = extraire_vignette_carree(img, xmin, ymin, xmax, ymax, TAILLE_VIGNETTE, MARGE)
            if vignette is None:
                continue

            dossier_cat = os.path.join(DOSSIER_SORTIE, partie, cat)
            os.makedirs(dossier_cat, exist_ok=True)
            chemin_out = os.path.join(dossier_cat, f"{nom_base}_{k}.jpg")
            cv2.imwrite(chemin_out, vignette)
            compteurs[partie] += 1

print(f"Decoupage termine : {compteurs['train']} vignettes train | {compteurs['val']} vignettes val.")
print("Separation des images sauvegardee dans split_images.json")