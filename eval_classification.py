import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

modele = tf.keras.models.load_model("modele_classification.keras", compile=False)

DOSSIER_VAL = os.path.join("dataset_classification", "val")
TAILLE_IMAGE = int(modele.input_shape[1])

dataset_val = image_dataset_from_directory(
    DOSSIER_VAL,
    image_size=(TAILLE_IMAGE, TAILLE_IMAGE),
    batch_size=32,
    shuffle=False
)

noms_classes = dataset_val.class_names
nb_classes = len(noms_classes)

y_vrais = []
y_preds = []

print("Evaluation avec TTA complet (8 symetries : 4 rotations + 4 miroirs)...")
for images_batch, labels_batch in dataset_val:
    imgs = images_batch.numpy()
    imgs_flip = imgs[:, :, ::-1, :]

    vues = [
        imgs,
        np.rot90(imgs, k=1, axes=(1, 2)),
        np.rot90(imgs, k=2, axes=(1, 2)),
        np.rot90(imgs, k=3, axes=(1, 2)),
        imgs_flip,
        np.rot90(imgs_flip, k=1, axes=(1, 2)),
        np.rot90(imgs_flip, k=2, axes=(1, 2)),
        np.rot90(imgs_flip, k=3, axes=(1, 2)),
    ]
    preds = sum(modele.predict(v, verbose=0) for v in vues) / 8.0

    y_vrais.extend(labels_batch.numpy().tolist())
    y_preds.extend(np.argmax(preds, axis=1).tolist())

y_vrais = np.array(y_vrais)
y_preds = np.array(y_preds)

accuracy_globale = float(np.mean(y_vrais == y_preds))
matrice = np.zeros((nb_classes, nb_classes), dtype=int)
for v, p in zip(y_vrais, y_preds):
    matrice[v, p] += 1

bilan_par_classe = {}
precisions, rappels, f1_scores = [], [], []

for idx, nom in enumerate(noms_classes):
    vp = int(np.sum((y_preds == idx) & (y_vrais == idx)))
    fp = int(np.sum((y_preds == idx) & (y_vrais != idx)))
    fn = int(np.sum((y_preds != idx) & (y_vrais == idx)))
    total_classe = int(np.sum(y_vrais == idx))

    prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rap = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * prec * rap / (prec + rap)) if (prec + rap) > 0 else 0.0

    precisions.append(prec)
    rappels.append(rap)
    f1_scores.append(f1)

    bilan_par_classe[nom] = {
        "total_validation": total_classe,
        "corrects": vp,
        "precision": round(prec, 4),
        "rappel": round(rap, 4),
        "score_f1": round(f1, 4)
    }

rapport_json = {
    "modele": "modele_classification.keras",
    "metriques_globales": {
        "accuracy": round(accuracy_globale, 4),
        "precision_moyenne": round(float(np.mean(precisions)), 4),
        "rappel_moyen": round(float(np.mean(rappels)), 4),
        "score_f1_moyen": round(float(np.mean(f1_scores)), 4)
    },
    "classes": noms_classes,
    "matrice_confusion": matrice.tolist(),
    "par_categorie": bilan_par_classe
}

with open("evaluation_classification.json", "w", encoding="utf-8") as f:
    json.dump(rapport_json, f, indent=4)

print("-" * 50)
print(f"Accuracy globale : {accuracy_globale * 100:.2f} %")
print(f"Precision moy.   : {np.mean(precisions) * 100:.2f} %")
print(f"Rappel moyen     : {np.mean(rappels) * 100:.2f} %")
print(f"Score F1 moyen   : {np.mean(f1_scores) * 100:.2f} %")
print("-" * 50)