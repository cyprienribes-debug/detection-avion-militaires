import os
import json
import math
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DOSSIER_IMAGES = "images"
DOSSIER_XML = "annotations"
TAILLE_IMAGE = 256
GRILLE = 16
BATCH_SIZE = 16

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

reprendre = False
tmp = None
if os.path.exists("modele_detection.keras"):
    tmp = tf.keras.models.load_model("modele_detection.keras", compile=False)
    if int(tmp.input_shape[1]) == TAILLE_IMAGE and int(tmp.output_shape[1]) == GRILLE:
        rep = input("Reprendre l'entrainement depuis modele_detection.keras (256x256) ? (o/n) : ").strip().lower()
        reprendre = (rep == "o")
    else:
        print(f"Ancien modele detecte en {tmp.input_shape[1]}x{tmp.input_shape[2]} : retour propre a l'architecture 256x256 (grille 16x16).")


def charger_donnees(liste_xml):
    images = []
    boites_par_img = []
    for f_xml in liste_xml:
        chemin_img = os.path.join(DOSSIER_IMAGES, f_xml.replace(".xml", ".jpg"))
        if not os.path.exists(chemin_img):
            continue
        img_bgr = cv2.imread(chemin_img)
        if img_bgr is None:
            continue
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_redim = cv2.resize(img_rgb, (TAILLE_IMAGE, TAILLE_IMAGE), interpolation=cv2.INTER_AREA)

        arbre = ET.parse(os.path.join(DOSSIER_XML, f_xml))
        boites = []
        for obj in arbre.getroot().findall("object"):
            bnd = obj.find("bndbox")
            x1 = max(0.0, min(1.0, float(bnd.find("xmin").text) / w))
            y1 = max(0.0, min(1.0, float(bnd.find("ymin").text) / h))
            x2 = max(0.0, min(1.0, float(bnd.find("xmax").text) / w))
            y2 = max(0.0, min(1.0, float(bnd.find("ymax").text) / h))
            if (x2 - x1) > 0.01 and (y2 - y1) > 0.01:
                boites.append((x1, y1, x2, y2))

        images.append(img_redim)
        boites_par_img.append(boites)
    return images, boites_par_img


def encoder_grille(boites):
    cible = np.zeros((GRILLE, GRILLE, 5), dtype="float32")
    for (x1, y1, x2, y2) in boites:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        gx = min(GRILLE - 1, max(0, int(cx * GRILLE)))
        gy = min(GRILLE - 1, max(0, int(cy * GRILLE)))
        tx = cx * GRILLE - gx
        ty = cy * GRILLE - gy
        cible[gy, gx] = [1.0, tx, ty, bw, bh]
    return cible


def augmenter_image_et_boites(img, boites):
    img_aug = img.copy()
    b_aug = list(boites)

    if random.random() < 0.5:
        img_aug = cv2.flip(img_aug, 1)
        b_aug = [(1.0 - x2, y1, 1.0 - x1, y2) for (x1, y1, x2, y2) in b_aug]

    if random.random() < 0.5:
        img_aug = cv2.flip(img_aug, 0)
        b_aug = [(x1, 1.0 - y2, x2, 1.0 - y1) for (x1, y1, x2, y2) in b_aug]

    k = random.randint(0, 3)
    if k == 1:
        img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_CLOCKWISE)
        b_aug = [(1.0 - y2, x1, 1.0 - y1, x2) for (x1, y1, x2, y2) in b_aug]
    elif k == 2:
        img_aug = cv2.rotate(img_aug, cv2.ROTATE_180)
        b_aug = [(1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1) for (x1, y1, x2, y2) in b_aug]
    elif k == 3:
        img_aug = cv2.rotate(img_aug, cv2.ROTATE_90_COUNTERCLOCKWISE)
        b_aug = [(y1, 1.0 - x2, y2, 1.0 - x1) for (x1, y1, x2, y2) in b_aug]

    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12.0, 12.0)
        img_aug = np.clip(img_aug.astype("float32") * alpha + beta, 0, 255).astype("uint8")

    return img_aug, b_aug


class GenerateurDetection(tf.keras.utils.Sequence):
    def __init__(self, images, boites_list, batch_size=16, augmenter=False, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.boites_list = boites_list
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.indices = np.arange(len(self.images))
        if self.augmenter:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = np.empty((len(batch_ids), TAILLE_IMAGE, TAILLE_IMAGE, 3), dtype="uint8")
        y_batch = np.empty((len(batch_ids), GRILLE, GRILLE, 5), dtype="float32")
        for i, pos in enumerate(batch_ids):
            img = self.images[pos]
            boites = self.boites_list[pos]
            if self.augmenter:
                img, boites = augmenter_image_et_boites(img, boites)
            X_batch[i] = img
            y_batch[i] = encoder_grille(boites)
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.augmenter:
            np.random.shuffle(self.indices)


def perte_detection(y_true, y_pred):
    obj_true = y_true[..., 0:1]
    obj_pred = tf.clip_by_value(y_pred[..., 0:1], 1e-6, 1.0 - 1e-6)

    # Focal BCE equilibree (0.80 objets / 0.20 fond) -> convergence rapide et fiable
    bce = -(obj_true * tf.math.log(obj_pred) + (1.0 - obj_true) * tf.math.log(1.0 - obj_pred))
    pt = obj_true * obj_pred + (1.0 - obj_true) * (1.0 - obj_pred)
    alpha = obj_true * 0.80 + (1.0 - obj_true) * 0.20
    focal = alpha * tf.pow(1.0 - pt, 2.0) * bce
    perte_obj = tf.reduce_mean(tf.reduce_sum(focal, axis=[1, 2, 3]))

    # Erreur sur centre et taille uniquement sur les cases contenant un avion
    diff_xy = tf.square(y_true[..., 1:3] - y_pred[..., 1:3])
    diff_wh = tf.square(
        tf.sqrt(tf.maximum(y_true[..., 3:5], 1e-6)) - tf.sqrt(tf.maximum(y_pred[..., 3:5], 1e-6))
    )
    err_boite = tf.reduce_sum(diff_xy, axis=-1, keepdims=True) + 2.0 * tf.reduce_sum(diff_wh, axis=-1, keepdims=True)
    nb_pos = tf.maximum(tf.reduce_sum(obj_true, axis=[1, 2, 3]), 1.0)
    perte_boite = tf.reduce_mean(tf.reduce_sum(obj_true * err_boite, axis=[1, 2, 3]) / nb_pos)

    return perte_obj + 5.0 * perte_boite


def construire_modele():
    entree = layers.Input(shape=(TAILLE_IMAGE, TAILLE_IMAGE, 3), name="image_entree")
    x = layers.Rescaling(1.0 / 255.0)(entree)

    x = layers.Conv2D(24, (3, 3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Conv2D(48, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(80, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(80, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    p1 = layers.MaxPooling2D((3, 3), strides=1, padding="same")(x)
    p2 = layers.MaxPooling2D((5, 5), strides=1, padding="same")(x)
    x = layers.Concatenate()([x, p1, p2])

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.2)(x)

    sortie = layers.Conv2D(5, (1, 1), activation="sigmoid", name="grille_detection")(x)
    return models.Model(inputs=entree, outputs=sortie)


def calculer_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return (inter / union) if union > 0 else 0.0


def decoder_grille(pred, seuil=0.50, seuil_nms=0.35):
    cands = []
    for gy in range(GRILLE):
        for gx in range(GRILLE):
            score = float(pred[gy, gx, 0])
            if score < seuil:
                continue
            tx, ty, bw, bh = pred[gy, gx, 1:5]
            cx = (gx + float(tx)) / GRILLE
            cy = (gy + float(ty)) / GRILLE
            x1 = max(0.0, cx - float(bw) / 2.0)
            y1 = max(0.0, cy - float(bh) / 2.0)
            x2 = min(1.0, cx + float(bw) / 2.0)
            y2 = min(1.0, cy + float(bh) / 2.0)
            if (x2 - x1) > 0.015 and (y2 - y1) > 0.015:
                cands.append((x1, y1, x2, y2, score))
    cands.sort(key=lambda d: d[4], reverse=True)
    finales = []
    for c in cands:
        if all(calculer_iou(c[:4], f[:4]) < seuil_nms for f in finales):
            finales.append(c)
    return finales


def extraire_perte(res):
    return float(res[0]) if isinstance(res, (list, tuple)) else float(res)


if os.path.exists("split_images.json"):
    with open("split_images.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    xml_train, xml_val = split_data["train"], split_data["val"]
else:
    tous_xml = sorted([
        f for f in os.listdir(DOSSIER_XML)
        if f.endswith(".xml") and os.path.exists(os.path.join(DOSSIER_IMAGES, f.replace(".xml", ".jpg")))
    ])
    random.shuffle(tous_xml)
    s = int(0.8 * len(tous_xml))
    xml_train, xml_val = tous_xml[:s], tous_xml[s:]
    with open("split_images.json", "w", encoding="utf-8") as f:
        json.dump({"train": xml_train, "val": xml_val}, f, indent=2)

print("Chargement des images de detection...")
imgs_train, boites_train = charger_donnees(xml_train)
imgs_val, boites_val = charger_donnees(xml_val)
print(f"Pret : {len(imgs_train)} images (train) | {len(imgs_val)} images (val).")

gen_train = GenerateurDetection(imgs_train, boites_train, batch_size=BATCH_SIZE, augmenter=True)
gen_val = GenerateurDetection(imgs_val, boites_val, batch_size=BATCH_SIZE, augmenter=False)

if reprendre:
    print("Reprise depuis modele_detection.keras (learning rate = 3e-4)...")
    modele = tmp
    lr_depart = 3e-4
    nb_epochs = 20
else:
    print("Creation du detecteur Mini-YOLO 256x256 (learning rate = 1e-3)...")
    modele = construire_modele()
    lr_depart = 1e-3
    nb_epochs = 25

modele.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_depart),
    loss=perte_detection
)

perte_avant = extraire_perte(modele.evaluate(gen_val, verbose=0)) if reprendre else float("inf")
if reprendre:
    print(f"val_loss actuelle avant reprise : {perte_avant:.4f}")

callbacks = [
    ModelCheckpoint(
        "modele_detection.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        initial_value_threshold=perte_avant if reprendre else None,
        verbose=1
    ),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

historique = None
try:
    historique = modele.fit(
        gen_train,
        validation_data=gen_val,
        epochs=nb_epochs,
        callbacks=callbacks
    )
except KeyboardInterrupt:
    print("\nArret demande : le meilleur modele atteint est deja sauvegarde dans modele_detection.keras.")

modele = tf.keras.models.load_model("modele_detection.keras", compile=False)

if historique is not None:
    hist_courant = {k: [float(v) for v in vals] for k, vals in historique.history.items()}
    if reprendre and os.path.exists("historique_detection.json"):
        with open("historique_detection.json", "r", encoding="utf-8") as f:
            ancien_hist = json.load(f)
        for cle in ["loss", "val_loss"]:
            if cle in ancien_hist and cle in hist_courant:
                hist_courant[cle] = ancien_hist[cle] + hist_courant[cle]
    with open("historique_detection.json", "w", encoding="utf-8") as f:
        json.dump(hist_courant, f, indent=4)

print("Calibration du seuil optimal sur le jeu de validation...")
X_val_np = np.array(imgs_val, dtype="uint8")
preds_val = modele.predict(X_val_np, batch_size=32, verbose=0)

meilleur_f1 = -1.0
meilleur_seuil = 0.50
meilleurs_scores = {}

for seuil in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    vp_tot, fp_tot, fn_tot = 0, 0, 0
    ious = []
    for i in range(len(imgs_val)):
        dets = decoder_grille(preds_val[i], seuil=seuil)
        vrais = boites_val[i]
        trouves = set()
        vp = 0
        for d in dets:
            best_iou, best_j = 0.0, -1
            for j, vb in enumerate(vrais):
                if j in trouves:
                    continue
                iou = calculer_iou(d[:4], vb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.40 and best_j != -1:
                vp += 1
                trouves.add(best_j)
                ious.append(best_iou)
        vp_tot += vp
        fp_tot += len(dets) - vp
        fn_tot += len(vrais) - vp

    prec = vp_tot / (vp_tot + fp_tot) if (vp_tot + fp_tot) > 0 else 0.0
    rap = vp_tot / (vp_tot + fn_tot) if (vp_tot + fn_tot) > 0 else 0.0
    f1 = (2.0 * prec * rap / (prec + rap)) if (prec + rap) > 0 else 0.0

    if f1 > meilleur_f1:
        meilleur_f1 = f1
        meilleur_seuil = seuil
        meilleurs_scores = {
            "seuil_optimal": seuil,
            "precision": round(prec, 4),
            "rappel": round(rap, 4),
            "score_f1": round(f1, 4),
            "iou_moyen": round(float(np.mean(ious)) if ious else 0.0, 4)
        }

with open("config_detection.json", "w", encoding="utf-8") as f:
    json.dump(meilleurs_scores, f, indent=4)

print("-" * 50)
print(f"Seuil optimal retenu : {meilleur_seuil}")
print(f"Precision validation : {meilleurs_scores['precision'] * 100:.2f} %")
print(f"Rappel validation    : {meilleurs_scores['rappel'] * 100:.2f} %")
print(f"Score F1 validation  : {meilleurs_scores['score_f1'] * 100:.2f} %")
print(f"IoU moyen            : {meilleurs_scores['iou_moyen'] * 100:.2f} %")
print("-" * 50)