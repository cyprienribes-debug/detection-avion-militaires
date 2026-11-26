import os
import json
import math
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DOSSIER_TRAIN = os.path.join("dataset_classification", "train")
DOSSIER_VAL = os.path.join("dataset_classification", "val")
TAILLE_IMAGE = 128
BATCH_SIZE = 32
MIN_PAR_CLASSE = 280
MIN_CLASSE_DIFFICILE = 420

noms_classes = sorted([
    d for d in os.listdir(DOSSIER_TRAIN)
    if os.path.isdir(os.path.join(DOSSIER_TRAIN, d))
])
nb_classes = len(noms_classes)

with open("classes.txt", "w", encoding="utf-8") as f:
    for c in noms_classes:
        f.write(c + "\n")


def lire_classes_difficiles(seuil_f1=0.82):
    difficiles = set()
    if os.path.exists("evaluation_classification.json"):
        with open("evaluation_classification.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for cat, stats in data.get("par_categorie", {}).items():
            if float(stats.get("score_f1", 1.0)) < seuil_f1:
                difficiles.add(cat)
    return difficiles


def charger_dossier(dossier, equilibrer=False, classes_difficiles=None):
    if classes_difficiles is None:
        classes_difficiles = set()

    X_list, y_list = [], []
    for idx, nom in enumerate(noms_classes):
        chemin_cat = os.path.join(dossier, nom)
        if not os.path.isdir(chemin_cat):
            continue
        fichiers = sorted([f for f in os.listdir(chemin_cat) if f.endswith(".jpg")])
        imgs_cat = []
        for f_img in fichiers:
            img_bgr = cv2.imread(os.path.join(chemin_cat, f_img))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[0] != TAILLE_IMAGE or img_rgb.shape[1] != TAILLE_IMAGE:
                img_rgb = cv2.resize(img_rgb, (TAILLE_IMAGE, TAILLE_IMAGE), interpolation=cv2.INTER_AREA)
            imgs_cat.append(img_rgb)

        if not imgs_cat:
            continue

        if equilibrer:
            cible = MIN_CLASSE_DIFFICILE if nom in classes_difficiles else MIN_PAR_CLASSE
            if len(imgs_cat) < cible:
                manque = cible - len(imgs_cat)
                imgs_cat.extend(random.choices(imgs_cat, k=manque))

        for im in imgs_cat:
            X_list.append(im)
            y_list.append(idx)

    X = np.array(X_list, dtype="uint8")
    y = tf.keras.utils.to_categorical(np.array(y_list, dtype="int32"), num_classes=nb_classes)
    return X, y


def augmenter_luminosite(img_rgb):
    """Augmente ou diminue aleatoirement la luminosite (exposition, gamma ou canal V en HSV)."""
    if random.random() < 0.20:
        return img_rgb

    img = img_rgb.astype("float32")
    mode = random.choice(["exposition", "gamma", "hsv"])

    if mode == "exposition":
        facteur = random.uniform(0.65, 1.35)
        decalage = random.uniform(-22.0, 22.0)
        return np.clip(img * facteur + decalage, 0, 255).astype("uint8")

    elif mode == "gamma":
        gamma = random.uniform(0.70, 1.40)
        img = np.power(np.clip(img / 255.0, 0.0, 1.0), gamma) * 255.0
        return np.clip(img, 0, 255).astype("uint8")

    else:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype("float32")
        hsv[..., 2] = np.clip(
            hsv[..., 2] * random.uniform(0.65, 1.35) + random.uniform(-18.0, 18.0),
            0, 255
        )
        return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)


class GenerateurClassification(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augmenter=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.indices = np.arange(len(self.X))
        if self.augmenter:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = np.empty((len(batch_ids), TAILLE_IMAGE, TAILLE_IMAGE, 3), dtype="uint8")
        y_batch = self.y[batch_ids]
        for i, pos in enumerate(batch_ids):
            im = self.X[pos]
            X_batch[i] = augmenter_luminosite(im) if self.augmenter else im
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.augmenter:
            np.random.shuffle(self.indices)


def bloc_res(x, filtres, stride=1):
    raccourci = x
    x = layers.Conv2D(filtres, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filtres, (3, 3), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or raccourci.shape[-1] != filtres:
        raccourci = layers.Conv2D(filtres, (1, 1), strides=stride, padding="same", use_bias=False)(raccourci)
        raccourci = layers.BatchNormalization()(raccourci)

    x = layers.Add()([x, raccourci])
    x = layers.Activation("relu")(x)
    return x


def construire_modele():
    entree = layers.Input(shape=(TAILLE_IMAGE, TAILLE_IMAGE, 3))

    x = layers.RandomFlip("horizontal_and_vertical")(entree)
    x = layers.RandomRotation(0.5, fill_mode="reflect")(x)
    x = layers.RandomTranslation(0.08, 0.08, fill_mode="reflect")(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness(0.15)(x)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, (3, 3), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = bloc_res(x, 48, stride=1)
    x = bloc_res(x, 80, stride=2)
    x = bloc_res(x, 128, stride=2)
    x = bloc_res(x, 192, stride=2)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])

    x = layers.Dropout(0.35)(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    sortie = layers.Dense(nb_classes, activation="softmax")(x)

    return models.Model(inputs=entree, outputs=sortie)


reprendre = False
if os.path.exists("modele_classification.keras"):
    rep = input("Reprendre l'entrainement depuis modele_classification.keras ? (o/n) : ").strip().lower()
    reprendre = (rep == "o")

if not reprendre:
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

classes_difficiles = lire_classes_difficiles(seuil_f1=0.82) if reprendre else set()
if reprendre and classes_difficiles:
    print(f"Renforcement cible sur les classes < 82% F1 : {sorted(classes_difficiles)}")

print("Chargement du jeu d'entrainement et de validation...")
X_train, y_train = charger_dossier(DOSSIER_TRAIN, equilibrer=True, classes_difficiles=classes_difficiles)
X_val, y_val = charger_dossier(DOSSIER_VAL, equilibrer=False)

perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]
print(f"Dataset pret : {len(X_train)} train | {len(X_val)} val ({nb_classes} classes).")

gen_train = GenerateurClassification(X_train, y_train, batch_size=BATCH_SIZE, augmenter=True)

if reprendre:
    print("Reprise depuis modele_classification.keras (learning rate = 1e-4)...")
    modele = tf.keras.models.load_model("modele_classification.keras", compile=False)
    lr_depart = 1e-4
    nb_epochs = 12
else:
    print("Creation d'un nouveau ResNet maison (learning rate = 1e-3)...")
    modele = construire_modele()
    lr_depart = 1e-3
    nb_epochs = 12

modele.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_depart, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05 if reprendre else 0.1),
    metrics=["accuracy"]
)

acc_avant = float(modele.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)[1]) if reprendre else 0.0
if reprendre:
    print(f"val_accuracy actuelle avant reprise (sans TTA) : {acc_avant * 100:.2f} %")

callbacks = [
    ModelCheckpoint(
        "modele_classification.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        initial_value_threshold=acc_avant if reprendre else None,
        verbose=1
    ),
    EarlyStopping(monitor="val_accuracy", mode="max", patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=3, min_lr=5e-6, verbose=1)
]

historique = None
try:
    historique = modele.fit(
        gen_train,
        validation_data=(X_val, y_val),
        epochs=nb_epochs,
        callbacks=callbacks
    )
except KeyboardInterrupt:
    print("\nArret demande : le meilleur modele atteint est deja sauvegarde sur le disque.")

if historique is not None:
    hist_courant = {k: [float(v) for v in vals] for k, vals in historique.history.items()}
    if reprendre and os.path.exists("historique_classification.json"):
        with open("historique_classification.json", "r", encoding="utf-8") as f:
            ancien_hist = json.load(f)
        for cle in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            if cle in ancien_hist and cle in hist_courant:
                hist_courant[cle] = ancien_hist[cle] + hist_courant[cle]
    with open("historique_classification.json", "w", encoding="utf-8") as f:
        json.dump(hist_courant, f, indent=4)