import os
import json
import time
import random
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

DOSSIER_IMAGES = "images"
DOSSIER_XML = "annotations"
DOSSIER_VAL_CLF = os.path.join("dataset_classification", "val")
DOSSIER_TRAIN_CLF = os.path.join("dataset_classification", "train")
TAILLE_DET = 256
TAILLE_CLF = 128
GRILLE = 16


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


def extraire_vignette_carree(image_rgb, xmin, ymin, xmax, ymax, taille=128, marge=0.15):
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cote = max(xmax - xmin, ymax - ymin) * (1.0 + 2.0 * marge)
    demi = max(8, int(round(cote / 2.0)))
    pad = demi + 4
    img_pad = cv2.copyMakeBorder(image_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    pcx, pcy = int(round(cx)) + pad, int(round(cy)) + pad
    crop = img_pad[pcy - demi : pcy + demi, pcx - demi : pcx + demi]
    return cv2.resize(crop, (taille, taille), interpolation=cv2.INTER_AREA)


def inverser_boite_rotation(x1, y1, x2, y2, k):
    if k == 0:
        return x1, y1, x2, y2
    elif k == 1:
        return y1, 1.0 - x2, y2, 1.0 - x1
    elif k == 2:
        return 1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1
    else:
        return 1.0 - y2, x1, 1.0 - y1, x2


def decoder_une_vue(pred, k, seuil_cand=0.30, seuil_nms=0.35):
    cands = []
    for gy in range(GRILLE):
        for gx in range(GRILLE):
            score = float(pred[gy, gx, 0])
            if score < seuil_cand:
                continue
            tx, ty, bw, bh = pred[gy, gx, 1:5]
            cx = (gx + float(tx)) / GRILLE
            cy = (gy + float(ty)) / GRILLE
            x1 = max(0.0, cx - float(bw) / 2.0)
            y1 = max(0.0, cy - float(bh) / 2.0)
            x2 = min(1.0, cx + float(bw) / 2.0)
            y2 = min(1.0, cy + float(bh) / 2.0)
            if (x2 - x1) > 0.015 and (y2 - y1) > 0.015:
                rx1, ry1, rx2, ry2 = inverser_boite_rotation(x1, y1, x2, y2, k)
                cands.append((rx1, ry1, rx2, ry2, score, k))

    cands.sort(key=lambda d: d[4], reverse=True)
    retenues = []
    for c in cands:
        if all(calculer_iou(c[:4], r[:4]) < seuil_nms for r in retenues):
            retenues.append(c)
    return retenues


def lire_verite_terrain(nom_image):
    vraies = []
    chemin_xml = os.path.join(DOSSIER_XML, nom_image.replace(".jpg", ".xml"))
    if os.path.exists(chemin_xml):
        arbre = ET.parse(chemin_xml)
        for obj in arbre.getroot().findall("object"):
            nom = obj.find("name").text.strip()
            bnd = obj.find("bndbox")
            vraies.append((
                float(bnd.find("xmin").text),
                float(bnd.find("ymin").text),
                float(bnd.find("xmax").text),
                float(bnd.find("ymax").text),
                nom
            ))
    return vraies


def dessiner_rect_pointille(img, pt1, pt2, couleur, epaisseur=2, tiret=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, tiret * 2):
        cv2.line(img, (x, y1), (min(x + tiret, x2), y1), couleur, epaisseur)
        cv2.line(img, (x, y2), (min(x + tiret, x2), y2), couleur, epaisseur)
    for y in range(y1, y2, tiret * 2):
        cv2.line(img, (x1, y), (x1, min(y + tiret, y2)), couleur, epaisseur)
        cv2.line(img, (x2, y), (x2, min(y + tiret, y2)), couleur, epaisseur)


class InterfaceTestAvions:
    def __init__(self, root):
        self.root = root
        self.root.title("Test Interactif - Détection & Classification d'Avions Militaires")
        self.root.geometry("1220x780")
        self.root.configure(bg="#1e1e2e")

        self.modele_det = None
        self.modele_clf = None
        self.categories = []
        self.images_val = []
        self.toutes_images = []
        self.vignettes_val = []
        self.vignettes_toutes = []
        self.photo_tk = None

        self.img_courante_rgb = None
        self.nom_courant = ""
        self.vraies_courantes = []
        self.vraie_classe_courante = None
        self.cache_preds_det = None
        self.cache_tta_det = False
        self.dernier_temps_ms = 0.0
        self.mode_precedent = "pipeline"

        self.seuil_defaut = 0.50
        self.charger_ressources()

        self.var_mode = tk.StringVar(value="pipeline")
        self.var_tta = tk.BooleanVar(value=True)
        self.var_val_uniquement = tk.BooleanVar(value=True)
        self.var_afficher_xml = tk.BooleanVar(value=True)
        self.var_seuil = tk.DoubleVar(value=self.seuil_defaut)
        self.var_recherche = tk.StringVar(value="")

        self.construire_ui()
        self.root.bind("<space>", self._on_espace)
        self.root.after(150, self.tester_aleatoire)

    def charger_ressources(self):
        global TAILLE_DET, GRILLE, TAILLE_CLF
        print("Chargement des modèles en mémoire...")
        if os.path.exists("modele_detection.keras"):
            self.modele_det = tf.keras.models.load_model("modele_detection.keras", compile=False)
            TAILLE_DET = int(self.modele_det.input_shape[1])
            GRILLE = int(self.modele_det.output_shape[1])
            print(f" -> modele_detection.keras chargé ({TAILLE_DET}x{TAILLE_DET}, grille {GRILLE}x{GRILLE})")
        if os.path.exists("modele_classification.keras"):
            self.modele_clf = tf.keras.models.load_model("modele_classification.keras", compile=False)
            TAILLE_CLF = int(self.modele_clf.input_shape[1])
            print(f" -> modele_classification.keras chargé ({TAILLE_CLF}x{TAILLE_CLF})")
        if os.path.exists("classes.txt"):
            with open("classes.txt", "r", encoding="utf-8") as f:
                self.categories = [l.strip() for l in f if l.strip()]
        if os.path.exists("config_detection.json"):
            with open("config_detection.json", "r", encoding="utf-8") as f:
                self.seuil_defaut = float(json.load(f).get("seuil_optimal", 0.50))

        if os.path.isdir(DOSSIER_IMAGES):
            self.toutes_images = sorted([f for f in os.listdir(DOSSIER_IMAGES) if f.endswith(".jpg")])
        if os.path.exists("split_images.json"):
            with open("split_images.json", "r", encoding="utf-8") as f:
                self.images_val = [
                    x.replace(".xml", ".jpg")
                    for x in json.load(f).get("val", [])
                    if os.path.exists(os.path.join(DOSSIER_IMAGES, x.replace(".xml", ".jpg")))
                ]
        if not self.images_val:
            self.images_val = list(self.toutes_images)

        self.vignettes_val = self._lister_vignettes(DOSSIER_VAL_CLF)
        self.vignettes_toutes = self.vignettes_val + self._lister_vignettes(DOSSIER_TRAIN_CLF)

    def _lister_vignettes(self, dossier):
        res = []
        if not os.path.isdir(dossier):
            return res
        for cat in sorted(os.listdir(dossier)):
            chemin_cat = os.path.join(dossier, cat)
            if os.path.isdir(chemin_cat):
                for f in os.listdir(chemin_cat):
                    if f.endswith(".jpg"):
                        res.append((os.path.join(chemin_cat, f), cat))
        return res

    def construire_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("Card.TFrame", background="#28283c")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#28283c", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#28283c", foreground="#89b4fa", font=("Segoe UI", 11, "bold"))
        style.configure("TRadiobutton", background="#28283c", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#28283c", foreground="#cdd6f4", font=("Segoe UI", 10))

        cadre_gauche = tk.Frame(self.root, bg="#181825", bd=0)
        cadre_gauche.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.lbl_bandeau = tk.Label(
            cadre_gauche,
            text="Prêt — Appuie sur [Espace] pour une image aléatoire",
            bg="#181825", fg="#a6e3a1", font=("Segoe UI", 11, "bold"), anchor="w"
        )
        self.lbl_bandeau.pack(fill="x", padx=10, pady=(8, 4))

        self.canvas_img = tk.Canvas(cadre_gauche, bg="#11111b", highlightthickness=0)
        self.canvas_img.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cadre_droit = tk.Frame(self.root, bg="#1e1e2e", width=385)
        cadre_droit.pack(side="right", fill="y", padx=(0, 12), pady=12)
        cadre_droit.pack_propagate(False)

        c1 = ttk.Frame(cadre_droit, style="Card.TFrame", padding=12)
        c1.pack(fill="x", pady=(0, 10))
        ttk.Label(c1, text="1. Modèle à tester", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        modes = [
            ("Pipeline complet (Détection + Classif.)", "pipeline"),
            ("Détection seule (Mini-YOLO)", "detection"),
            ("Classification seule (ResNet)", "classification"),
        ]
        for texte, val in modes:
            ttk.Radiobutton(
                c1, text=texte, value=val, variable=self.var_mode,
                command=self.changer_mode
            ).pack(anchor="w", pady=2)

        c2 = ttk.Frame(cadre_droit, style="Card.TFrame", padding=12)
        c2.pack(fill="x", pady=(0, 10))
        ttk.Label(c2, text="2. Sélection d'image", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        tk.Button(
            c2, text="Image aléatoire  [Espace]",
            bg="#89b4fa", fg="#11111b", activebackground="#b4befe",
            font=("Segoe UI", 11, "bold"), relief="flat", pady=8, cursor="hand2",
            command=self.tester_aleatoire
        ).pack(fill="x", pady=(0, 8))

        ligne_rech = tk.Frame(c2, bg="#28283c")
        ligne_rech.pack(fill="x", pady=(0, 6))
        tk.Label(
            ligne_rech, text="N° :", bg="#28283c", fg="#a6adc8", font=("Segoe UI", 9)
        ).pack(side="left", padx=(0, 4))
        self.entree_nom = tk.Entry(
            ligne_rech, textvariable=self.var_recherche,
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            relief="flat", font=("Segoe UI", 10)
        )
        self.entree_nom.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
        self.entree_nom.bind("<Return>", lambda e: self.tester_par_nom())

        tk.Button(
            ligne_rech, text="Charger", bg="#45475a", fg="#cdd6f4",
            activebackground="#585b70", relief="flat", padx=10, pady=4, cursor="hand2",
            command=self.tester_par_nom
        ).pack(side="right")

        tk.Button(
            c2, text="Ouvrir une image du PC...",
            bg="#313244", fg="#cdd6f4", activebackground="#45475a",
            font=("Segoe UI", 9), relief="flat", pady=5, cursor="hand2",
            command=self.ouvrir_image_externe
        ).pack(fill="x")

        c3 = ttk.Frame(cadre_droit, style="Card.TFrame", padding=12)
        c3.pack(fill="x", pady=(0, 10))
        ttk.Label(c3, text="3. Options", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        ttk.Checkbutton(
            c3, text="Images de validation uniquement (non vues)",
            variable=self.var_val_uniquement
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            c3, text="Activer TTA (plus précis / un peu plus lent)",
            variable=self.var_tta, command=self.relancer_courante
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            c3, text="Afficher vérité terrain XML (pointillés verts)",
            variable=self.var_afficher_xml, command=self.redessiner_seulement
        ).pack(anchor="w", pady=2)

        ligne_seuil = tk.Frame(c3, bg="#28283c")
        ligne_seuil.pack(fill="x", pady=(8, 0))
        self.lbl_seuil = tk.Label(
            ligne_seuil, text=f"Seuil détection : {self.var_seuil.get():.2f}",
            bg="#28283c", fg="#cdd6f4", font=("Segoe UI", 9, "bold")
        )
        self.lbl_seuil.pack(anchor="w")

        self.slider_seuil = tk.Scale(
            c3, from_=0.20, to=0.85, resolution=0.05, orient="horizontal",
            variable=self.var_seuil, bg="#28283c", fg="#cdd6f4",
            troughcolor="#181825", highlightthickness=0, showvalue=False,
            command=self._on_seuil_move
        )
        self.slider_seuil.pack(fill="x")
        self.slider_seuil.bind("<ButtonRelease-1>", lambda e: self.recalculer_seuil())

        c4 = ttk.Frame(cadre_droit, style="Card.TFrame", padding=12)
        c4.pack(fill="both", expand=True)
        ttk.Label(c4, text="4. Résultats", style="Title.TLabel").pack(anchor="w", pady=(0, 6))

        self.txt_resultats = tk.Text(
            c4, bg="#181825", fg="#cdd6f4", font=("Consolas", 9),
            relief="flat", height=10, wrap="word", state="disabled"
        )
        self.txt_resultats.pack(fill="both", expand=True)

        self.canvas_barres = tk.Canvas(c4, bg="#181825", height=115, highlightthickness=0)
        self.canvas_barres.pack(fill="x", pady=(8, 0))

    def _on_espace(self, event):
        if self.root.focus_get() != self.entree_nom:
            self.tester_aleatoire()

    def _on_seuil_move(self, val):
        self.lbl_seuil.config(text=f"Seuil détection : {float(val):.2f}")
        if self.var_mode.get() == "detection":
            self.recalculer_seuil()

    def changer_mode(self):
        nouveau = self.var_mode.get()
        ancien = self.mode_precedent
        self.mode_precedent = nouveau
        if (
            self.img_courante_rgb is not None
            and self.cache_preds_det is not None
            and ancien in ("pipeline", "detection")
            and nouveau in ("pipeline", "detection")
        ):
            self.recalculer_seuil()
        else:
            self.cache_preds_det = None
            self.tester_aleatoire()

    def relancer_courante(self):
        if self.img_courante_rgb is None:
            return
        self.cache_preds_det = None
        if self.var_mode.get() == "classification":
            self._evaluer_classification_courante()
        else:
            self._evaluer_detection_courante()

    def redessiner_seulement(self):
        if self.img_courante_rgb is not None and self.var_mode.get() != "classification":
            self.recalculer_seuil()

    def tester_aleatoire(self):
        mode = self.var_mode.get()
        if mode == "classification":
            if self.modele_clf is None:
                messagebox.showerror("Modèle manquant", "modele_classification.keras introuvable.")
                return
            pool = self.vignettes_val if self.var_val_uniquement.get() else self.vignettes_toutes
            if not pool:
                messagebox.showerror("Données manquantes", "Aucune vignette trouvée dans dataset_classification/.")
                return
            chemin, vraie_cat = random.choice(pool)
            self._charger_et_tester_vignette(chemin, vraie_cat)
        else:
            if self.modele_det is None:
                messagebox.showerror("Modèle manquant", "modele_detection.keras introuvable.")
                return
            pool = self.images_val if self.var_val_uniquement.get() else self.toutes_images
            if not pool:
                messagebox.showerror("Données manquantes", "Aucune image trouvée dans images/.")
                return
            nom_img = random.choice(pool)
            self._charger_et_tester_satellite(os.path.join(DOSSIER_IMAGES, nom_img), nom_img)

    def tester_par_nom(self):
        saisie = self.var_recherche.get().strip()
        if not saisie:
            self.tester_aleatoire()
            return
        if not saisie.endswith(".jpg"):
            saisie += ".jpg"
        chemin = os.path.join(DOSSIER_IMAGES, saisie)
        if not os.path.exists(chemin):
            messagebox.showwarning("Introuvable", f"Image {saisie} introuvable dans {DOSSIER_IMAGES}/.")
            return
        if self.var_mode.get() == "classification":
            self.var_mode.set("pipeline")
            self.mode_precedent = "pipeline"
        self.root.focus_set()
        self._charger_et_tester_satellite(chemin, saisie)

    def ouvrir_image_externe(self):
        chemin = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not chemin:
            return
        nom = os.path.basename(chemin)
        if self.var_mode.get() == "classification":
            self._charger_et_tester_vignette(chemin, None)
        else:
            self._charger_et_tester_satellite(chemin, nom, externe=True)

    def _charger_et_tester_vignette(self, chemin, vraie_cat):
        img_bgr = cv2.imread(chemin)
        if img_bgr is None:
            return
        self.img_courante_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.nom_courant = os.path.basename(chemin)
        self.vraie_classe_courante = vraie_cat
        self._evaluer_classification_courante()

    def _predire_batch_clf(self, vignettes_rgb, utiliser_tta):
        if not utiliser_tta:
            return self.modele_clf.predict(vignettes_rgb, verbose=0)
        v_flip = vignettes_rgb[:, :, ::-1, :]
        vues = [
            vignettes_rgb,
            np.rot90(vignettes_rgb, k=1, axes=(1, 2)),
            np.rot90(vignettes_rgb, k=2, axes=(1, 2)),
            np.rot90(vignettes_rgb, k=3, axes=(1, 2)),
            v_flip,
            np.rot90(v_flip, k=1, axes=(1, 2)),
            np.rot90(v_flip, k=2, axes=(1, 2)),
            np.rot90(v_flip, k=3, axes=(1, 2)),
        ]
        return sum(self.modele_clf.predict(v, verbose=0) for v in vues) / 8.0

    def _evaluer_classification_courante(self):
        entree = cv2.resize(self.img_courante_rgb, (TAILLE_CLF, TAILLE_CLF), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        probs = self._predire_batch_clf(np.expand_dims(entree, axis=0), self.var_tta.get())[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        top5_idx = np.argsort(probs)[::-1][:5]
        pred_cls = self.categories[top5_idx[0]] if self.categories else str(top5_idx[0])
        pred_conf = float(probs[top5_idx[0]])

        if self.vraie_classe_courante is not None:
            ok = (pred_cls == self.vraie_classe_courante)
            statut = "CORRECT" if ok else "ERREUR"
            couleur = "#a6e3a1" if ok else "#f38ba8"
            self.lbl_bandeau.config(
                text=f"[Classification] {self.nom_courant} — {statut} ({pred_cls} à {pred_conf*100:.1f}%, vrai : {self.vraie_classe_courante})",
                fg=couleur
            )
        else:
            self.lbl_bandeau.config(
                text=f"[Classification] {self.nom_courant} — Prédit : {pred_cls} ({pred_conf*100:.1f}%)",
                fg="#89b4fa"
            )

        self._afficher_image_canvas(self.img_courante_rgb, zoom_max=480)

        lignes = [
            f"Fichier       : {self.nom_courant}",
            f"Temps         : {dt_ms:.0f} ms ({'TTA x8' if self.var_tta.get() else '1 passe'})",
            f"Vraie classe  : {self.vraie_classe_courante or 'Inconnue'}",
            f"Prédit        : {pred_cls} ({pred_conf*100:.2f} %)",
            "",
            "Top 5 des probabilités :"
        ]
        top5_data = []
        for idx in top5_idx:
            nom_c = self.categories[idx] if self.categories else str(idx)
            p = float(probs[idx])
            marqueur = "  <-- VRAI" if nom_c == self.vraie_classe_courante else ""
            lignes.append(f"  {nom_c:<5} : {p*100:5.1f} %{marqueur}")
            top5_data.append((nom_c, p))

        self._ecrire_resultats("\n".join(lignes))
        self._dessiner_barres_top5(top5_data)

    def _charger_et_tester_satellite(self, chemin, nom_img, externe=False):
        img_bgr = cv2.imread(chemin)
        if img_bgr is None:
            return
        self.img_courante_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.nom_courant = nom_img
        self.vraies_courantes = [] if externe else lire_verite_terrain(nom_img)
        self.cache_preds_det = None
        self._evaluer_detection_courante()

    def _evaluer_detection_courante(self):
        base = cv2.resize(self.img_courante_rgb, (TAILLE_DET, TAILLE_DET), interpolation=cv2.INTER_AREA)
        utiliser_tta = self.var_tta.get()

        t0 = time.perf_counter()
        if utiliser_tta:
            batch_rot = np.array([
                base,
                cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(base, cv2.ROTATE_180),
                cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ], dtype="uint8")
            self.cache_preds_det = self.modele_det.predict(batch_rot, verbose=0)
        else:
            self.cache_preds_det = self.modele_det.predict(np.expand_dims(base, axis=0), verbose=0)

        self.cache_tta_det = utiliser_tta
        self.dernier_temps_ms = (time.perf_counter() - t0) * 1000.0
        self.recalculer_seuil()

    def _decoder_boites_depuis_cache(self, seuil, seuil_nms=0.35):
        h_img, w_img, _ = self.img_courante_rgb.shape
        if not self.cache_tta_det:
            cands = decoder_une_vue(self.cache_preds_det[0], 0, seuil_cand=seuil, seuil_nms=seuil_nms)
            return [
                (x1 * w_img, y1 * h_img, x2 * w_img, y2 * h_img, sc)
                for (x1, y1, x2, y2, sc, _) in cands
                if (x2 - x1) * w_img >= 12.0 and (y2 - y1) * h_img >= 12.0
            ]

        seuil_cand = max(0.25, seuil * 0.60)
        toutes = []
        for k in range(4):
            toutes.extend(decoder_une_vue(self.cache_preds_det[k], k, seuil_cand=seuil_cand, seuil_nms=seuil_nms))

        toutes.sort(key=lambda d: d[4], reverse=True)
        clusters = []
        for cand in toutes:
            meilleur_grp, meilleur_iou = None, 0.0
            for grp in clusters:
                if any(item[5] == cand[5] for item in grp):
                    continue
                iou = calculer_iou(cand[:4], grp[0][:4])
                if iou >= 0.35 and iou > meilleur_iou:
                    meilleur_iou = iou
                    meilleur_grp = grp
            if meilleur_grp is not None:
                meilleur_grp.append(cand)
            else:
                clusters.append([cand])

        boites_finales = []
        seuil_final = seuil * 0.92
        for grp in clusters:
            if len(grp) < 2:
                continue
            poids = np.array([d[4] for d in grp], dtype="float32")
            score_moy = float(np.sum(poids) / 4.0)
            if score_moy < seuil_final:
                continue
            w_norm = poids / np.sum(poids)
            fx1 = float(np.sum([d[0] * w_norm[i] for i, d in enumerate(grp)])) * w_img
            fy1 = float(np.sum([d[1] * w_norm[i] for i, d in enumerate(grp)])) * h_img
            fx2 = float(np.sum([d[2] * w_norm[i] for i, d in enumerate(grp)])) * w_img
            fy2 = float(np.sum([d[3] * w_norm[i] for i, d in enumerate(grp)])) * h_img
            if (fx2 - fx1) >= 12.0 and (fy2 - fy1) >= 12.0:
                boites_finales.append((fx1, fy1, fx2, fy2, score_moy))

        boites_finales.sort(key=lambda d: d[4], reverse=True)
        propres = []
        for b in boites_finales:
            if all(calculer_iou(b[:4], p[:4]) < seuil_nms for p in propres):
                propres.append(b)
        return propres

    def recalculer_seuil(self):
        if self.cache_preds_det is None or self.img_courante_rgb is None:
            return

        seuil = float(self.var_seuil.get())
        boites = self._decoder_boites_depuis_cache(seuil)

        mode = self.var_mode.get()
        t_cls_ms = 0.0
        detections = []

        if mode == "pipeline" and self.modele_clf is not None and len(boites) > 0:
            t0 = time.perf_counter()
            vignettes = np.array([
                extraire_vignette_carree(self.img_courante_rgb, x1, y1, x2, y2, TAILLE_CLF, 0.15)
                for (x1, y1, x2, y2, _) in boites
            ], dtype="uint8")
            preds_cls = self._predire_batch_clf(vignettes, self.var_tta.get())
            t_cls_ms = (time.perf_counter() - t0) * 1000.0
            for i, (x1, y1, x2, y2, sc) in enumerate(boites):
                idx_c = int(np.argmax(preds_cls[i]))
                nom_c = self.categories[idx_c] if self.categories else str(idx_c)
                detections.append((x1, y1, x2, y2, sc, nom_c, float(preds_cls[i][idx_c])))
        else:
            for (x1, y1, x2, y2, sc) in boites:
                detections.append((x1, y1, x2, y2, sc, "Avion", sc))

        vp, cls_ok = 0, 0
        trouves = set()
        ious = []
        for det in detections:
            best_iou, best_idx = 0.0, -1
            for idx_v, vbox in enumerate(self.vraies_courantes):
                if idx_v in trouves:
                    continue
                iou = calculer_iou(det[:4], vbox[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx_v
            if best_iou >= 0.40 and best_idx != -1:
                vp += 1
                trouves.add(best_idx)
                ious.append(best_iou)
                if det[5] == self.vraies_courantes[best_idx][4]:
                    cls_ok += 1

        fp = len(detections) - vp
        fn = len(self.vraies_courantes) - vp
        prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
        rap = vp / (vp + fn) if (vp + fn) > 0 else 0.0
        f1 = (2 * prec * rap / (prec + rap)) if (prec + rap) > 0 else 0.0
        iou_m = float(np.mean(ious)) if ious else 0.0

        visu, scale = self._preparer_image_affichage(self.img_courante_rgb)
        if self.var_afficher_xml.get():
            for (vx1, vy1, vx2, vy2, vnom) in self.vraies_courantes:
                p1 = (int(vx1 * scale), int(vy1 * scale))
                p2 = (int(vx2 * scale), int(vy2 * scale))
                dessiner_rect_pointille(visu, p1, p2, (80, 235, 110), epaisseur=2, tiret=7)
                cv2.putText(
                    visu, vnom, (p1[0], min(visu.shape[0] - 4, p2[1] + 13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 235, 110), 1, cv2.LINE_AA
                )

        for (dx1, dy1, dx2, dy2, sc_det, nom_cls, sc_cls) in detections:
            p1 = (int(dx1 * scale), int(dy1 * scale))
            p2 = (int(dx2 * scale), int(dy2 * scale))
            cv2.rectangle(visu, p1, p2, (245, 70, 90), 2)
            label = f"{nom_cls} {sc_det*100:.0f}%" if mode == "pipeline" else f"Avion {sc_det*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = max(th + 4, p1[1] - 3)
            cv2.rectangle(visu, (p1[0], ty - th - 4), (p1[0] + tw + 6, ty + 2), (245, 70, 90), -1)
            cv2.putText(visu, label, (p1[0] + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        self._afficher_image_prete(visu)

        tag_mode = "Pipeline" if mode == "pipeline" else "Détection"
        if self.vraies_courantes:
            coul = "#a6e3a1" if f1 >= 0.85 else "#f9e2af" if f1 >= 0.60 else "#f38ba8"
            self.lbl_bandeau.config(
                text=f"[{tag_mode}] {self.nom_courant} — F1 : {f1*100:.1f}% ({len(detections)}/{len(self.vraies_courantes)} avions)",
                fg=coul
            )
        else:
            self.lbl_bandeau.config(
                text=f"[{tag_mode}] {self.nom_courant} — {len(detections)} avion(s) détecté(s)",
                fg="#89b4fa"
            )

        lignes = [
            f"Image         : {self.nom_courant}",
            f"Temps         : {self.dernier_temps_ms + t_cls_ms:.0f} ms (det {self.dernier_temps_ms:.0f} + cls {t_cls_ms:.0f})",
            f"Seuil / TTA   : {seuil:.2f} / {'ON' if self.cache_tta_det else 'OFF'}",
        ]
        if self.vraies_courantes:
            lignes += [
                f"Détectés/Vrai : {len(detections)} / {len(self.vraies_courantes)} (VP:{vp} FP:{fp} FN:{fn})",
                f"Précision     : {prec*100:.1f} %   | Rappel : {rap*100:.1f} %",
                f"Score F1      : {f1*100:.1f} %   | IoU    : {iou_m*100:.1f} %",
            ]
            if mode == "pipeline" and vp > 0:
                lignes.append(f"Classe exacte : {cls_ok}/{vp} ({cls_ok/vp*100:.1f} %)")
        else:
            lignes.append(f"Détections    : {len(detections)} (pas de XML)")

        if detections:
            lignes.append("\nDétail des avions trouvés :")
            for k, d in enumerate(detections, 1):
                if mode == "pipeline":
                    lignes.append(f"  {k}. {d[5]:<4} (det {d[4]*100:4.0f}% | cls {d[6]*100:4.0f}%)")
                else:
                    lignes.append(f"  {k}. Avion (confiance {d[4]*100:4.0f}%)")

        self._ecrire_resultats("\n".join(lignes))
        self.canvas_barres.delete("all")

    def _preparer_image_affichage(self, img_rgb, zoom_max=None):
        self.root.update_idletasks()
        cw = max(500, self.canvas_img.winfo_width() - 16)
        ch = max(450, self.canvas_img.winfo_height() - 16)
        if zoom_max is not None:
            cw = min(cw, zoom_max)
            ch = min(ch, zoom_max)
        h, w, _ = img_rgb.shape
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(img_rgb, (nw, nh), interpolation=interp), scale

    def _afficher_image_canvas(self, img_rgb, zoom_max=None):
        visu, _ = self._preparer_image_affichage(img_rgb, zoom_max=zoom_max)
        self._afficher_image_prete(visu)

    def _afficher_image_prete(self, img_rgb):
        pil_img = Image.fromarray(img_rgb)
        self.photo_tk = ImageTk.PhotoImage(pil_img)
        self.canvas_img.delete("all")
        cw = max(1, self.canvas_img.winfo_width())
        ch = max(1, self.canvas_img.winfo_height())
        self.canvas_img.create_image(cw // 2, ch // 2, image=self.photo_tk, anchor="center")

    def _ecrire_resultats(self, texte):
        self.txt_resultats.config(state="normal")
        self.txt_resultats.delete("1.0", "end")
        self.txt_resultats.insert("1.0", texte)
        self.txt_resultats.config(state="disabled")

    def _dessiner_barres_top5(self, top5_data):
        self.canvas_barres.delete("all")
        w = max(320, self.canvas_barres.winfo_width())
        y = 10
        for nom, prob in top5_data:
            coul = "#a6e3a1" if nom == self.vraie_classe_courante else "#89b4fa"
            self.canvas_barres.create_text(8, y + 7, text=nom, fill="#cdd6f4", anchor="w", font=("Consolas", 9, "bold"))
            bar_x0 = 52
            bar_max = w - 115
            self.canvas_barres.create_rectangle(bar_x0, y + 1, bar_x0 + bar_max, y + 13, fill="#313244", outline="")
            self.canvas_barres.create_rectangle(bar_x0, y + 1, bar_x0 + int(bar_max * prob), y + 13, fill=coul, outline="")
            self.canvas_barres.create_text(w - 8, y + 7, text=f"{prob*100:4.1f}%", fill="#cdd6f4", anchor="e", font=("Consolas", 9))
            y += 21


if __name__ == "__main__":
    fenetre = tk.Tk()
    app = InterfaceTestAvions(fenetre)
    fenetre.mainloop()