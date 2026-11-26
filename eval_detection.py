import os
import json
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

DOSSIER_IMAGES = "images"
DOSSIER_XML = "annotations"

modele_det = tf.keras.models.load_model("modele_detection.keras", compile=False)
TAILLE_DET = int(modele_det.input_shape[1])
GRILLE = int(modele_det.output_shape[1])

seuil_conf = 0.50
if os.path.exists("config_detection.json"):
    with open("config_detection.json", "r", encoding="utf-8") as f:
        seuil_conf = float(json.load(f).get("seuil_optimal", 0.50))

modele_clf = None
categories = []
TAILLE_CLF = 128
if os.path.exists("modele_classification.keras") and os.path.exists("classes.txt"):
    modele_clf = tf.keras.models.load_model("modele_classification.keras", compile=False)
    TAILLE_CLF = int(modele_clf.input_shape[1])
    with open("classes.txt", "r", encoding="utf-8") as f:
        categories = [l.strip() for l in f if l.strip()]


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


def detecter_sur_image(image_rgb, seuil=0.50, seuil_nms=0.35):
    h_img, w_img, _ = image_rgb.shape
    base = cv2.resize(image_rgb, (TAILLE_DET, TAILLE_DET), interpolation=cv2.INTER_AREA)

    batch_rot = np.array([
        base,
        cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(base, cv2.ROTATE_180),
        cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ], dtype="uint8")

    preds = modele_det.predict(batch_rot, verbose=0)

    seuil_cand = max(0.25, seuil * 0.60)
    toutes_cands = []
    for k in range(4):
        toutes_cands.extend(decoder_une_vue(preds[k], k, seuil_cand=seuil_cand, seuil_nms=seuil_nms))

    toutes_cands.sort(key=lambda d: d[4], reverse=True)
    clusters = []
    for cand in toutes_cands:
        meilleur_grp = None
        meilleur_iou = 0.0
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
        score_moyen = float(np.sum(poids) / 4.0)
        if score_moyen < seuil_final:
            continue
        w_norm = poids / np.sum(poids)
        fx1 = float(np.sum([d[0] * w_norm[i] for i, d in enumerate(grp)])) * w_img
        fy1 = float(np.sum([d[1] * w_norm[i] for i, d in enumerate(grp)])) * h_img
        fx2 = float(np.sum([d[2] * w_norm[i] for i, d in enumerate(grp)])) * w_img
        fy2 = float(np.sum([d[3] * w_norm[i] for i, d in enumerate(grp)])) * h_img
        if (fx2 - fx1) >= 12.0 and (fy2 - fy1) >= 12.0:
            boites_finales.append((fx1, fy1, fx2, fy2, score_moyen))

    boites_finales.sort(key=lambda d: d[4], reverse=True)
    boites_propres = []
    for b in boites_finales:
        if all(calculer_iou(b[:4], p[:4]) < seuil_nms for p in boites_propres):
            boites_propres.append(b)

    resultats = []
    if len(boites_propres) > 0 and modele_clf is not None:
        vignettes = np.array([
            extraire_vignette_carree(image_rgb, x1, y1, x2, y2, TAILLE_CLF, 0.15)
            for (x1, y1, x2, y2, _) in boites_propres
        ], dtype="uint8")

        vignettes_flip = vignettes[:, :, ::-1, :]
        vues_cls = [
            vignettes,
            np.rot90(vignettes, k=1, axes=(1, 2)),
            np.rot90(vignettes, k=2, axes=(1, 2)),
            np.rot90(vignettes, k=3, axes=(1, 2)),
            vignettes_flip,
            np.rot90(vignettes_flip, k=1, axes=(1, 2)),
            np.rot90(vignettes_flip, k=2, axes=(1, 2)),
            np.rot90(vignettes_flip, k=3, axes=(1, 2)),
        ]
        preds_cls = sum(modele_clf.predict(v, verbose=0) for v in vues_cls) / 8.0

        for i, (x1, y1, x2, y2, score) in enumerate(boites_propres):
            idx_c = int(np.argmax(preds_cls[i]))
            conf_c = float(preds_cls[i][idx_c])
            resultats.append((x1, y1, x2, y2, score, categories[idx_c], conf_c))
    else:
        for (x1, y1, x2, y2, score) in boites_propres:
            resultats.append((x1, y1, x2, y2, score, "Avion", score))

    return resultats


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


def evaluer_metriques(detections, vraies, seuil_iou=0.40):
    vp, cls_ok = 0, 0
    trouves = set()
    ious = []
    for det in detections:
        best_iou, best_idx = 0.0, -1
        for idx_v, vbox in enumerate(vraies):
            if idx_v in trouves:
                continue
            iou = calculer_iou(det[:4], vbox[:4])
            if iou > best_iou:
                best_iou, best_idx = iou, idx_v
        if best_iou >= seuil_iou and best_idx != -1:
            vp += 1
            trouves.add(best_idx)
            ious.append(best_iou)
            if det[5] == vraies[best_idx][4]:
                cls_ok += 1
    fp = len(detections) - vp
    fn = len(vraies) - vp
    return vp, fp, fn, cls_ok, ious


print("1 : Tester et afficher une image (nom precis ou hasard en validation)")
print("2 : Evaluer tout le jeu de validation (scores globaux + JSON)")
mode = input("Choix (1 ou 2) : ").strip()

if os.path.exists("split_images.json"):
    with open("split_images.json", "r", encoding="utf-8") as f:
        images_val = [x.replace(".xml", ".jpg") for x in json.load(f)["val"]]
else:
    images_val = sorted([f for f in os.listdir(DOSSIER_IMAGES) if f.endswith(".jpg")])

if mode == "2":
    vp_t, fp_t, fn_t, cls_t = 0, 0, 0, 0
    tous_ious = []
    for nom_img in images_val:
        img_bgr = cv2.imread(os.path.join(DOSSIER_IMAGES, nom_img))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dets = detecter_sur_image(img_rgb, seuil=seuil_conf)
        vraies = lire_verite_terrain(nom_img)
        vp, fp, fn, cok, ious = evaluer_metriques(dets, vraies)
        vp_t += vp
        fp_t += fp
        fn_t += fn
        cls_t += cok
        tous_ious.extend(ious)

    prec = vp_t / (vp_t + fp_t) if (vp_t + fp_t) > 0 else 0.0
    rap = vp_t / (vp_t + fn_t) if (vp_t + fn_t) > 0 else 0.0
    f1 = (2 * prec * rap / (prec + rap)) if (prec + rap) > 0 else 0.0
    acc_cls = cls_t / vp_t if vp_t > 0 else 0.0
    iou_m = float(np.mean(tous_ious)) if tous_ious else 0.0

    bilan = {
        "nb_images_val": len(images_val),
        "resolution_detection": TAILLE_DET,
        "seuil_utilise": seuil_conf,
        "precision": round(prec, 4),
        "rappel": round(rap, 4),
        "score_f1": round(f1, 4),
        "iou_moyen": round(iou_m, 4),
        "precision_classification_sur_detections": round(acc_cls, 4)
    }
    with open("evaluation_detection_globale.json", "w", encoding="utf-8") as f:
        json.dump(bilan, f, indent=4)

    print("-" * 50)
    print(f"Precision globale : {prec * 100:.2f} %")
    print(f"Rappel global     : {rap * 100:.2f} %")
    print(f"Score F1 global   : {f1 * 100:.2f} %")
    print(f"IoU moyen         : {iou_m * 100:.2f} %")
    print(f"Classe exacte     : {acc_cls * 100:.2f} % (sur avions detectes)")
    print("-" * 50)
else:
    toutes_images = sorted([f for f in os.listdir(DOSSIER_IMAGES) if f.endswith(".jpg")])
    choix = input("Nom de l'image (ex: 3045 ou 3045.jpg) ou Entree pour hasard : ").strip()
    if choix and not choix.endswith(".jpg"):
        choix = choix + ".jpg"
    nom_image = choix if choix in toutes_images else random.choice(images_val)

    img_bgr = cv2.imread(os.path.join(DOSSIER_IMAGES, nom_image))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detecter_sur_image(img_rgb, seuil=seuil_conf)
    vraies = lire_verite_terrain(nom_image)
    vp, fp, fn, cok, ious = evaluer_metriques(detections, vraies)

    prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rap = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * prec * rap / (prec + rap)) if (prec + rap) > 0 else 0.0

    resultat_json = {
        "image": nom_image,
        "seuil_confiance": seuil_conf,
        "nombre_detections": len(detections),
        "metriques": {
            "precision": round(prec, 4),
            "rappel": round(rap, 4),
            "score_f1": round(f1, 4),
            "iou_moyen": round(float(np.mean(ious)) if ious else 0.0, 4)
        },
        "detections": [
            {
                "classe": d[5],
                "confiance_detection": round(d[4], 4),
                "confiance_classe": round(d[6], 4),
                "boite": {
                    "xmin": round(d[0], 1),
                    "ymin": round(d[1], 1),
                    "xmax": round(d[2], 1),
                    "ymax": round(d[3], 1)
                }
            }
            for d in detections
        ]
    }

    nom_json = f"detection_{nom_image.replace('.jpg', '.json')}"
    with open(nom_json, "w", encoding="utf-8") as f:
        json.dump(resultat_json, f, indent=4)

    print(f"JSON sauvegarde : {nom_json}")
    print(f"Precision : {prec * 100:.1f} % | Rappel : {rap * 100:.1f} % | Score F1 : {f1 * 100:.1f} %")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)

    for i, (vx1, vy1, vx2, vy2, vnom) in enumerate(vraies):
        ax.add_patch(patches.Rectangle(
            (vx1, vy1), vx2 - vx1, vy2 - vy1,
            linewidth=2, edgecolor="lime", linestyle="--", facecolor="none",
            label="Vraie boite (XML)" if i == 0 else None
        ))

    for i, (dx1, dy1, dx2, dy2, sc_det, nom_cls, sc_cls) in enumerate(detections):
        ax.add_patch(patches.Rectangle(
            (dx1, dy1), dx2 - dx1, dy2 - dy1,
            linewidth=2, edgecolor="red", facecolor="none",
            label="Detection + Classification IA" if i == 0 else None
        ))
        ax.text(
            dx1, max(12, dy1 - 5),
            f"{nom_cls} ({sc_det * 100:.0f}%)",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="red", alpha=0.8, edgecolor="none", pad=1.5)
        )

    plt.title(f"{nom_image} - F1: {f1 * 100:.1f}% ({len(detections)}/{len(vraies)} avions)", fontsize=13, fontweight="bold")
    if vraies or detections:
        plt.legend()
    plt.axis("off")
    plt.show()