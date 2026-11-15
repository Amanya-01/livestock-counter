# pipeline.py : inference code for the livestock counter (tiled + whole-image detection)
import numpy as np

CLASS_NAMES = ["Cattle", "Goat", "Sheep"]


def nms_np(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def classwise_nms(boxes, scores, classes, iou_thr):
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        k = nms_np(boxes[idx], scores[idx], iou_thr)
        keep.extend(idx[k].tolist())
    return np.array(sorted(keep), dtype=int)


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _to_pred(r):
    return dict(conf=r.boxes.conf.cpu().numpy(),
                cls=r.boxes.cls.cpu().numpy().astype(int),
                xyxyn=r.boxes.xyxyn.cpu().numpy())


def predict_full_batch(model, paths, imgsz, conf=0.005, iou=0.7, device=None, chunk=32):
    out = {}
    for i in range(0, len(paths), chunk):
        part = [str(p) for p in paths[i:i + chunk]]
        res = model.predict(part, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False, max_det=1000)
        for p, r in zip(part, res):
            out[p.replace("\\", "/").split("/")[-1]] = _to_pred(r)
    return out


def predict_tiled(model, img_bgr, imgsz, tile=512, overlap=0.25, conf=0.005, iou=0.7, merge_iou=0.55,
                  device=None, edge_drop=True, edge_margin=3, include_full=True, chunk=16, tile_max_area=None,
                  gate_k=None, gate_conf=0.3):
    H, W = img_bgr.shape[:2]
    boxes, scores, classes = [], [], []
    skip_tiles = False
    if include_full:
        r = model.predict(img_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False, max_det=1000)[0]
        full_conf = r.boxes.conf.cpu().numpy()
        boxes.append(r.boxes.xyxy.cpu().numpy()); scores.append(full_conf)
        classes.append(r.boxes.cls.cpu().numpy().astype(int))
        skip_tiles = gate_k is not None and int((full_conf >= gate_conf).sum()) < gate_k
    crops, origins = [], []
    if not skip_tiles:
        for y0 in tile_starts(H, tile, overlap):
            for x0 in tile_starts(W, tile, overlap):
                crops.append(img_bgr[y0:y0 + tile, x0:x0 + tile]); origins.append((x0, y0))
    for i in range(0, len(crops), chunk):
        res = model.predict(crops[i:i + chunk], imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False, max_det=1000)
        for r, (x0, y0), crop in zip(res, origins[i:i + chunk], crops[i:i + chunk]):
            b = r.boxes.xyxy.cpu().numpy().copy()
            if len(b) == 0:
                continue
            th, tw = crop.shape[:2]
            b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
            ok = np.ones(len(b), dtype=bool)
            if edge_drop:
                if x0 > 0:           ok &= b[:, 0] > x0 + edge_margin
                if x0 + tw < W:      ok &= b[:, 2] < x0 + tw - edge_margin
                if y0 > 0:           ok &= b[:, 1] > y0 + edge_margin
                if y0 + th < H:      ok &= b[:, 3] < y0 + th - edge_margin
            if tile_max_area is not None:
                ok &= ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) <= tile_max_area * H * W
            boxes.append(b[ok]); scores.append(r.boxes.conf.cpu().numpy()[ok])
            classes.append(r.boxes.cls.cpu().numpy().astype(int)[ok])
    if not boxes or sum(len(b) for b in boxes) == 0:
        return dict(conf=np.zeros(0), cls=np.zeros(0, dtype=int), xyxyn=np.zeros((0, 4)))
    boxes = np.concatenate(boxes); scores = np.concatenate(scores); classes = np.concatenate(classes)
    keep = classwise_nms(boxes, scores, classes, merge_iou)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    return dict(conf=scores, cls=classes, xyxyn=boxes / np.array([W, H, W, H], dtype=float))


def run_pipeline(model, img_bgr, cfg):
    # cfg keys: imgsz, conf, iou, tile (None for whole image only), overlap, merge_iou, tile_max_area, gate_k, gate_conf, device
    if cfg.get("tile"):
        p = predict_tiled(model, img_bgr, cfg["imgsz"], tile=cfg["tile"], overlap=cfg.get("overlap", 0.25),
                          conf=min(cfg["conf"], 0.05), iou=cfg.get("iou", 0.7), merge_iou=cfg.get("merge_iou", 0.55),
                          device=cfg.get("device"), tile_max_area=cfg.get("tile_max_area"),
                          gate_k=cfg.get("gate_k"), gate_conf=cfg.get("gate_conf", 0.3))
    else:
        r = model.predict(img_bgr, imgsz=cfg["imgsz"], conf=min(cfg["conf"], 0.05), iou=cfg.get("iou", 0.7),
                          device=cfg.get("device"), verbose=False, max_det=1000)[0]
        p = _to_pred(r)
    keep = p["conf"] >= cfg["conf"]
    p = dict(conf=p["conf"][keep], cls=p["cls"][keep], xyxyn=p["xyxyn"][keep])
    counts = {n: int((p["cls"] == i).sum()) for i, n in enumerate(CLASS_NAMES)}
    return p, counts
