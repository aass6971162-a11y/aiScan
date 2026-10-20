"""Fixed UVDoc 20-image corner regression. No training; development and held-out sets are explicit."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(os.environ.get("SCAN_BENCH_WORK", Path(__file__).resolve().parent))
DATA = Path(os.environ.get("SCAN_BENCH_DATA", ROOT / "extreme20/UVDoc_benchmark"))
OUT = Path(os.environ.get("SCAN_BENCH_OUTPUT", ROOT.parent / "outputs/极端场景验证"))
MODEL = Path(os.environ.get("SCAN_CORNER_MODEL",ROOT.parent / "outputs/public-models/lcnet100_h_e_bifpn_256_fp32.onnx"))
IDS = [f"{i:05d}" for i in range(0, 40, 2)]

def round_positive(value):
    """Match Kotlin roundToInt for positive pixel dimensions, not Python ties-to-even."""
    return math.floor(float(value) + .5)

def valid(p):
    if p is None or p.shape != (4, 2) or not np.isfinite(p).all() or (p < -.04).any() or (p > 1.04).any():
        return False
    edges = np.roll(p, -1, axis=0) - p
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    return bool((turns > .000001).all() and .003 <= area(p) <= 1.08 and (np.linalg.norm(edges, axis=1) > .015).all())

def area(p):
    return abs(cv2.contourArea(np.float32(p)))

def touches(p, margin=.025):
    return bool((p < margin).any() or (p > 1-margin).any())

def agrees(a, b):
    return bool((np.linalg.norm(a-b, axis=1) <= max(.015, np.linalg.norm(np.ptp(a, axis=0))*.18)).all())

class Detector:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])

    def candidate(self, image, left=0, top=0, width=1, height=1):
        x = cv2.resize(image, (256, 256)).astype(np.float32).transpose(2, 0, 1)[None] / 255
        maps = self.session.run(None, {"img": x})[0][0]
        if maps.max(axis=(1, 2)).min() < .5:
            return None
        points = []
        for h in maps:
            n, _, stats, centers = cv2.connectedComponentsWithStats((h >= .3).astype(np.uint8))
            if n < 2:
                return None
            label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            points.append((centers[label] + .5) / [h.shape[1], h.shape[0]])
        p = np.array(points) * [width, height] + [left, top]
        if not valid(p) or not valid(np.clip(p, 0, 1)):
            return None
        return dict(points=np.clip(p, 0, 1), score=float(maps.max(axis=(1,2)).min()), clipped=touches(p), review=False)

    def padded(self, image, fraction=.15, legacy=False):
        h, w = image.shape[:2]
        scale = min(1, 1024/max(h,w))
        size = (max(1,int(w*scale)),max(1,int(h*scale))) if legacy else (max(1,round_positive(w*scale)),max(1,round_positive(h*scale)))
        image = cv2.resize(image,size)
        h,w = image.shape[:2]
        # Android multiplies Int dimensions by a Float scale before roundToInt.
        # E.g. 750 * .15f == 112.5f must produce 113, not Python round's 112.
        px,py = (max(1,round_positive(np.float32(d)*np.float32(fraction))) for d in (w,h))
        samples = []
        for i in range(32):
            x,y = i*(w-1)//31,i*(h-1)//31
            samples.extend([image[0,x],image[-1,x],image[y,0],image[y,-1]])
        color = np.sort(np.array(samples), axis=0)[len(samples)//2]
        padded = cv2.copyMakeBorder(image,py,py,px,px,cv2.BORDER_CONSTANT,value=tuple(float(x) for x in color))
        return self.candidate(padded,-px/w,-py/h,(w+2*px)/w,(h+2*py)/h)

    def cropped(self, image, points):
        h,w = image.shape[:2]
        low,high = points.min(axis=0)*[w,h],points.max(axis=0)*[w,h]
        if min(high-low) < 12:
            return None
        margin = np.maximum(20,(high-low)*.25)
        left,top = np.maximum(np.floor(low-margin),0).astype(int)
        right,bottom = np.minimum(np.ceil(high+margin),[w,h]).astype(int)
        c = self.candidate(image[top:bottom,left:right],left/w,top/h,(right-left)/w,(bottom-top)/h)
        if c is not None:
            p = c["points"]*[w,h]
            if ((left>0 and (p[:,0]<left+2).any()) or (right<w and (p[:,0]>right-2).any()) or
                (top>0 and (p[:,1]<top+2).any()) or (bottom<h and (p[:,1]>bottom-2).any())):
                return None
        return c

    def baseline(self, image, preview=False):
        fallback = opencv(image)
        if preview:
            if fallback is not None and fallback["score"] >= .55 and not fallback["clipped"]:
                return fallback
            return self.padded(image,legacy=True) or fallback
        c = self.candidate(image)
        if c is None or c["score"] < .7 or touches(c["points"], .04) or area(c["points"]) > .7:
            pad = self.padded(image,legacy=True)
            if pad is not None and (c is None or (agrees(c["points"],pad["points"]) and pad["score"]>=c["score"]-.05)):
                pad["review"] = c is None
                c = pad
        seed = c if c is not None else fallback
        if seed is not None and (c is None or area(seed["points"]) < .15 or c["score"] < .75):
            crop = self.cropped(image,seed["points"])
            if crop is not None and agrees(seed["points"],crop["points"]) and (c is None or crop["score"]>=c["score"]-.1):
                crop["review"] = c is None or (c["review"] and crop["score"]<.7)
                c = crop
        return c

    def improved(self, image):
        candidates=[]
        raw=self.candidate(image)
        if raw is not None:
            candidates.append(raw)
            if area(raw["points"])<.15:
                crop=self.cropped(image,raw["points"])
                if crop is not None and agrees(raw["points"],crop["points"]):
                    return crop
        for fraction in [.15,.25,.4,.6,.75]:
            c=self.padded(image,fraction)
            if c is not None:
                candidates.append(c)
        if not candidates:
            return None
        def close(a,b):
            tolerance=max(.012,np.linalg.norm(np.ptp(a["points"],axis=0))*.035)
            return np.max(np.linalg.norm(a["points"]-b["points"],axis=1))<=tolerance
        supported=[]
        for c in candidates:
            group=[other for other in candidates if close(c,other)]
            if len(group)>=2:
                supported.append((c,group))
        if supported:
            maximum=max(area(item[0]["points"]) for item in supported)
            c,group=next(item for item in supported if area(item[0]["points"])>=maximum*.8)
            points=c["points"]
            result=dict(points=points,score=min(x["score"] for x in group),review=raw is None,clipped=touches(points))
        else:
            c=max(candidates,key=lambda x:x["score"])
            result=dict(c,review=True)
        if area(result["points"])<.15:
            crop=self.cropped(image,result["points"])
            if crop is not None and agrees(result["points"],crop["points"]):
                result=dict(crop,review=result["review"])
        return result

def opencv(image):
    h,w = image.shape[:2]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if max(h,w)>640:
        gray = cv2.resize(gray,None,fx=640/max(h,w),fy=640/max(h,w),interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),55,155)
    edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    best,score = None,0
    for contour in contours:
        a = abs(cv2.contourArea(contour))
        if not .012*gray.size<=a<=.995*gray.size:
            continue
        approx = cv2.approxPolyDP(contour,cv2.arcLength(contour,True)*.025,True)
        if len(approx)!=4 or not cv2.isContourConvex(approx):
            continue
        p = approx[:,0,:]
        value = a/gray.size*min(1,a/max(1,np.prod(np.ptp(p,axis=0))))
        if value>score:
            score = value
            best = np.array([p[np.argmin(p.sum(1))],p[np.argmax(p[:,0]-p[:,1])],
                             p[np.argmax(p.sum(1))],p[np.argmin(p[:,0]-p[:,1])]],dtype=float)/[gray.shape[1],gray.shape[0]]
    return None if best is None else dict(points=best,score=min(1,score/.72),clipped=touches(best),review=False)

def truth(sample):
    import h5py
    metadata = json.loads((DATA/f"metadata_sample/{sample}.json").read_text())
    geom = metadata["geom_name"]
    with h5py.File(DATA/f"grid2d/{geom}.mat") as f:
        grid = np.array(f["grid2d"]).transpose(2,1,0)
    image = cv2.imread(str(DATA/f"img/{sample}.png"))
    corners = np.array([grid[0,0],grid[0,-1],grid[-1,-1],grid[-1,0]])
    # MATLAB grid is expressed in image pixel coordinates (confirmed against the source visualizer).
    corners /= [image.shape[1],image.shape[0]]
    geom_meta = json.loads((DATA/f"metadata_geom/{geom}.json").read_text())
    return image,corners,{**metadata,"geometry":geom_meta}

def metric(candidate, expected, shape):
    if candidate is None:
        return dict(found=False,corner_error_pct=None,quad_iou=0,review=True)
    points = candidate["points"]
    h,w = shape[:2]
    error = float(np.mean(np.linalg.norm((points-expected)*[w,h],axis=1))/np.hypot(w,h)*100)
    overlap = cv2.intersectConvexConvex(np.float32(points),np.float32(expected))[0]
    iou = max(0,float(overlap/(area(points)+area(expected)-overlap)))
    return dict(found=True,corner_error_pct=error,quad_iou=iou,
                review=bool(candidate["review"] or candidate["clipped"] or candidate["score"]<.55),
                confidence=min(.54,candidate["score"]) if candidate["review"] else candidate["score"],
                corners=points.tolist())

def sheet(records, path, fields=()):
    canvas = np.full((4*350,5*250,3),245,np.uint8)
    for index,record in enumerate(records):
        image,gt,_ = truth(record["id"])
        h,w=image.shape[:2]
        scale=min(238/w,292/h)
        thumb=cv2.resize(image,(round(w*scale),round(h*scale)))
        # Yellow = official corners; red = old; green = new.
        cv2.polylines(thumb,[np.int32(gt*[thumb.shape[1],thumb.shape[0]])],True,(0,190,240),2,cv2.LINE_AA)
        for field,color in zip(fields,[(30,30,230),(40,200,0)]):
            result=record.get(field,{})
            if result.get("found"):
                cv2.polylines(thumb,[np.int32(np.array(result["corners"])*[thumb.shape[1],thumb.shape[0]])],True,color,2,cv2.LINE_AA)
        x=(index%5)*250+(250-thumb.shape[1])//2;y=(index//5)*350+24
        canvas[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb
        cv2.putText(canvas,record["id"]+" "+record["split"],((index%5)*250+8,(index//5)*350+18),cv2.FONT_HERSHEY_SIMPLEX,.46,(35,35,35),1,cv2.LINE_AA)
        for j,field in enumerate(fields):
            r=record.get(field,{})
            label="miss" if not r.get("found") else "E {:.2f}% IoU {:.2f}".format(r["corner_error_pct"],r["quad_iou"])
            cv2.putText(canvas,label,((index%5)*250+8,(index//5)*350+325+j*17),cv2.FONT_HERSHEY_SIMPLEX,.4,(35,35,35),1,cv2.LINE_AA)
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),canvas)

def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["contact","baseline","explore","develop","final"],default="contact")
    args=parser.parse_args()
    records=[]
    detector=Detector() if args.mode!="contact" else None
    for i,sample in enumerate(IDS):
        if args.mode=="develop" and i>=10:
            continue
        image,expected,metadata=truth(sample)
        record=dict(id=sample,split="dev" if i<10 else "holdout",sha256=hashlib.sha256((DATA/f"img/{sample}.png").read_bytes()).hexdigest(),
                    source="https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip",entry=f"UVDoc_benchmark/img/{sample}.png",
                    width=image.shape[1],height=image.shape[0],expected_corners=expected.tolist(),metadata=metadata)
        if args.mode=="baseline":
            for preview in [False,True]:
                name="old_preview" if preview else "old_still"
                record[name]=metric(detector.baseline(image,preview),expected,image.shape)
            print(sample,record["old_still"]["corner_error_pct"],record["old_preview"]["corner_error_pct"],flush=True)
        elif args.mode in ("develop","final"):
            record["new"]=metric(detector.improved(image),expected,image.shape)
            print(sample, record["new"]["corner_error_pct"], record["new"]["quad_iou"],record["new"]["review"],flush=True)
        elif args.mode=="explore" and i<10:
            options = {"raw":detector.candidate(image)}
            for fraction in [.08,.15,.25,.4,.6]:
                options[str(fraction)] = detector.padded(image,fraction)
            record["candidates"] = {k:metric(v,expected,image.shape) for k,v in options.items()}
            print(sample, {k:(None if v is None else (round(metric(v,expected,image.shape)["corner_error_pct"],2),round(v["score"],2),round(area(v["points"]),2))) for k,v in options.items()},flush=True)
        records.append(record)
    save_json(OUT/("sources.json" if args.mode=="contact" else args.mode+".json"),records)
    sheet(records,OUT/("20张样本与官方角点.jpg" if args.mode=="contact" else args.mode+"检测.jpg"),() if args.mode=="contact" else (("old_still",) if args.mode=="baseline" else ("new",)))
