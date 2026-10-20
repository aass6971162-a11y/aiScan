"""Adaptive context search. No sample IDs or ground truth enter selection."""
import numpy as np
from benchmark_extreme import Detector, area, touches, metric, truth, IDS, OUT, save_json

def close(a,b):
    tolerance=max(.012,np.linalg.norm(np.ptp(a['points'],axis=0))*.035)
    distances=np.linalg.norm(a['points']-b['points'],axis=1)
    return bool(distances.max()<=tolerance)

class AdaptiveDetector(Detector):
    calls=0
    def improved(self,image):
        self.calls=0
        raw=self.candidate(image)
        candidates=[] if raw is None else [raw]
        if raw is not None and not touches(raw['points'],.04):
            crop=self.cropped(image,raw['points'])
            if crop is not None and close(raw,crop):
                self.reason='roi'
                return dict(crop,score=min(raw['score'],crop['score']))
        # Preserve fine-scale context before the large padded canvas loses page detail.
        fractions=[.08,.15,.25,.4,.6,.75]
        if raw is not None and not touches(raw['points'],.04) and .15<=area(raw['points'])<.5:
            fractions.insert(1,.12)
        for fraction in fractions:
            c=self.padded(image,fraction)
            if c is not None: candidates.append(c)
            if fraction==.15 and c is not None and area(c['points'])>.70:
                extra=self.padded(image,.1)
                if extra is not None: candidates.append(extra)
            supported=[(seed,[other for other in candidates if close(seed,other)]) for seed in candidates]
            supported=[(s,g) for s,g in supported if len(g)>=2]
            # Do not let a later, larger hypothesis displace the first interior page.
            first=next((seed for seed in candidates if not touches(seed['points'],.04) and area(seed['points'])>=.15),None)
            match=next(((s,g) for s,g in supported if s is first),None)
            if match is not None:
                s,g=match; s=max(g,key=lambda x:x['score']); self.reason='interior'
                return dict(s,score=min(v['score'] for v in g),review=raw is None)
            if fraction==.12:
                for seed in candidates:
                    tolerance=max(.012,np.linalg.norm(np.ptp(seed['points'],axis=0))*.035)
                    votes=sum(np.linalg.norm(other['points']-seed['points'],axis=1)<=tolerance for other in candidates)
                    if np.min(votes)>=2:
                        self.reason='corner-support'
                        return dict(seed,review=True)
            complete=next(((s,g) for s,g in supported if area(s['points'])>.70),None)
            if complete is not None:
                maximum=max(area(s['points']) for s,g in supported)
                s,g=next((s,g) for s,g in supported if area(s['points'])>=maximum*.8)
                self.reason='near'
                return dict(s,score=min(v['score'] for v in g),review=raw is None)
        if not candidates: self.reason='none'; return None
        supported=[(seed,[other for other in candidates if close(seed,other)]) for seed in candidates]
        supported=[(s,g) for s,g in supported if len(g)>=2]
        first=next((seed for seed in candidates if not touches(seed['points'],.04) and area(seed['points'])>=.15),None)
        if first is not None:
            self.reason='ambiguous-interior'
            return dict(first,review=True)
        if supported:
            # Only unresolved frame-clipped searches use completeness; interior searches above
            # already preserve their first verified page regardless of a larger background.
            maximum=max(area(s['points']) for s,g in supported)
            s,g=next((s,g) for s,g in supported if area(s['points'])>=maximum*.8)
            self.reason='full'
            return dict(s,score=min(v['score'] for v in g),review=raw is None)
        self.reason='unconfirmed'
        return dict(candidates[0],review=True)

    def candidate(self,*args,**kwargs):
        self.calls+=1
        return super().candidate(*args,**kwargs)

if __name__=='__main__':
    import sys
    if '--public' in sys.argv:
        variants=lambda: []
    else:
        from inspect_layers import variants
    d=AdaptiveDetector(); records=[]
    for name,image,gt in list(variants())+[(sample,*truth(sample)[:2]) for sample in IDS]:
        before=Detector.improved(d,image)
        after=d.improved(image)
        row=dict(id=name,old=metric(before,gt,image.shape),new=metric(after,gt,image.shape),calls=d.calls,reason=d.reason)
        records.append(row)
        print(name,round(row['old']['corner_error_pct'] or 0,3),round(row['new']['corner_error_pct'] or 0,3),d.calls,d.reason,flush=True)
    save_json(OUT/'v5-experiment.json',records)
