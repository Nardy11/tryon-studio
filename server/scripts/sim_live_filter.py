# Offline check of the live filter (same maths as web/src/components/LiveCamera.tsx): lays outfit
# photos onto a still photo, full body and upper body only, and saves a comparison sheet.
#   cd server; .venv\Scripts\python scripts\sim_live_filter.py <folder with 25.jpg and 26.jpg>/
import os
sys_path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
_sys.path.insert(0, sys_path_root)
import base64, io, sys
import numpy as np, cv2
from PIL import Image
from app.live import outfit_layer, _pose

SEEN = 0.55
LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
P = lambda x, y, v=1: np.array([x, y, v], float)
def sub(a, b): return P(a[0]-b[0], a[1]-b[1])
def add(a, b): return P(a[0]+b[0], a[1]+b[1])
def mul(a, k): return P(a[0]*k, a[1]*k)
def mid(a, b): return P((a[0]+b[0])/2, (a[1]+b[1])/2, min(a[2], b[2]))
def ln(a): return float(np.hypot(a[0], a[1])) or 1e-6
def unit(a): return mul(a, 1/ln(a))
def dist(a, b): return ln(sub(a, b))
def affine3(s, d):
    src = np.float32([[p[0], p[1]] for p in s]); dst = np.float32([[p[0], p[1]] for p in d])
    return cv2.getAffineTransform(src, dst)
def similar(s0, s1, d0, d1):
    vs, vd = sub(s1, s0), sub(d1, d0); k = ln(vd)/ln(vs)
    ang = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0]); a, b = k*np.cos(ang), k*np.sin(ang)
    return np.float32([[a, -b, d0[0]-(a*s0[0]-b*s0[1])], [b, a, d0[1]-(b*s0[0]+a*s0[1])]])
def capsule(a, b, r, extA=0, extB=0):
    u = unit(sub(b, a)); n = P(-u[1], u[0]); A = sub(a, mul(u, extA)); B = add(b, mul(u, extB)); pts = []
    for i in range(9):
        t = -np.pi/2 + np.pi*i/8; pts.append(add(B, add(mul(u, r*np.cos(t)), mul(n, r*np.sin(t)))))
    for i in range(9):
        t = np.pi/2 + np.pi*i/8; pts.append(add(A, add(mul(u, r*np.cos(t)), mul(n, r*np.sin(t)))))
    return pts

def bones(r):
    has = lambda i: r[i][2] > 0.3
    if not (has(LS) and has(RS)): return []
    sw = dist(r[LS], r[RS]); hk = has(LH) and has(RH); mS = mid(r[LS], r[RS])
    mH = mid(r[LH], r[RH]) if hk else add(mS, P(0, sw*1.4)); T = dist(mS, mH)
    hw = max(dist(r[LH], r[RH]), sw*.55) if hk else sw*.8
    W = max(sw, T/1.3)  # frontal shoulder width, also for a model turned sideways
    out = [('torso', mS, mH, W*.55)]
    if hk and has(LK) and has(RK): out.append(('pelvis', mH, mid(r[LK], r[RK]), hw*.55))
    for n, a, b, rad in (('thighL', LH, LK, hw*.3), ('thighR', RH, RK, hw*.3), ('shinL', LK, LA, hw*.22), ('shinR', RK, RA, hw*.22),
                         ('armL', LS, LE, W*.17), ('armR', RS, RE, W*.17), ('foreL', LE, LW, W*.14), ('foreR', RE, RW, W*.14)):
        if has(a) and has(b): out.append((n, r[a], r[b], rad))
    return out

def assign(bs, gw, gh, alpha):
    ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
    best = np.full((gh, gw), np.inf, np.float32); owner = np.full((gh, gw), -1, np.int16)
    for k, (n, a, b, rad) in enumerate(bs):
        ab = np.array([b[0]-a[0], b[1]-a[1]], np.float32); L2 = float(ab @ ab) or 1e-6
        t = np.clip(((xs-a[0])*ab[0] + (ys-a[1])*ab[1]) / L2, 0, 1)
        d = np.hypot(xs-(a[0]+t*ab[0]), ys-(a[1]+t*ab[1])) / rad
        m = d < best; best[m] = d[m]; owner[m] = k
    owner[alpha == 0] = -1
    return owner

def maps(r, live, bs):
    seen = lambda i: live[i][2] > SEEN
    out = {}
    if not (seen(LS) and seen(RS)): return out
    sw = dist(r[LS], r[RS]); hk = r[LH][2] > .3 and r[RH][2] > .3; mS = mid(r[LS], r[RS])
    mH = mid(r[LH], r[RH]) if hk else add(mS, P(0, sw*1.4)); T = dist(mS, mH); W = max(sw, T/1.3)
    lS = mid(live[LS], live[RS])
    if hk and seen(LH) and seen(RH):
        lH = mid(live[LH], live[RH])
        out['torso'] = affine3([r[LS], r[RS], mH], [live[LS], live[RS], lH]) if sw >= W*.85 else similar(mS, mH, lS, lH)
    else:
        # Shoulders only: scale from the frontal width, so a turned model isn't blown up.
        M = similar(r[LS], r[RS], live[LS], live[RS]); k = sw / W
        M[:, :2] *= k; c = np.array([lS[0], lS[1]]); M[:, 2] = c - M[:, :2] @ np.array([mS[0], mS[1]])
        out['torso'] = M
    names = {'pelvis': None, 'thighL': (LH, LK), 'thighR': (RH, RK), 'shinL': (LK, LA), 'shinR': (RK, RA),
             'armL': (LS, LE), 'armR': (RS, RE), 'foreL': (LE, LW), 'foreR': (RE, RW)}
    for n, a, b, rad in bs:
        if n == 'pelvis':
            if seen(LH) and seen(RH):
                out[n] = affine3([r[LH], r[RH], b], [live[LH], live[RH], mid(live[LK], live[RK])]) if seen(LK) and seen(RK) else out['torso']
        elif n != 'torso':
            i, j = names[n]
            if seen(i) and seen(j): out[n] = similar(r[i], r[j], live[i], live[j])
    return out

ORDER = ['pelvis', 'thighL', 'thighR', 'shinL', 'shinR', 'torso', 'armL', 'armR', 'foreL', 'foreR']

def render(frame: Image.Image, layer: dict) -> Image.Image:
    g = Image.open(io.BytesIO(base64.b64decode(layer['image'].split(',')[1]))).convert('RGBA')
    gw, gh = g.size; garr = np.asarray(g)
    ref = [P(x*gw, y*gh, v) for x, y, v in layer['landmarks']]
    F = np.asarray(frame.convert('RGB')).astype(np.float32); H, W = F.shape[:2]
    lm = _pose(frame)
    if lm is None: return frame
    live = [P(x*W, y*H, v) for x, y, v in lm]
    bs = bones(ref)
    owner = assign(bs, gw, gh, garr[..., 3])
    ms = maps(ref, live, bs)
    for name in ORDER:
        k = next((q for q, b in enumerate(bs) if b[0] == name), None)
        if k is None or name not in ms: continue
        rgba = garr.copy()
        own = cv2.dilate((owner == k).astype(np.uint8), np.ones((5, 5), np.uint8))  # a little overlap: no seams
        rgba[..., 3] = rgba[..., 3] * own
        w = cv2.warpAffine(rgba, ms[name], (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)).astype(np.float32)
        a = w[..., 3:4] / 255.0
        F = F * (1 - a) + w[..., :3] * a
    return Image.fromarray(F.clip(0, 255).astype(np.uint8))

if __name__ == '__main__':
    I = sys.argv[1]
    person = Image.open('../samples/person-elevator.jpg').convert('RGB')
    w, h = person.size
    upper = person.crop((0, 0, w, int(h * 0.55)))  # only the upper body in frame
    tiles = []
    for f in ('25.jpg', '26.jpg'):
        layer = outfit_layer(Image.open(I + f))
        tiles.append(render(person, layer))
        tiles.append(render(upper, layer))
    hh = 520
    tiles = [t.resize((int(t.width * hh / t.height), hh)) for t in tiles]
    s = Image.new('RGB', (sum(t.width for t in tiles), hh), 'white'); x = 0
    for t in tiles: s.paste(t, (x, 0)); x += t.width
    s.save('../samples/out/filter-sim.jpg', quality=88)
    print('ok')
