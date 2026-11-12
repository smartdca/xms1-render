"""xms1 樹層 4K 片段渲染（燈光調整台的 WebGL 算法原封不動搬過來，用 moderngl + EGL 在沒有顯示卡的機器上跑）。

usage:
  render_seg.py abs <work> <logo a> <b> <c> <h> <outdir> <k0> <k1>   正式版：含螢幕，每段一個檔
  render_seg.py prep <badges dir> <work dir>         先做素材：4K 亮/熄燈圖、遮罩、384px 徽章圖集
  render_seg.py seg  <work dir> <out.mp4> [fx] [--from F --to F] [--still t,t,...]
      fx：base（預設）或 rise fall spiral breathe ripple sweep sparkle front out
      --from/--to：只算某段影格（給 Actions 平行切片用），輸出成獨立 mp4，之後 concat
      --still：只輸出指定秒數的 PNG（檢查用），不出影片

數值＝Henry 在燈光調整台定案的版本（見 HANDOFF-xms1.md）。
無縫：片長 SEG 秒內每顆球轉整數圈（一圈秒數只取 SEG/n 落在 20–36 秒的值），閃爍排程頭尾相接；
彩蛋片段跟基本片段用同一組亂數，效果放在正中間，效果以外的影格跟基本片段完全一樣。
"""
import json, math, os, subprocess, sys
import numpy as np, cv2

K = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(K, '..', '..'))
W4, H4 = 3840, 2160
FPS = 30
SEG = float(os.environ.get('SEG', 180))          # 片長（秒）
P = dict(gSpin=1.7, gStatic=1.4, gBg=1.75, gLogo=0.45, bGap=0.01, bFade=1.0, bHold=0.0, bBack=0.5, bDepth=1.0,
         sMin=20, sMax=36, dbl=True, fxLen=1.0)
FX = [('rise', 10), ('fall', 8), ('spiral', 10), ('breathe', 10), ('ripple', 8), ('sweep', 7), ('sparkle', 7),
      ('front', 7), ('out', 10)]
GOP = 2 * FPS                                      # 關鍵格固定每 2 秒，片段可以直接接（不重新編碼）
TREE_X = (430, 1450)                               # 樹的左右範圍（1920 座標）；範圍外的閃爍點、遮罩一律不作用
CELL = 384                                         # 徽章圖集每格（工具是 192，4K 用兩倍）

# ---------------------------------------------------------------- 素材
def prep(bdir, work):
    os.makedirs(work, exist_ok=True)
    sc = f'{ROOT}/videos/xms1/assets/scene'
    U = cv2.imread(f'{sc}/unlit.png'); L = cv2.imread(f'{sc}/lit.png'); H, W = U.shape[:2]
    assert (W, H) == (W4, H4)
    L = cv2.warpPerspective(L, np.array(json.load(open(f'{K}/lit_to_unlit_H.json'))), (W, H),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(f'{work}/lit.png', L); cv2.imwrite(f'{work}/unlit.png', U)
    ticks = list(json.load(open(f'{K}/badges.json')))
    at = np.zeros((8 * CELL, 10 * CELL, 4), np.uint8)
    for k, t in enumerate(ticks):
        b = cv2.imread(f'{bdir}/{t}.png', cv2.IMREAD_UNCHANGED)
        at[(k // 10) * CELL:(k // 10 + 1) * CELL, (k % 10) * CELL:(k % 10 + 1) * CELL] = \
            cv2.resize(b, (CELL, CELL), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{work}/atlas.png', at)
    m = cv2.imread(f'{K}/tool/mask.png')                     # 工具用的遮罩（1920），放大到 4K
    m = cv2.resize(m, (W4, H4), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # 遮罩 R（樹的範圍）只留在樹附近：用留下來的 Logo 球＋閃爍點外擴、模糊，樹外（右邊電腦螢幕等）一律 0
    hull = np.zeros((H4, W4), np.uint8)
    for b in json.load(open(f'{K}/tool/balls.json')): cv2.circle(hull, (int(b['x'] * 2), int(b['y'] * 2)), int(max(b['a'], b['b']) * 2 + 60), 255, -1)
    for q in json.load(open(f'{K}/tool/plain.json')):
        if TREE_X[0] <= q[0] <= TREE_X[1]: cv2.circle(hull, (int(q[0] * 2), int(q[1] * 2)), int(q[2] * 2 + 60), 255, -1)
    hull = cv2.GaussianBlur(hull.astype(np.float32) / 255, (0, 0), 30)
    m[..., 2] *= hull
    cv2.imwrite(f'{work}/mask.png', m.clip(0, 255).astype(np.uint8))
    print('prep ok', work)

# ---------------------------------------------------------------- 著色器（與 kit/xms1/tool/index.html 相同，只換成桌機 GLSL 寫法）
VS = """#version 330
in vec2 c;uniform vec4 uB;uniform vec2 uR;uniform float uFull;out vec2 vP;
void main(){vec2 p=uFull>0.5?(c*0.5+0.5)*uR:uB.xy+c*uB.zw*1.05;vP=p;gl_Position=vec4(p.x/uR.x*2.-1.,1.-p.y/uR.y*2.,0,1);}"""

def fs_from_tool():
    src = open(f'{K}/tool/index.html', encoding='utf8').read()
    s = src[src.index('const FS=`') + 10:]; s = s[:s.index('`;')]
    s = s.replace('precision highp float;', '#version 330\nout vec4 fragColor;\n#define texture2D texture\n#define gl_FragColor fragColor\n')
    s = s.replace('varying vec2 vP;', 'in vec2 vP;')
    return s

# ---------------------------------------------------------------- 動態（與工具相同的規則，改成可重現、頭尾相接）
class Seed:
    def __init__(s, v): s.s = v
    def __call__(s): s.s = (s.s * 16807) % 2147483647; return s.s / 2147483647

def ease(u): u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)

def build_state():
    rnd = Seed(7)
    balls = json.load(open(f'{K}/tool/balls.json'))
    for j, p in enumerate(json.load(open(f'{K}/tool/plain.json'))):
        if not TREE_X[0] <= p[0] <= TREE_X[1]: continue          # 樹外面的閃爍點不要（右邊電腦螢幕在彩蛋時會扭曲，Henry 2026-11-12）
        balls.append(dict(plain=True, spin=False, x=p[0], y=p[1], a=p[2], b=p[2], i=1000 + j, yaw=0, bk=[0, 0, 0]))
    laps = [n for n in range(1, 400) if P['sMin'] - 1e-9 <= SEG / n <= P['sMax'] + 1e-9]
    assert laps, f'片長 {SEG} 秒找不到 20–36 秒一圈的整數圈數'
    for b in balls:
        b.setdefault('plain', False); b['u'] = rnd(); b['dir'] = -1 if rnd() < 0.5 else 1; b['ph'] = rnd() * 2 * math.pi
        b['r'] = (b['a'] + b['b']) / 2
        if b['spin']:
            n = laps[min(int(b['u'] * len(laps)), len(laps) - 1)]
            b['per'] = SEG / n
    lay = lambda b: 0 if b['plain'] else (2 if b['spin'] else 1)
    balls.sort(key=lambda b: (lay(b), b['r']))            # 閃爍點 → 不轉的球 → 轉的球（最後畫）
    # 閃爍：每 bGap 秒挑一個空著的（不轉的球＋閃爍點），排程在片長內頭尾相接
    stat = [b for b in balls if not b['spin']]
    busy = {b['i']: -1.0 for b in stat}; ev = {b['i']: [] for b in stat}
    dur = P['bFade'] + P['bBack']; t = 0.0; n = 0
    while t < SEG:
        free = [b for b in stat if busy[b['i']] <= t]
        if free:
            b = free[int(rnd() * len(free))]; hold = P['bHold'] * (0.7 + 0.6 * rnd())
            ev[b['i']].append((t, hold)); busy[b['i']] = t + dur + hold
        n += 1; t = n * P['bGap']
    for i, e in ev.items():                                 # 尾巴跨回片頭會跟第一個撞到的，丟掉
        if e:
            first = e[0][0]
            ev[i] = [x for x in e if not (x[0] + dur + x[1] - SEG > first)]
    for b in balls:
        b['ev'] = ev.get(b['i'], [])
    return balls

def dim(b, t):
    t = t % SEG; k = 0.0; dur = P['bFade'] + P['bBack']
    for t0, hold in b['ev']:
        for sh in (0.0, -SEG, SEG):
            u = t - (t0 + sh)
            if u < 0 or u > dur + hold: continue
            v = ease(u / P['bFade']) if u < P['bFade'] else (1 if u < P['bFade'] + hold else 1 - ease((u - P['bFade'] - hold) / P['bBack']))
            k = max(k, v)
    return k * P['bDepth']

def spin_yaw(b, t):
    return b['yaw'] + b['ph'] + b['dir'] * 2 * math.pi * ((t % b['per']) / b['per'])   # 取餘數，頭尾影格完全一樣

# ---------------------------------------------------------------- 渲染
def make_ctx(work):
    import moderngl
    ctx = moderngl.create_standalone_context(backend='egl')
    prog = ctx.program(vertex_shader=VS, fragment_shader=fs_from_tool())
    def tex(path, unit):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA if im.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
        t = ctx.texture((im.shape[1], im.shape[0]), 4, np.ascontiguousarray(im).tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR); t.repeat_x = t.repeat_y = False; t.use(unit); return t
    T = [tex(f'{work}/lit.png', 0), tex(f'{work}/unlit.png', 1), tex(f'{work}/atlas.png', 2), tex(f'{work}/mask.png', 3)]
    for n, u in (('tL', 0), ('tU', 1), ('tA', 2), ('tM', 3)):
        if n in prog: prog[n].value = u
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '2f', 'c')])
    fbo = ctx.simple_framebuffer((W4, H4), components=4); fbo.use()
    ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    return ctx, prog, vao, fbo, T

def put_regions(T, regs, last):
    """螢幕內容有變才更新貼圖（亮燈、熄燈兩張都換成同一份，閃爍點不會把舊畫面帶回來）"""
    for r, a in regs:
        key = (r, hash(a.tobytes()))
        if last.get(r) == key: continue
        last[r] = key
        rgba = np.ascontiguousarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGBA)).tobytes()
        for t in T[:2]: t.write(rgba, viewport=(r[0], r[1], r[2] - r[0], r[3] - r[1]))

def setu(prog, **kw):
    for k, v in kw.items():
        if k in prog: prog[k].value = v

def fx_window(fx):
    if fx == 'base': return None
    D = dict(FX)[fx] * P['fxLen']; idx = [f for f, _ in FX].index(fx) + 1
    t0 = round((SEG / 2 - D / 2) * FPS) / FPS                # 放在片段正中間，對齊影格
    return dict(id=fx, idx=idx, t0=t0, D=D)

def front_target(b, F):
    """Logo 一起轉正：以效果中點的角度找最近的正面；淡入淡出時從原本的旋轉軌跡平滑過去、再平滑回來（回到原軌跡，頭尾才接得上）"""
    step = math.pi if P['dbl'] else 2 * math.pi
    mid = spin_yaw(b, F['t0'] + F['D'] / 2)
    return round(mid / step) * step

def frame(prog, vao, fbo, balls, t, F):
    t = t % SEG                                              # 樹層只看段內秒數（每段一樣，只有彩蛋不同）
    on = F is not None and F['t0'] <= t < F['t0'] + F['D']
    u = t - F['t0'] if on else 0.0
    fw = ease(max(0.0, min(1.0, u / 1.5, (F['D'] - u) / 1.5))) if on else 0.0
    setu(prog, uR=(1920.0, 1080.0), uFx=float(F['idx'] if on and F['id'] != 'front' else 0), uFt=u, uFD=F['D'] if on else 1.0,
         uFw=fw, uFxOn=1.0 if on else 0.0, uIsSpin=0.0, uId=0.0, uFull=1.0, uGain=P['gBg'])
    fbo.clear(0, 0, 0, 0); vao.render(mode=5)
    setu(prog, uFull=0.0, uDbl=1.0 if P['dbl'] else 0.0, uLogo=P['gLogo'])
    for b in balls:
        if b['spin']:
            yaw = spin_yaw(b, t)
            if on and F['id'] == 'front':
                tgt = b['_tgt']; yaw = yaw + fw * (tgt - yaw)
        else:
            yaw = b['yaw']
        setu(prog, uPlain=1.0 if b['plain'] else 0.0, uB=(b['x'], b['y'], b['a'], b['b']),
             uCell=(float(b['i'] % 10), float(b['i'] // 10)), uBk=tuple(b['bk']),
             uGain=P['gBg'] if b['plain'] else (P['gSpin'] if b['spin'] else P['gStatic']),
             uK=0.0 if b['spin'] else dim(b, t), uYaw=yaw, uId=float(b['i']), uIsSpin=1.0 if b['spin'] else 0.0)
        vao.render(mode=5)
    a = np.frombuffer(fbo.read(components=4), np.uint8).reshape(H4, W4, 4)[::-1]
    # 工具的畫布是不預乘透明度、底下是黑色舞台：螢幕上看到的是 rgb×alpha。照工具的樣子輸出。
    rgb = (a[..., :3].astype(np.uint16) * a[..., 3:4] + 127) // 255
    return np.ascontiguousarray(rgb[..., ::-1].astype(np.uint8))   # BGR

def seg(work, out, fx='base', fr0=0, fr1=None, stills=None):
    balls = build_state(); F = fx_window(fx)
    if F and F['id'] == 'front':
        for b in balls:
            if b['spin']: b['_tgt'] = front_target(b, F)
    ctx, prog, vao, fbo, _ = make_ctx(work)
    if stills:
        for s in stills:
            cv2.imwrite(out.replace('.png', f'_{s:g}.png') if out.endswith('.png') else f'{out}_{s:g}.png',
                        frame(prog, vao, fbo, balls, s, F))
        return
    N = int(round(SEG * FPS)); fr1 = N if fr1 is None else min(fr1, N)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W4}x{H4}',
                           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', os.environ.get('X264', 'medium'),
                           '-crf', os.environ.get('CRF', '16'), '-pix_fmt', 'yuv420p',
                           '-x264-params', f'keyint={GOP}:min-keyint={GOP}:scenecut=0', out], stdin=subprocess.PIPE)
    import time; t0 = time.time()
    for f in range(fr0, fr1):
        ff.stdin.write(frame(prog, vao, fbo, balls, f / FPS, F).tobytes())
        if (f - fr0) % 300 == 0: print(f'{fx} {f}/{fr1} {time.time() - t0:.0f}s', flush=True)
    ff.stdin.close(); ff.wait()
    print('done', out, f'{time.time() - t0:.0f}s')

def make_plan(n_seg, seed=1):
    """每段放不放彩蛋：每隔 3、3、4 段（隨機）一個，平均約 10 分鐘；9 個輪完才重洗，不連續重複"""
    import random
    rng = random.Random(seed); names = [f for f, _ in FX]
    plan, bag, last, gap = [], [], None, rng.choice([3, 3, 4])
    for k in range(n_seg):
        fx = None; gap -= 1
        if gap == 0:
            if not bag:
                bag = names[:]; rng.shuffle(bag)
                if bag[0] == last: bag.append(bag.pop(0))
            fx = bag.pop(0); last = fx; gap = rng.choice([3, 3, 4])
        plan.append(dict(seg=k, start=k * SEG, fx=fx))
    return plan

def abs_segments(work, logo_dirs, outdir, k0, k1, hours=4, seed=1):
    """正式版：第 k0～k1-1 段，每段一個檔 seg_kkk.mp4；螢幕依影片的絕對秒數變化（給 Actions 分批平行算）"""
    import screens as SC
    os.makedirs(outdir, exist_ok=True)
    plan = make_plan(math.ceil(hours * 3600 / SEG), seed)
    scr = SC.Screens(dict(zip('abch', logo_dirs)), cv2.imread(f'{work}/lit.png'))
    for k in range(k0, min(k1, len(plan))):
        p = plan[k]; balls = build_state(); F = fx_window(p['fx'] or 'base')
        if F and F['id'] == 'front':
            for b in balls:
                if b['spin']: b['_tgt'] = front_target(b, F)
        ctx, prog, vao, fbo, T = make_ctx(work); last = {}
        out = f'{outdir}/seg_{k:03d}.mp4'
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W4}x{H4}',
                               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', os.environ.get('X264', 'medium'),
                               '-crf', os.environ.get('CRF', '16'), '-pix_fmt', 'yuv420p',
                               '-x264-params', f'keyint={GOP}:min-keyint={GOP}:scenecut=0', out], stdin=subprocess.PIPE)
        N = int(round(SEG * FPS)); n0 = int(os.environ.get('FRAMES', N))
        for f in range(min(N, n0)):
            ta = p['start'] + f / FPS
            put_regions(T, scr.regions(ta), last)
            ff.stdin.write(frame(prog, vao, fbo, balls, ta, F).tobytes())
        ff.stdin.close(); ff.wait(); ctx.release()
        print('done', out, p['fx'] or '-', flush=True)

def mid_window():
    """所有彩蛋都落在這一段（對齊 2 秒關鍵格）：基本片段切成 head / mid / tail，彩蛋片段只換 mid"""
    D = max(d for _, d in FX) * P['fxLen']; g = GOP / FPS
    a = math.floor((SEG / 2 - D / 2) / g) * g; b = math.ceil((SEG / 2 + D / 2) / g) * g
    return int(round(a * FPS)), int(round(b * FPS))

def pieces(work, outdir, names):
    os.makedirs(outdir, exist_ok=True); a, b = mid_window(); N = int(round(SEG * FPS))
    for n in names or ['head', 'mid', 'tail'] + [f for f, _ in FX]:
        if n == 'head': seg(work, f'{outdir}/base_head.mp4', 'base', 0, a)
        elif n == 'tail': seg(work, f'{outdir}/base_tail.mp4', 'base', b, N)
        elif n == 'mid': seg(work, f'{outdir}/base_mid.mp4', 'base', a, b)
        else: seg(work, f'{outdir}/fx_{n}.mp4', n, a, b)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'prep': prep(a[1], a[2])
    elif a[0] == 'pieces': pieces(a[1], a[2], a[3:])
    elif a[0] == 'abs':      # abs <work> <a> <b> <c> <h> <outdir> <k0> <k1>
        abs_segments(a[1], a[2:6], a[6], int(a[7]), int(a[8]))
    else:
        opt = {}; pos = []
        i = 1
        while i < len(a):
            if a[i].startswith('--'): opt[a[i][2:]] = a[i + 1]; i += 2
            else: pos.append(a[i]); i += 1
        work, out = pos[0], pos[1]; fx = pos[2] if len(pos) > 2 else 'base'
        st = [float(x) for x in opt['still'].split(',')] if 'still' in opt else None
        seg(work, out, fx, int(opt.get('from', 0)), int(opt['to']) if 'to' in opt else None, st)
