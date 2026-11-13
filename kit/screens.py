"""xms1 背景三塊螢幕（Henry 2026-11-12 定案的內容）。改的是 4K 亮燈／熄燈底圖，渲染時當貼圖用，
所以透過玻璃球看到的也會是新畫面。所有變化都只由「影片第幾秒」決定（亂數固定），分段算接得起來。

- 左：Microsoft 那塊只換中間的 Logo，用樹上 79 個 Logo 隨機輪流、不重複（每段開頭換）；
      Logo 跟深藍底太接近時，連那塊底色一起換成品牌色。
- 右：只換中間的走勢曲線，隨機走勢，綠（往上）紅（往下）交錯；在每段 1/2 處換，跟左邊錯開。
- 遠方大看板：不換內容，每格數字各自在紅／綠／黃之間跳色，每 5～10 秒跳一次。

usage（預覽）: screens.py <badges 用的四個 Logo 資料夾 a b c h> <work> <out.jpg>
"""
import json, math, os, sys
import numpy as np, cv2
from PIL import Image

K = os.path.dirname(os.path.abspath(__file__))
SEG = float(os.environ.get('SEG', 180))
XF = float(os.environ.get('XF', 0))              # 換畫面淡入淡出秒數，0＝直接切（待 Henry 決定）
NAVY = np.array([52, 22, 14], np.float32)        # 螢幕底色（BGR，從原圖取）

# ---------------------------------------------------------------- 區域（4K 熄燈圖座標，量自原圖）
# 左：Logo 放置區（中心、寬、高、隨螢幕的傾斜）與要清掉的原 Logo 範圍、品牌底色範圍
# 角度一律取螢幕「裡面」的線，不取外框（Henry 2026-11-12）：
#   左螢幕：原圖 Microsoft 四個方塊的列與字的走向約 13.5°；方塊的直行往上略往右倒（每往上 1 px 往右 0.09 px）
#   右螢幕：走勢圖的 X 軸線 (3553,732)→(3805,685)；上方表頭下緣線 (3680,463)→(3782,437)；圖右緣直線 (3797,593)→(3805,683)
LOGO_C, LOGO_W, LOGO_H = (112, 682), 150, 60
LOGO_ANG, LOGO_LEAN = math.radians(13.5), 0.09
AXIS = ((3553, 732), (3805, 685)); HEADER = ((3680, 463), (3782, 437)); RIGHT = ((3797, 593), (3805, 683))
def _logo_box(w, h, c=None):
    """以 Logo 中心、照原 Microsoft 的走向（13.5° 與直行斜度）圍出寬 w 高 h 的框"""
    cx, cy = c or LOGO_C; ux = np.array([math.cos(LOGO_ANG), math.sin(LOGO_ANG)]); uy = np.array([-LOGO_LEAN, 1.0])
    return np.array([[cx, cy] - ux * w / 2 - uy * h / 2, [cx, cy] + ux * w / 2 - uy * h / 2,
                     [cx, cy] + ux * w / 2 + uy * h / 2, [cx, cy] - ux * w / 2 + uy * h / 2], np.float32)
LOGO_ERASE = _logo_box(156, 98, (112, 683))
LOGO_PLATE = _logo_box(178, 140)
# 右：走勢圖的繪圖區（左下、右下、右上、左上）
def _isect(a, b, c, d):
    a, b, c, d = (np.array(v, np.float64) for v in (a, b, c, d)); r = b - a; q = d - c
    cr = lambda u, w: u[0] * w[1] - u[1] * w[0]
    t = cr(c - a, q) / cr(r, q); return a + r * t
def _chart_quad(xl=3565, xr=3803, h=215, lift=0.0):
    (ax0, ay0), (ax1, ay1) = AXIS; k = (ay1 - ay0) / (ax1 - ax0)
    BL = np.array([xl, ay0 + k * (xl - ax0) - lift]); BR = np.array([xr, ay0 + k * (xr - ax0) - lift])
    vp = _isect(AXIS[0], AXIS[1], HEADER[0], HEADER[1])                     # 水平線的消失點（透視）
    up = np.array(RIGHT[0], float) - np.array(RIGHT[1], float); up /= np.linalg.norm(up)   # 往上的方向（照圖右緣直線）
    TL = BL + up * h; TR = _isect(TL, vp, BR, BR + up)
    return np.array([BL, BR, TR, TL], np.float32)
CHART = _chart_quad(3567, 3803, 200)
# 遠方大看板
BOARD = (2826, 560, 3472, 742)
# 每塊螢幕會改到的範圍（x0, y0, x1, y1）：渲染時只更新這幾塊貼圖
ROI_L, ROI_C = (0, 560, 260, 860), (3460, 420, 3840, 800)
ROIS = [ROI_L, ROI_C, BOARD]

# 螢幕的顯色：用原圖 Microsoft 四個方塊＋白字＋黑底，跟官方色比對擬合出來（新內容一律經過這一步，色溫、亮度、飽和度才跟原螢幕一樣）
SCREEN_M = np.array([[0.598, 0.084, 0.074, 17.26], [-0.101, 0.732, 0.037, 9.43], [-0.196, 0.409, 0.287, 12.426]], np.float32)
SOFT = 1.1                                       # 原螢幕畫面的柔和程度（高斯半徑，4K 像素）
GRAIN = 1.0                                      # 原螢幕底色的細微雜訊（標準差）
def tone(bgr):
    """把「想顯示的顏色」換成「原螢幕實際顯示出來的顏色」（BGR 0–255）"""
    a = np.asarray(bgr, np.float32); return np.clip(a @ SCREEN_M[:, :3].T + SCREEN_M[:, 3], 0, 255)
def soft_comp(img, col, a, glow=0.12):
    """新內容跟原螢幕一樣柔：顏色（預乘）和透明度一起模糊，再加一點發光"""
    a = a.astype(np.float32)
    a = a[..., None] if a.ndim == 2 else a
    pc = cv2.GaussianBlur(col * a, (0, 0), SOFT); a = cv2.GaussianBlur(a, (0, 0), SOFT)[..., None]
    img[:] = img * (1 - a) + pc
    if glow: img[:] = img + cv2.GaussianBlur(pc, (0, 0), 4) * glow

# ---------------------------------------------------------------- Logo（取 make_badges 同一個圖案，不加圓底）
def hexrgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)

def mark(cfg, dirs, t):
    e = cfg[t]
    if e[0] == 'text':
        return None, hexrgb(e[2])
    d, i, bg, mode = e[:4]
    im = Image.open(f'{dirs[d]}/{t}_{i}.png').convert('RGBA'); im = im.crop(im.getbbox())
    if len(e) > 4:
        x0, y0, x1, y1 = e[4]; W_, H_ = im.size; im = im.crop((int(x0 * W_), int(y0 * H_), int(x1 * W_), int(y1 * H_)))
    a = np.asarray(im).astype(np.float32); bgc = hexrgb(bg); rgb, al = a[..., :3], a[..., 3:] / 255
    if mode == 'white':
        dist = 255 - rgb.min(2, keepdims=True); al = al * np.clip(dist / 60, 0, 1); rgb = np.full_like(rgb, 255)
    elif mode == 'invertdark':
        lum = (rgb @ np.array([.2126, .7152, .0722], np.float32))[..., None]; k = np.clip((110 - lum) / 60, 0, 1); rgb = rgb * (1 - k) + 255 * k
    elif mode == 'plate':
        op = al[..., 0] > 0.9; px = rgb[op]; q = (px // 16).astype(int); key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        bgc = px[key == np.bincount(key).argmax()].mean(0)
        comp = rgb * al + bgc * (1 - al); diff = np.sqrt(((comp - bgc) ** 2).sum(2)); al = al * np.clip((diff - 25) / 40, 0, 1)[..., None]
    if mode != 'round':
        comp = rgb * al + bgc * (1 - al); diff = np.sqrt(((comp - bgc) ** 2).sum(2))
        ys, xs = np.nonzero((diff > 45) & (al[..., 0] > 0.15))
        if len(xs): rgb, al = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], al[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.dstack([rgb[..., ::-1], al]).astype(np.float32), bgc[::-1]       # BGR + alpha(0–1)

def contrast_low(m, navy_rgb):
    """Logo 本身的顏色跟深藍底太接近 → 要連底色換掉"""
    a = m[..., 3]; w = a > 0.5
    if w.sum() < 20: return True
    px = m[..., :3][w]; lum = lambda c: c @ np.array([.0722, .7152, .2126], np.float32)
    far = np.sqrt(((px - navy_rgb) ** 2).sum(1)) > 110
    return far.mean() < 0.5 or np.median(lum(px)) < 70

# ---------------------------------------------------------------- 排程（只看秒數）
def logo_order(ticks, n):
    rng = np.random.default_rng(2025); out = []
    while len(out) < n:
        p = list(rng.permutation(len(ticks)))
        if out and p[0] == out[-1]: p.append(p.pop(0))
        out += p
    return [ticks[i] for i in out[:n]]

def chart_curve(k, n=120):
    """第 k 張走勢：隨機漫步，偶數張綠色往上、奇數張紅色往下"""
    rng = np.random.default_rng(1000 + k); up = k % 2 == 0
    v = np.cumsum(rng.normal(0.012 if up else -0.012, 0.045, n)); v -= v.min(); v /= max(v.max(), 1e-6)
    return 0.12 + 0.76 * v, up

def board_cells(L):
    x0, y0, x1, y1 = BOARD; reg = L[y0:y1, x0:x1].astype(np.float32)
    hsv = cv2.cvtColor(L[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
    col = (hsv[..., 1] > 90) & (hsv[..., 2] > 70) & ((hsv[..., 0] < 35) | (hsv[..., 0] > 160) | ((hsv[..., 0] > 40) & (hsv[..., 0] < 90)))
    m = cv2.dilate(col.astype(np.uint8), np.ones((3, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cells = [j for j in range(1, n) if st[j, 4] > 25]
    return col, lab, cells

HUE = {'g': (75, 203), 'r': (175, 199), 'y': (16, 147)}       # OpenCV H、S：取原看板紅綠黃各自的中位數
BOARD_FIX = 30.0      # 每 600 秒區塊交界前這幾秒（拉長到 30 秒，換色分散開來比較不明顯），各格在隨機時間換成下一區塊開頭的顏色（Henry 2026-11-13：要隨機才自然）

def board_state(cells, t):
    """每格的顏色＝board_raw，但在區塊交界（600 秒的倍數）前 BOARD_FIX 秒內，每格在自己的隨機時間
    先換成交界那一刻的顏色，交界時就不會整片一起變。交界以外的時間跟原本完全一樣。"""
    raw = board_raw(cells, t)
    B = (int(t // 600) + 1) * 600
    if B - t > BOARD_FIX or B <= 0: return raw
    tgt = board_raw(cells, float(B))
    for j in cells:
        sw = B - 0.5 - np.random.default_rng(j * 104729 + B).uniform(0, BOARD_FIX - 1.0)
        if t >= sw: raw[j] = tgt[j]
    return raw

def board_raw(cells, t):
    """每格各自的顏色：從固定亂數出發，每 5～10 秒換一次（紅綠各 45%、黃 10%），只看秒數"""
    out = {}
    for j in cells:
        # 每 600 秒一個區塊各自用固定亂數，跳到 t 附近不必從頭模擬 4 小時
        blk = int(t // 600); r2 = np.random.default_rng((j * 7919 + 13) * 100003 + blk)
        tt = blk * 600 - r2.uniform(0, 10); c = r2.choice(['g', 'r', 'y'], p=[.45, .45, .1])
        while True:
            nx = tt + r2.uniform(5, 10)
            if nx > t: break
            tt = nx; c = r2.choice([x for x in 'gry' if x != c], p=None if c == 'y' else ([.82, .18] if c == 'g' else [.82, .18]))
        out[j] = c
    return out

class Screens:
    def __init__(s, dirs, L):
        s.cfg = json.load(open(f'{K}/badges.json')); s.ticks = list(s.cfg); s.dirs = dirs
        s.marks = {t: mark(s.cfg, dirs, t) for t in s.ticks}
        s.base = L.copy()
        # 左：先把原本的 Microsoft 清成深藍
        s.base = L.copy(); s._smooth_fill(ROI_L, LOGO_ERASE)
        # 右：繪圖區（X 軸上方、表頭下方，四邊照原圖的線）整塊換成平滑的底色：
        #   底色＝繪圖區裡沒有曲線、沒有填色的暗色像素，用二次曲面擬合（保留原本由上到下的漸層），邊緣柔化
        # 右螢幕的亮度基準：原曲線線條、原填色的平均亮度（新的綠／紅照這個亮度，只換色相）
        r = ROI_C; reg = L[r[1]:r[3], r[0]:r[2]]; hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV).astype(np.int32)
        area = np.zeros(reg.shape[:2], np.uint8); cv2.fillPoly(area, [(_chart_quad(3562, 3806, 228) - np.float32(r[:2])).astype(np.int32)], 1)
        warm = ((hsv[..., 0] < 35) | (hsv[..., 0] > 125)) & (hsv[..., 1] > 35) & (area > 0)
        lum = reg.astype(np.float32) @ np.array([.0722, .7152, .2126], np.float32)
        s.line_lum = float(np.percentile(lum[warm], 99.5)); s.fill_lum = float(np.mean(lum[warm & (lum > 20) & (lum < s.line_lum * 0.5)]))
        fp = warm & (lum > 20) & (lum < s.line_lum * 0.5)
        s.fill_sv = (int(np.median(hsv[..., 1][fp])), int(np.median(hsv[..., 2][fp])))   # 原填色的飽和度、明度
        s._smooth_fill(ROI_C, _chart_quad(3562, 3806, 228, lift=2.5))
        s.col, s.lab, s.cells = board_cells(L)
        s.order = logo_order(s.ticks, 400)

    def _smooth_fill(s, r, quad):
        """quad 範圍換成平滑底色：用範圍內沒有內容的暗色像素擬合二次曲面（保留原漸層），加原螢幕的細雜訊，邊緣柔化"""
        o = np.float32(r[:2]); reg = s.base[r[1]:r[3], r[0]:r[2]].astype(np.float32); q = quad - o
        area = np.zeros(reg.shape[:2], np.float32); cv2.fillPoly(area, [np.round(q * 8).astype(np.int32)], 1.0, cv2.LINE_AA, 3)
        hsv = cv2.cvtColor(reg.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int32)
        busy = ((hsv[..., 1] > 35) & ((hsv[..., 0] < 35) | (hsv[..., 0] > 125))) | (hsv[..., 2] > 60)
        ok = (area > 0.99) & ~cv2.dilate(busy.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        yy, xx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]].astype(np.float32)
        Xf = lambda x, y: np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1) / np.array([1, 400, 400, 16e4, 16e4, 16e4], np.float32)
        fitc = np.linalg.lstsq(Xf(xx[ok], yy[ok]), reg[ok], rcond=None)[0]
        smooth = Xf(xx, yy) @ fitc + np.random.default_rng(5).normal(0, GRAIN, reg.shape).astype(np.float32)
        a = cv2.GaussianBlur(area, (0, 0), 0.8)[..., None]
        s.base[r[1]:r[3], r[0]:r[2]] = (smooth * a + reg * (1 - a)).clip(0, 255).astype(np.uint8)

    def draw_logo_roi(s, img, t):
        o = np.float32(ROI_L[:2])
        tk = s.order[int(t // SEG)]; m, bgc = s.marks[tk]
        if m is None or contrast_low(m, NAVY):
            pm = np.zeros(img.shape[:2], np.float32); cv2.fillPoly(pm, [(LOGO_PLATE - o).astype(np.int32)], 1.0)
            pm = cv2.GaussianBlur(pm, (0, 0), 1.0)[..., None]
            soft_comp(img, np.broadcast_to(tone(bgc), img.shape).copy(), pm[..., 0], glow=0)
            if m is None: return tk
        h, w = m.shape[:2]; sc = min(LOGO_W / w, LOGO_H / h)
        cx, cy = LOGO_C[0] - o[0], LOGO_C[1] - o[1]
        ux = np.array([math.cos(LOGO_ANG), math.sin(LOGO_ANG)]) * sc          # Logo 的水平方向＝原 Microsoft 的走向
        uy = np.array([-LOGO_LEAN, 1.0]) * sc                                   # 往下的方向（直行略斜）
        A = np.float32([[ux[0], uy[0], cx - ux[0] * w / 2 - uy[0] * h / 2],
                        [ux[1], uy[1], cy - ux[1] * w / 2 - uy[1] * h / 2]])
        warp = cv2.warpAffine(m, A, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA, borderValue=0)
        soft_comp(img, tone(warp[..., :3]), warp[..., 3])
        return tk

    def draw_chart_roi(s, img, t):
        """走勢畫在正面的畫布上，再用透視矩陣貼到原本繪圖區（四邊照原圖裡的線）"""
        o = np.float32(ROI_C[:2])
        k = int((t - SEG / 2) // SEG) + 1 if t >= SEG / 2 else 0
        v, up = chart_curve(k); n = len(v)
        def at_lum(c, target):                                                  # 經過螢幕顯色後，亮度對齊原曲線
            c = tone(c); return np.clip(c * target / max(float(c @ np.array([.0722, .7152, .2126])), 1), 0, 255)
        col = at_lum([60, 230, 40] if up else [40, 40, 255], s.line_lum)
        fill = cv2.cvtColor(np.uint8([[[60 if up else 178, *s.fill_sv]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)   # 填色：飽和度、明度跟原填色一樣，只換色相
        CW, CH = 960, 860; sx = 4                                               # 畫布（下＝X 軸）
        xs = np.linspace(0, CW, n); ys = CH * (1 - v)
        pts = np.round(np.stack([xs, ys], 1) * 16).astype(np.int32)
        fm = np.zeros((CH, CW), np.float32)
        cv2.fillPoly(fm, [np.vstack([pts, [[CW * 16, CH * 16], [0, CH * 16]]])], 1.0, cv2.LINE_AA, 4)
        g = np.linspace(0, 1, CH, dtype=np.float32)[:, None]
        fa = fm * (0.95 - 0.5 * g)                                               # 由上往下漸淡
        lm = np.zeros((CH, CW), np.float32); cv2.polylines(lm, [pts], False, 1.0, 18, cv2.LINE_AA, 4)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, CH], [CW, CH], [CW, 0], [0, 0]]), CHART - o)
        size = (img.shape[1], img.shape[0])
        wp = lambda a: cv2.warpPerspective(a, Hm, size, flags=cv2.INTER_AREA)
        fa, lm = wp(fa), np.clip(wp(lm) * 1.3, 0, 1)
        soft_comp(img, np.broadcast_to(fill, img.shape).copy(), fa, glow=0)
        soft_comp(img, np.broadcast_to(col, img.shape).copy(), lm, glow=0.3)
        return ('綠' if up else '紅') + str(k)

    def draw_board_roi(s, img, t):
        st = board_state(s.cells, t)
        reg = img.clip(0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.float32); Sx = hsv[..., 1].astype(np.float32)
        for j, c in st.items():
            sel = (s.lab == j) & s.col
            H[sel], Sx[sel] = HUE[c]
        hsv[..., 0] = H.astype(np.uint8); hsv[..., 1] = Sx.astype(np.uint8)
        img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    def at(s, t):
        """回傳第 t 秒的 4K 底圖（BGR uint8）"""
        img = s.base.copy(); f = {}
        for r in ROIS: f[r] = img[r[1]:r[3], r[0]:r[2]].astype(np.float32)
        tk = s.draw_logo_roi(f[ROI_L], t); ch = s.draw_chart_roi(f[ROI_C], t); s.draw_board_roi(f[BOARD], t)
        for r in ROIS: img[r[1]:r[3], r[0]:r[2]] = f[r].clip(0, 255).astype(np.uint8)
        return img, tk, ch

    def regions(s, t):
        """只回傳三塊改過的範圍 [(roi, BGR uint8)]，給渲染程式更新貼圖用"""
        out = []
        for r, fn in ((ROI_L, s.draw_logo_roi), (ROI_C, s.draw_chart_roi), (BOARD, s.draw_board_roi)):
            a = s.base[r[1]:r[3], r[0]:r[2]].astype(np.float32); fn(a, t); out.append((r, a.clip(0, 255).astype(np.uint8)))
        return out

if __name__ == '__main__':
    a, b, c, h, work, out = sys.argv[1:7]
    L = cv2.imread(f'{work}/lit.png'); scr = Screens(dict(zip('abch', (a, b, c, h))), L)
    rows = []
    for t in (0, 95, 185, 280, 370, 460):
        img, tk, ch = scr.at(t)
        crops = [cv2.resize(img[560:860, 0:260], None, fx=2, fy=2), cv2.resize(img[420:800, 3460:3840], (380 * 520 // 380, 520)),
                 cv2.resize(img[500:800, 2790:3500], (710 * 520 // 300, 520))]
        r = np.hstack([cv2.resize(x, (int(x.shape[1] * 520 / x.shape[0]), 520)) for x in crops])
        cv2.putText(r, f't={t}s {tk} {ch}', (10, 40), 0, 1.1, (0, 255, 255), 2); rows.append(r)
    wmax = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(len(scr.cells), 'cells')
