"""xms1 片頭（樹層畫面）：暗場 → 第 9 秒鐘聲 → 燈由下往上、先慢後快亮起，鐘聲結束全亮 → 閃爍在 6.5 秒內從 0 加速到定速
→ 第 T 秒剛好接上正式第 0 段的第 0 格（旋轉、閃爍、螢幕都接得上）。

usage: intro.py <work> <logo a> <b> <c> <h> <out.mp4> [--still 秒,秒]
時間軸（片頭秒數）：
  0–1.5   由黑淡入（暗場：樹不亮、Logo 很暗、環境只剩螢幕和燈）
  9.0     鐘聲開始（音軌另做，見 intro_audio.py）
  9.0–22.8 燈由下往上亮（前緣位置 ∝ 進度^2.2，先慢後快），環境同步恢復；22.8 鐘聲結束＝全亮
  22.8–29.3 閃爍出現的機率從 0 平滑升到 1（之後就是正常閃爍）
  31.0    接正式第 0 段（樹層時間＝片頭秒數−31，所以片頭其實是循環的最後 31 秒）
螢幕：左 Logo、右走勢＝第 0 段的第一張；遠方看板照常跳色，片頭裡每格在隨機時間換到第 0 段開頭的顏色，接點不會整片一起變。
"""
import math, os, subprocess, sys
import numpy as np, cv2

K = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, K)
import render_seg as R, screens as SC

T = 31.0
LIT0, LIT1 = 9.0, 22.8
BLK0, BLK1 = 22.8, 29.3
FADE_IN = 1.5

def it_at(t):
    if t < LIT0: return -0.15
    if t >= LIT1: return 10.0
    return -0.15 + 1.32 * ((t - LIT0) / (LIT1 - LIT0)) ** 2.2

def env_at(t):
    return 1.0 if t < LIT0 else max(0.0, 1.0 - ((t - LIT0) / (LIT1 - LIT0)) ** 1.6)   # 環境跟著燈一起亮回來

def p_blink(ta):
    return R.ease((ta - BLK0) / (BLK1 - BLK0))

def hsh(i, x):
    return (math.sin(i * 12.9898 + x * 78.233) * 43758.5453) % 1.0

def kfun_at(t_abs):
    """閃爍：跟正式段落同一份排程，但事件開始時間在片頭的話，依 p_blink 決定要不要出現（29.3 秒以後全部出現）"""
    tt = t_abs - T; dur = R.P['bFade'] + R.P['bBack']
    def kf(b, _):
        k = 0.0
        for t0, hold in b['ev']:
            for sh in (0.0, -R.SEG):
                u = tt - (t0 + sh)
                if u < 0 or u > dur + hold: continue
                if hsh(b['i'], t0) >= p_blink(t0 + sh + T): continue
                v = R.ease(u / R.P['bFade']) if u < R.P['bFade'] else (1 if u < R.P['bFade'] + hold else 1 - R.ease((u - R.P['bFade'] - hold) / R.P['bBack']))
                k = max(k, v)
        return k * R.P['bDepth']
    return kf

class Board:
    """片頭的看板：第 -1 區塊的跳色，每格在片頭裡隨機一個時間換成第 0 段開頭的顏色"""
    def __init__(s, scr):
        s.scr = scr; s.target = SC.board_state(scr.cells, 0.0)
        rng = np.random.default_rng(31); s.sw = {j: -rng.uniform(1.0, 25.0) for j in scr.cells}
    def draw(s, img, t_rel):
        st = SC.board_state(s.scr.cells, t_rel)
        st = {j: (s.target[j] if t_rel >= s.sw[j] else c) for j, c in st.items()}
        reg = img.clip(0, 255).astype(np.uint8); hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.float32); Sx = hsv[..., 1].astype(np.float32)
        for j, c in st.items():
            sel = (s.scr.lab == j) & s.scr.col; H[sel], Sx[sel] = SC.HUE[c]
        hsv[..., 0] = H.astype(np.uint8); hsv[..., 1] = Sx.astype(np.uint8)
        img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

def regions(scr, board, t_abs):
    out = []
    for r, fn in ((SC.ROI_L, lambda a: scr.draw_logo_roi(a, 0.0)), (SC.ROI_C, lambda a: scr.draw_chart_roi(a, 0.0)),
                  (SC.BOARD, lambda a: board.draw(a, t_abs - T))):
        a = scr.base[r[1]:r[3], r[0]:r[2]].astype(np.float32); fn(a); out.append((r, a.clip(0, 255).astype(np.uint8)))
    return out

def main():
    a = sys.argv[1:]; still = None
    if '--still' in a:
        i = a.index('--still'); still = [float(x) for x in a[i + 1].split(',')]; a = a[:i] + a[i + 2:]
    work, la, lb, lc, lh, out = a
    scr = SC.Screens(dict(zip('abch', (la, lb, lc, lh))), cv2.imread(f'{work}/lit.png')); board = Board(scr)
    balls = R.build_state(); ctx, prog, vao, fbo, T_ = R.make_ctx(work); last = {}
    def render(t_abs):
        R.put_regions(T_, regions(scr, board, t_abs), last)
        f = R.frame(prog, vao, fbo, balls, t_abs - T, None, dict(it=it_at(t_abs), env=env_at(t_abs), kfun=kfun_at(t_abs)))
        if t_abs < FADE_IN: f = (f.astype(np.float32) * R.ease(t_abs / FADE_IN)).astype(np.uint8)
        return f
    if still:
        for t in still: cv2.imwrite(f'{out}_{t:g}.png', render(t))
        return
    N = int(round(T * R.FPS)); n = int(os.environ.get('FRAMES', N))
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{R.W4}x{R.H4}',
                           '-r', str(R.FPS), '-i', '-', '-c:v', 'libx264', '-preset', os.environ.get('X264', 'medium'),
                           '-crf', os.environ.get('CRF', '18'), '-pix_fmt', 'yuv420p',
                           '-x264-params', f'keyint={R.GOP}:min-keyint={R.GOP}:scenecut=0', out], stdin=subprocess.PIPE)
    for f in range(min(n, N)):
        ff.stdin.write(render(f / R.FPS).tobytes())
        if f % 150 == 0: print('intro', f, '/', N, flush=True)
    ff.stdin.close(); ff.wait(); print('done', out)

if __name__ == '__main__':
    main()
