"""xms1 片頭音效（依 HANDOFF：暗場＋交易所人聲 → 第 9 秒鐘聲（左上二樓、偏左、帶回音，含掌聲歡呼）→ 人聲淡出 → 第一首）。
usage: intro_audio.py <out.wav> [--preview <秒數>]
  輸出只有「片頭專屬」的聲音（鐘聲、交易所人聲、電話、鍵盤），長 INTRO_LEN 秒。
  --preview：另外疊上全片持續的環境底噪＋第一首（從 MUSIC_AT 開始），給 Henry 試聽用；正式版的底噪和音樂由 4 小時音軌負責。
時間對齊 intro.py：鐘聲錄音裡的鐘從 6.9 秒開始，放在片頭 2.1 秒 → 片頭第 9.0 秒響鐘、22.8 秒停（燈剛好全亮）。
"""
import os, subprocess, sys
import numpy as np, scipy.signal as ss, scipy.io.wavfile as wv

K = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(f'{K}/../..')
A = f'{ROOT}/videos/xms1/assets'; AMB = f'{A}/ambience'; MUS = f'{A}/music'
SR = 48000
INTRO_LEN = 33.0                 # 片頭專屬聲音的長度（歡呼尾巴會蓋到第一首開頭一點點）
BELL_FILE_AT = 2.1               # 鐘聲檔放的位置（檔內 6.9 秒響鐘）
MUSIC_AT = 29.5                  # 第一首開始（片頭畫面 31 秒接正式第 0 段）
FIRST = 'pulsebox-slow-christmas-jazz-432316.mp3'

def load(p, off=0, dur=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(off), '-i', p] + (['-t', str(dur)] if dur else []) + ['-ac', '2', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).reshape(-1, 2).copy()

def lufs(x):
    p = f'/tmp/_xms1_lufs_{os.getpid()}.wav'; wv.write(p, SR, x.astype(np.float32))
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', p, '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
    os.remove(p); return float([l for l in r.split('\n') if l.strip().startswith('I:')][-1].split()[1])

def to(x, target): return (x * 10 ** ((target - lufs(x)) / 20)).astype(np.float32)
def place(buf, x, t):
    i = int(t * SR); n = min(len(x), len(buf) - i)
    if n > 0: buf[i:i + n] += x[:n]
def env(n, pts):
    t = np.arange(n) / SR; ts, ds = zip(*pts); return (10 ** (np.interp(t, ts, ds) / 20)).astype(np.float32)[:, None]

def intro_stem():
    N = int(SR * INTRO_LEN); out = np.zeros((N, 2), np.float32)
    # 鐘聲＋歡呼：鐘在左上二樓 → 高頻（鐘的泛音）偏左、加回音
    bell = load(f'{A}/opening-bell-crowd.mp3')
    b, a = ss.butter(4, 1800 / (SR / 2), 'high'); hi = ss.filtfilt(b, a, bell, axis=0); lo = bell - hi
    hi = hi * np.array([1.0, 0.45], np.float32)
    ir = (np.random.default_rng(1).standard_normal(int(SR * 1.8)) * np.exp(-np.arange(int(SR * 1.8)) / (SR * 0.45))).astype(np.float32); ir /= np.sqrt((ir ** 2).sum())
    wet = np.stack([ss.fftconvolve(hi[:, 0], ir)[:len(hi)], ss.fftconvolve(hi[:, 1], ir[::-1].copy())[:len(hi)]], 1)
    bell = lo + hi * 0.85 + wet * 0.35
    b, a = ss.butter(2, 9000 / (SR / 2)); bell = ss.filtfilt(b, a, bell, axis=0).astype(np.float32)
    bell = to(bell, -21) * env(len(bell), [(0, -14), (4.8, -2), (6.9, 0), (24.5, 0), (27.5, -6), (29.9, -60)])
    place(out, bell, BELL_FILE_AT)
    # 開盤前的交易所：遠近人聲，鐘響後慢慢淡出
    crowd = to(load(f'{ROOT}/videos/wst1/assets/trading-floor-crowd-quicksounds.mp3', 20, 28), -31)
    talk = to(load(f'{AMB}/freesound_community-people-talking-20327.mp3', 0, 28), -32)
    busy = crowd[:SR * 28] + talk[:SR * 28]
    place(out, busy * env(len(busy), [(0, -8), (1.0, 0), (17, 0), (25, -12), (28, -70)]), 0)
    # 電話、鍵盤（只在片頭）
    ph = load(f'{AMB}/freesound_community-office_phone-ring_medium-loudaif-14604_1.mp3', 0, 3)
    b, a = ss.butter(2, [400 / (SR / 2), 3500 / (SR / 2)], 'band'); ph = to(ss.filtfilt(b, a, ph, axis=0).astype(np.float32), -38)
    e = env(len(ph), [(0, 0), (2.4, 0), (3, -60)])
    place(out, ph * np.array([0.5, 1.0], np.float32) * e, 2.6)
    place(out, ph * np.array([1.4, 0.4], np.float32) * 0.7 * e, 6.0)
    kb = to(load(f'{AMB}/rmultimediaeu-keyboard-typing-sound-250308.mp3', 0, 5), -36) * np.array([0.6, 1.0], np.float32)
    place(out, kb * env(len(kb), [(0, -30), (0.4, 0), (4.5, 0), (5, -60)]), 3.2)
    return out

def bed(dur):
    """全片持續的環境底噪（預覽用；正式版在 4 小時音軌裡做）"""
    x = to(load(f'{AMB}/cliffordjohnson-company-office-work-modern-contemporary-534889.mp3', 30, dur), -38)
    y = to(load(f'{AMB}/freesound_community-museum-ambience-17352.mp3', 10, dur), -42)
    z = to(load(f'{AMB}/freesound_community-012631_people-talking-loud-in-the-distance-at-late-night-58293.mp3', 0, dur), -45)
    n = min(len(x), len(y), len(z)); return x[:n] + y[:n] + z[:n]

if __name__ == '__main__':
    out = sys.argv[1]; st = intro_stem()
    if '--preview' in sys.argv:
        D = float(sys.argv[sys.argv.index('--preview') + 1]); N = int(SR * D)
        mix = np.zeros((N, 2), np.float32); place(mix, st, 0)
        bd = bed(D); place(mix, bd * env(len(bd), [(0, -8), (22, -8), (29, 0), (D, 0)]), 0)
        m = to(load(f'{MUS}/{FIRST}', 0, D - MUSIC_AT), -20); place(mix, m * env(len(m), [(0, -24), (1.5, 0), (999, 0)]), MUSIC_AT)
        mix *= env(N, [(0, -60), (0.4, 0), (D - 1.5, 0), (D, -60)]); st = mix
    pk = np.abs(st).max()
    if pk > 0.95: st *= 0.95 / pk
    wv.write(out, SR, st.astype(np.float32)); print('peak', round(float(pk), 3), 'LUFS', lufs(st))
