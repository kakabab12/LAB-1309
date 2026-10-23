"""4K 시연 화면 (10/22, 사용자: "일지 쓸때마다 정확도 높은걸로 gif 4k 화질로 올려줘")
   평가 화면(256×256)을 키우면 흐릿하다. 시뮬을 3840×2160 으로 따로 그린다 (모델 입력은 그대로 256).
   작업대를 꽉 채우는 자유 시점 카메라 + 오른쪽 위에 손목 카메라(720×720 으로 따로 그림) + 아래에 한국어 명령·해석.
   그리는 방향은 실제 그대로 (위아래만 뒤집음 — 평가 화면은 LIBERO 처럼 180° 돌려 보여 좌우가 거울처럼 바뀌어 있다)."""
import numpy as np

W, H = 3840, 2160
CAM = dict(azimuth=165.0, elevation=-38.0, distance=0.96, lookat=(-0.05, 0.0, 1.0))   # 10/24: 앞쪽 판(회색)이 글자 칸에 안 가리게 더 내려다봄
FONT_B = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
FONT_R = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"


def render(ep, cam=CAM):
    import mujoco
    sim = ep.inner.sim
    sim.render(width=64, height=64, camera_name="agentview")          # 그리기 문맥이 없으면 만든다
    ctx = sim._render_context_offscreen
    ctx.render(W, H, camera_id=-1)                                     # 자유 시점 (-1) 으로 크기 맞춤
    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ctx.cam.lookat[:] = cam["lookat"]
    ctx.cam.azimuth, ctx.cam.elevation, ctx.cam.distance = cam["azimuth"], cam["elevation"], cam["distance"]
    ctx.render(W, H, camera_id=-1)
    img = ctx.read_pixels(W, H, depth=False)[::-1]
    wrist = sim.render(width=720, height=720, camera_name="robot0_eye_in_hand")[::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(wrist)


def frame(ep, lines, status, title="", cam=CAM):
    """4K 한 장: 장면 + 손목 화면 + 아래 글 (lines: [(글, (r,g,b))], 마지막 줄을 크게)."""
    from PIL import Image, ImageDraw, ImageFont
    img, wrist = render(ep, cam)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    # 손목 화면 (오른쪽 위)
    x0, y0 = W - 720 - 48, 48
    d.rectangle([x0 - 8, y0 - 8, x0 + 728, y0 + 728 + 64], fill=(252, 252, 251, 235))
    im.paste(Image.fromarray(wrist), (x0, y0))
    d.text((x0 + 4, y0 + 728), "손목 카메라", font=ImageFont.truetype(FONT_R, 44), fill=(82, 81, 78))
    # 제목 (왼쪽 위)
    if title:
        f = ImageFont.truetype(FONT_B, 64)
        tw = d.textlength(title, font=f)
        d.rectangle([40, 40, 40 + tw + 48, 40 + 100], fill=(252, 252, 251, 235))
        d.text((64, 56), title, font=f, fill=(11, 11, 11))
    # 아래 글
    sel = lines[-4:]
    if sel or status:
        top = H - 80 - 86 * len(sel) - 40
        d.rectangle([0, top, W, H], fill=(252, 252, 251, 235))
        y = top + 24
        for i, (txt, col) in enumerate(sel):
            big = i == len(sel) - 1
            d.text((64, y), txt, font=ImageFont.truetype(FONT_B if big else FONT_R, 68 if big else 56), fill=col)
            y += 86
        d.text((64, H - 76), status, font=ImageFont.truetype(FONT_R, 48), fill=(82, 81, 78))
    return np.asarray(im)


def save_gif(frames_path, gif_path, every=1, colors=128, fps=10, limit_mb=48):
    """4K mp4 → 4K GIF. 색 줄이기는 떨림 없이 (나뭇결이 떨리면 장마다 달라져 파일이 커진다). 48MB 넘으면 장 수를 줄여 다시
    (GitHub 은 50MB 넘으면 경고, 100MB 넘으면 거부 — 첫 GIF 70MB 에서 경고)."""
    import os
    import imageio.v3 as iio
    from PIL import Image
    while True:
        pal, first = [], None
        for i, f in enumerate(iio.imiter(frames_path)):
            if i % every:
                continue
            im = Image.fromarray(f)
            if first is None:
                first = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                pal.append(first)
            else:
                pal.append(im.quantize(palette=first, dither=Image.Dither.NONE))
        pal += [pal[-1]] * fps                                          # 마지막 모습 1초
        pal[0].save(gif_path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=False)
        mb = os.path.getsize(gif_path) / 1e6
        if mb <= limit_mb or every >= 8:
            return mb, len(pal), every
        every *= 2
