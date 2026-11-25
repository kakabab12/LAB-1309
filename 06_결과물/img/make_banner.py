"""저장소 첫 화면(README) 배너를 만든다 (밝은 화면용, 어두운 화면용 두 벌). 외부 서비스 없이 저장소 안 SVG 로.

    python make_banner.py .      → banner_dark.svg, banner_light.svg
"""
import sys

W, H = 1280, 300

def banner(dark):
    if dark:
        bg0, bg1, ink, sub, mute, line, acc, acc2, grid, paper = "#0d1117", "#132238", "#f0f6fc", "#c9d1d9", "#8b949e", "#9fb3c8", "#4fa3ff", "#ffb454", "#1f2a3a", "#1b2636"
    else:
        bg0, bg1, ink, sub, mute, line, acc, acc2, grid, paper = "#f6f8fa", "#e6eef7", "#1f2328", "#30363d", "#57606a", "#3d5a78", "#1f6feb", "#d97706", "#dde5ee", "#ffffff"
    f_ko = "'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR','Noto Sans CJK KR',sans-serif"
    f_en = "'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif"
    f_mono = "'JetBrains Mono','SFMono-Regular',Consolas,'Liberation Mono',monospace"
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="1.2"/>' for x in range(20, W, 28) for y in range(18, H, 28))
    # PiPER 비슷한 6축 팔 (선 그림): 밑동 → 어깨 → 팔꿈치 → 손목 → 집게
    base_x, base_y = 1010, 262
    sh = (1010, 196); el = (1092, 116); wr = (1176, 146); tip = (1176, 206)
    arm = f'''
    <g fill="none" stroke="{line}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M{base_x-34},{base_y} h68 v-14 h-68 z" fill="{bg1}"/>
      <path d="M{base_x},{base_y-14} L{sh[0]},{sh[1]}"/>
      <path d="M{sh[0]},{sh[1]} L{el[0]},{el[1]}"/>
      <path d="M{el[0]},{el[1]} L{wr[0]},{wr[1]}"/>
      <path d="M{wr[0]},{wr[1]} L{tip[0]},{tip[1]-22}"/>
      <path d="M{tip[0]-13},{tip[1]-22} h26 M{tip[0]-11},{tip[1]-22} v20 M{tip[0]+11},{tip[1]-22} v20" stroke-width="4"/>
    </g>
    <rect x="{tip[0]-9}" y="{tip[1]-18}" width="18" height="18" rx="2" fill="{acc2}"/>
    <g fill="{bg0}" stroke="{acc}" stroke-width="3.5">
      <circle cx="{sh[0]}" cy="{sh[1]}" r="9"/><circle cx="{el[0]}" cy="{el[1]}" r="8"/><circle cx="{wr[0]}" cy="{wr[1]}" r="7"/>
    </g>'''
    # 책상 위 종이와 + 표시, 물체
    desk = f'''
    <path d="M900,262 L1250,262" stroke="{mute}" stroke-width="2" opacity="0.6"/>
    <path d="M1128,250 l90,0 l14,10 l-90,0 z" fill="{paper}" stroke="{mute}" stroke-width="1.5"/>
    <path d="M1176,251 v8 M1169,255 h14" stroke="{ink}" stroke-width="2"/>
    <rect x="1050" y="236" width="20" height="20" rx="2" fill="{mute}" opacity="0.55"/>'''
    # 위 카메라와 시야 (점선이 흐름)
    cam = f'''
    <g>
      <rect x="884" y="44" width="38" height="20" rx="4" fill="{bg1}" stroke="{line}" stroke-width="3"/>
      <circle cx="903" cy="54" r="5" fill="{acc}"/>
      <path d="M903,64 L1140,248 M903,64 L1250,250" stroke="{acc}" stroke-width="1.8" stroke-dasharray="6 7" opacity="0.75">
        <animate attributeName="stroke-dashoffset" from="26" to="0" dur="1.6s" repeatCount="indefinite"/>
      </path>
    </g>'''
    # 말풍선: 일하는 도중 새 명령
    bubble = f'''
    <g font-family="{f_en}" font-size="15" font-weight="600">
      <rect x="1056" y="22" width="196" height="34" rx="17" fill="{bg1}" stroke="{acc2}" stroke-width="2"/>
      <text x="1154" y="44" text-anchor="middle" fill="{ink}">"Now do B, please"</text>
      <path d="M1100,56 l-8,14 l20,-14" fill="{bg1}" stroke="{acc2}" stroke-width="2" stroke-linejoin="round"/>
      <path d="M1093,55 h22" stroke="{bg1}" stroke-width="3"/>
    </g>'''
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="LAB-1309: 일하는 도중 새 명령을 받아도 이어서 해내는 로봇팔">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{bg0}"/><stop offset="1" stop-color="{bg1}"/>
    </linearGradient>
    <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{bg0}" stop-opacity="1"/><stop offset="0.55" stop-color="{bg0}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="{W}" height="{H}" rx="16" fill="url(#bg)"/>
  <g fill="{grid}">{dots}</g>
  <rect width="{W}" height="{H}" rx="16" fill="url(#fade)"/>
  {desk}{cam}{arm}{bubble}
  <g>
    <text x="64" y="78" font-family="{f_mono}" font-size="17" letter-spacing="3" fill="{acc}">KEIMYUNG UNIV. · ROBOT LEARNING</text>
    <text x="60" y="158" font-family="{f_en}" font-size="76" font-weight="800" fill="{ink}" letter-spacing="-1">LAB-1309</text>
    <text x="64" y="206" font-family="{f_ko}" font-size="27" font-weight="700" fill="{sub}">일하는 도중 새 명령을 받아도 이어서 해내는 로봇팔</text>
    <text x="64" y="244" font-family="{f_en}" font-size="17" fill="{mute}">Vision-Language-Action · SmolVLA · AgileX PiPER · Digital Twin → Real Robot</text>
  </g>
</svg>
'''
    return svg

out = sys.argv[1]
for name, dark in (("banner_dark.svg", True), ("banner_light.svg", False)):
    open(f"{out}/{name}", "w", encoding="utf-8").write(banner(dark))
print("ok")
