# ClaudeYe: Claude Code용 비전 미들웨어

Claude Code의 약한 시각 능력을 **도구로 보강하는 로컬 MCP 서버**입니다.
Claude는 "두뇌·지휘자"로 남습니다. ClaudeYe는 확대·측정·검출·OCR·영상 분해를 대신 해 주는 "눈과 손"입니다.

- **이미지**: 원본 해상도 줌, 격자·원본좌표 눈금자, 정확한 개수 세기·색상·비교·검출·분할·OCR
- **영상**: Claude는 원래 영상을 못 봅니다. 영상을 타임스탬프 스토리보드, 변화 이벤트, 음성 전사, 텍스트 검색으로 바꿔 줍니다.
- **2차 의견**: 로컬 VLM(Ollama/Qwen)과 Gemini(선택, 클라우드). 영상+오디오를 네이티브로 봅니다.

---

## 1. 왜 이 방식인가 (근거)

| 한계 | 사실 |
|---|---|
| 해상도 | Opus 4.7 이상도 긴 변 **2576px**, 이미지당 **≈4784 토큰**(28×28 패치당 1토큰)이 상한입니다. 4K 스크린샷은 축소되고, 한 패치보다 작은 디테일은 사라집니다. Anthropic cookbook 예시에서는 8px 간격을 확대 없이 판독하게 했을 때 **40번 중 36번 틀렸습니다**. |
| 영상 | 네이티브 영상 입력이 **없습니다**. |
| 개수·좌표 | Opus 4.7의 물체 세기 정확도는 30%(Roboflow)입니다. 좌표는 "근사치"라고 공식 문서에 명시되어 있습니다. |
| **공식 권고** | Anthropic 마이그레이션 가이드(Opus 5): *"Vision pipelines: give it crop/analyze/verify tools - cheaper and more effective than raising thinking"*. Fable 5.1은 "흐리거나 뒤집힌 입력에서 bash·crop 도구를 쓰도록 훈련"되었습니다. |

도구 보강의 효과로 보고된 수치는 다음과 같습니다.
- 크롭 도구: Chartography(밀집 차트)에서 **Fable 5가 29→73%**, **Sonnet 5가 13→44%**(Anthropic cookbook)
- CharXiv-R: Opus 4.7이 82.1→91.0%
- 반복 크롭(GUI-Lens, 프리프린트): ScreenSpot-Pro에서 **Opus 4.7이 57.4→82.3%**
- Set-of-Mark 번호 마커: GPT-4V RefCOCOg 25.7→86.4%
- 코드 도구(PyVision): Claude-4-Sonnet이 VLMsAreBlind-mini에서 **+31.1점**
- 영상: 에이전트형 탐색(Deep Video Discovery)이 LVBench 74.2%(전사 포함 76.0%)를 기록했고, 자막만 추가해도 Claude 3.5 Sonnet의 Video-MME 점수가 +2.9점 올랐습니다.

즉 **능동적 지각**(보고, 확대하고, 측정하고, 검증하는 루프)을 도구로 제공하는 것이 지금 가장 확실한 방법입니다.

## 2. 구조

```
Claude Code ──(MCP stdio)──► ClaudeYe 서버
  │  Skill: 인지 프로토콜(언제 무엇을 쓸지)         ├─ L0 결정적 도구 (모델 없음, 즉시)
  │  Subagent: vision-analyst(무거운 영상 분석)     │    look · zoom · scan · pixels · blobs · annotate · compare · locate
  │                                                │    video_info · video_storyboard · video_frames · video_changes
  └─ 서버 instructions(프로토콜 요약) 자동 주입      ├─ L1 전문 모델 (로컬, 지연 로딩, VRAM 관리)
                                                   │    ocr(RapidOCR 한국어) · transcribe(Whisper) · detect(LLMDet)
                                                   │    segment(SAM 2.1/3) · depth(Depth Anything V2) · video_search(SigLIP 2)
                                                   └─ L2 2차 의견 VLM
                                                        ask_vision · ask_video (Ollama Qwen3.5 / Gemini 옵트인)
```

설계 원칙은 다음과 같습니다.
- **모든 좌표는 원본 픽셀 기준**입니다. 모든 뷰에 원본 좌표 눈금자가 붙습니다.
- 영상의 모든 프레임은 `영상.mp4@12.5` 형태의 이미지 참조로 모든 이미지 도구에서 쓸 수 있습니다.
- 반환 이미지는 긴 변 1568px(상한 2000px)입니다. 대화에 이미지가 20장을 넘으면 API가 장당 2000px 이하를 요구하기 때문입니다. 각 결과에 예상 토큰 수를 표시합니다.
- 결과는 텍스트(JSON)와 번호 마커 오버레이 이미지로 나옵니다. Claude가 모델 결과를 **눈으로 검증**하게 하려는 것입니다.
- 무거운 라이브러리는 도구 호출 시에 불러옵니다. 서버 기동은 약 1.3초입니다.

## 3. 도구 (21개)

| 도구 | 용도 |
|---|---|
| `look` | 격자(A1..)와 원본 좌표 눈금자가 붙은 개요. 영상을 넣으면 스토리보드를 반환 |
| `zoom` | 영역(셀 `C3`, `B2:D4`, `x0,y0,x1,y1`, 비율, 점)을 원본 해상도로 확대(최대 6배). 선명화·CLAHE·반전·회전 등 `enhance` 지원 |
| `scan` | 큰 이미지를 원본 해상도 타일로 빠짐없이 훑기(페이지 단위) |
| `pixels` | 지점의 정확한 RGB/HEX, 영역 평균과 주요 색 |
| `blobs` | 색(또는 잉크) 연결영역으로 **정확한 개수와 위치** 산출. 마커, 점, 뱃지, 아이콘 |
| `annotate` | 박스·점·화살표를 그려 자기 위치 판단을 검증 |
| `compare` | 두 이미지·프레임의 픽셀 diff(변경 영역 번호, PSNR) |
| `locate` | 템플릿(아이콘·버튼) 다중 스케일 매칭 |
| `ocr` | 한국어+영어 줄 단위 OCR과 박스. `query`로 "이 글자가 어디 있나" 검색 |
| `detect` | 텍스트로 지정하는 물체 검출과 개수(LLMDet, 큰 이미지 타일 분할) |
| `segment` | SAM 2.1(점·박스) / SAM 3(텍스트) 마스크, 폴리곤, 면적 % |
| `depth` | 상대 깊이 맵과 셀별 깊이 표 |
| `video_info` | 길이, fps, 해상도, 회전, 오디오 |
| `video_storyboard` | 샷마다 1장 + 변화 직후 프레임으로 만든 **타임스탬프 콘택트시트**(분석 결과 캐시) |
| `video_frames` | 구간을 촘촘하게(개수 또는 fps), 시트 또는 개별 이미지로 |
| `video_changes` | 화면 녹화의 **변화 이벤트 타임라인**(언제·어디서·얼마나, 전후 참조) |
| `video_search` | "설명"으로 영상 속 순간 찾기(SigLIP 2 임베딩, 캐시) |
| `transcribe` | 음성 전사(Whisper, 단어 정렬로 정확한 시각) |
| `ask_vision` / `ask_video` | 다른 비전 모델의 2차 의견(로컬 Ollama 우선, Gemini는 옵트인) |
| `vision_status` | 설치된 백엔드, 장치, VRAM, 설치 안내 |

## 4. 이 PC(노트북, CPU)에서 검증한 결과

정답을 아는 합성 데이터(`scripts/make_samples.py`)로 측정했습니다.

| 과제 | 결과 |
|---|---|
| 4K 스크린샷 속 9px 한글 읽기 | `look`(0.39배 축소)에서는 판독 불가 → `zoom`(2배)으로 판독 성공. OCR은 "응답→용답" 1자를 틀림 → **OCR과 zoom 교차 검증**이 필요 |
| 빨간 점 23개와 파란 사각형 14개 세기 | `blobs`: **23 / 14 정확**(0.3초). LLMDet-tiny(CPU): 5 / 8, 타일 분할 시 28 / 28로 과다 집계 → 평면 도형은 결정적 도구로 |
| 영상 장면 전환 | **5.0 / 10.0 / 18.0초**로 정답과 일치 |
| 화면 녹화 UI 이벤트 | 대화상자 열림(11.0초, 박스 오차 5px 이내), 타이핑(13.5~14.5초), 버튼 색·오류 토스트(16.0초) 검출 |
| 한국어 음성 전사 | whisper-small(CPU): 문장 정확, 시작 시각 오차 ≤0.14초 |
| 영상 텍스트 검색 | "settings dialog" → 12~17초(정답 구간 11~18초) |
| 아이콘 찾기, 이미지 diff, SAM 2.1 | 좌표 정확, 변경 2곳 검출, 버튼 마스크 박스 오차 1px |
| 테스트 | `pytest` 11/11 통과, MCP 도구 21개 E2E 통과, 서버 기동 1.3초 |

아직 검증하지 못한 항목도 있습니다.
- 로컬 VLM 실제 추론: 이 PC의 Ollama 0.34는 `llama3.2-vision`(mllama)을 로드하지 못합니다. 개발 PC에서 `qwen3.5:9b`로 확인해야 합니다.
- Gemini 실제 호출: 키가 없어서 요청 구성까지만 오프라인 테스트로 검증했습니다.
- SAM 3: 게이트 모델이라 검증하지 못했습니다.

## 5. 설치 (개발 PC: RTX 4070 12GB, i7-13700, RAM 16GB)

사전 준비:
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- NVIDIA 드라이버 **≥580**(cu130) 또는 525~579(cu126)
- (선택) [Ollama](https://ollama.com)

```powershell
git clone https://github.com/SunWooBang/ClaudeYe.git
cd ClaudeYe
uv sync --extra ocr --extra asr --extra asr-cuda --extra ml --extra cu130 --extra gemini
ollama pull qwen3.5:9b
uv run python scripts/make_samples.py
uv run python scripts/make_speech_sample.py
uv run pytest -q
```

- 드라이버가 525~579 버전이면 `--extra cu130` 대신 `--extra cu126`을 씁니다.
- GPU가 없는 PC에서는 `--extra ocr --extra asr --extra ml --extra cpu`로 설치합니다.

### Claude Code에 등록

- **이 폴더에서만 쓰기**: `.mcp.json`이 있어서 이 폴더에서 `claude`를 실행하면 자동으로 인식합니다(최초 1회 승인). Skill과 서브에이전트도 `.claude/` 아래에 들어 있습니다.
- **모든 프로젝트에서 쓰기**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\install_claude.ps1 -Apply
  ```
  사용자 범위 MCP 등록과 Skill·서브에이전트 복사가 이루어집니다. 인자 없이 실행하면 미리보기만 보여 줍니다. Gemini까지 쓰려면 `-AllowCloud`를 붙이고 `GEMINI_API_KEY`를 설정합니다.
- 확인: Claude Code에서 `/mcp`로 `claudeye`가 연결됐는지 보고, "vision_status 실행해줘"라고 요청합니다.

### 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CLAUDEYE_MAX_EDGE` | 1568 | 반환 이미지의 긴 변 |
| `CLAUDEYE_HARD_MAX_EDGE` | 2000 | 반환 이미지 상한 |
| `CLAUDEYE_DEVICE` | auto | `cpu`로 설정하면 GPU를 쓰지 않음 |
| `CLAUDEYE_ALLOW_CLOUD` | 0 | 1일 때만 Gemini 사용(미디어가 Google로 전송됨) |
| `GEMINI_API_KEY` | - | Gemini 키 |
| `CLAUDEYE_GEMINI_MODEL` | gemini-flash-latest | 예: gemini-3.8-flash |
| `CLAUDEYE_OLLAMA_MODEL` | qwen3.5:9b | 로컬 2차 의견 모델(없으면 설치된 비전 모델 자동 선택) |
| `CLAUDEYE_DETECT_MODEL` | GPU: llmdet_large / CPU: llmdet_tiny | 검출 모델 |
| `CLAUDEYE_SEGMENT_MODEL` | - | `facebook/sam3`로 설정하면 텍스트 분할에 SAM 3 사용(HF 약관 동의 + `hf auth login` 필요) |
| `CLAUDEYE_WHISPER_MODEL` | auto | GPU: large-v3-turbo / CPU: small |
| `CLAUDEYE_CACHE_DIR` | ~/.claudeye/cache | 영상 분석, 전사, 임베딩 캐시 |

### 12GB VRAM 운용

- 상주: OCR(ONNX, CPU 동작도 충분)
- 필요할 때 로드: 검출(~1.5~3GB), SAM 2.1 large(~1GB), Whisper turbo int8(~2GB), SigLIP 2
- 무거운 슬롯 1개: Qwen3.5-9B Q4(~7GB)
- ClaudeYe는 GPU 락과 LRU 언로드를 쓰고, torch 모델을 올리기 전에 Ollama 모델을 내립니다(`keep_alive:0`).

## 6. 사용 예

일반 프롬프트로 요청하면 Claude가 Skill에 따라 도구를 고릅니다.

- "`D:\shots\error.png`에서 에러 코드랑 작은 회색 글씨 전부 정확히 읽어줘"
- "이 차트(`chart.png`) 3분기 막대의 값을 눈금 기준으로 읽어줘"
- "`bug.mp4` 화면 녹화 보고 재현 절차를 타임스탬프와 함께 정리해줘"
- "`meeting.mp4`에서 누가 화이트보드에 그림 그리는 장면 찾아서 무슨 내용인지 읽어줘"
- "`before.png`와 `after.png`의 UI 차이를 전부 찾아줘"
- "이 사진에 사람이 몇 명인지 세고 근거 오버레이 보여줘"
- 긴 영상은 "vision-analyst 서브에이전트로 분석해줘"라고 요청하면 이미지 토큰이 본 대화를 채우지 않습니다.

## 7. 한계와 주의

- **첫 실행 시 모델을 다운로드합니다**: OCR 20MB, Whisper small 0.5GB / large-v3-turbo 1.6GB, LLMDet tiny 0.7GB / large 1.4GB, SAM 2.1 tiny 0.16GB / large 0.9GB, SigLIP 2 base 1.5GB, DA-V2 small 0.1GB.
- **오픈 어휘 검출은 작은 합성 도형에 약합니다**. 평면 색 도형은 `blobs`를 쓰고, 검출 결과는 반드시 오버레이로 확인합니다.
- **OCR도 틀립니다**. 중요한 문자열은 `zoom`으로 확인합니다. Skill에 이 규칙이 들어 있습니다.
- **Gemini는 명시적 옵트인**입니다. 무료 티어 데이터는 Google이 제품 개선에 쓸 수 있으므로, 민감한 영상은 로컬 경로를 쓰세요.
- 긴 영상의 첫 분석은 전체 디코드를 거치므로 시간이 걸립니다(캐시됨). 36,000프레임을 넘으면 키프레임만 빠르게 분석합니다. 도구 호출이 시간 초과되면 `MCP_TOOL_TIMEOUT`을 늘리세요.
- 화면 녹화에서는 마우스 커서 이동도 작은 변화로 잡힙니다. `sensitivity`와 `region`으로 조절합니다.

## 8. 로드맵

1. **효과 측정 하네스**: headless `claude -p`로 ClaudeYe 유무를 A/B 비교(차트, UI 그라운딩, 세기, 영상 QA 샘플)
2. UI 요소 파서(OmniParser V2) 통합으로 스크린샷을 번호 붙은 UI 요소 목록으로 변환
3. `screen@` / `clipboard@` 참조(현재 화면·클립보드 직접 분석)
4. SAM 3.1 영상 추적(객체별 등장·이동 타임라인), 커서 마스킹
5. 샷별 자동 캡션 인덱스(Deep Video Discovery 방식의 장편 영상 탐색)
6. Claude Code 플러그인(마켓플레이스)으로 패키징

## 개발

```
src/claudeye/
  server.py          MCP 서버, L0 도구 등록, 오류 전달
  models_tools.py    L1/L2 도구
  image_tools.py     look/zoom/scan/pixels/blobs/annotate/compare/locate
  video_tools.py     video_* (스토리보드 선택, 변화 타임라인)
  analyze.py         1패스 영상 분석(샷 경계, 변화, 캐시)
  video.py           PyAV 디코딩(정확한 탐색, 회전 보정)
  render.py          격자, 눈금자, Set-of-Mark, 콘택트시트
  backends/          ocr, asr, detect, segment, depth, embed, vlm, device(VRAM 관리)
scripts/  make_samples.py, make_speech_sample.py, try_tool.py(직접 호출), mcp_smoke.py(E2E), install_claude.ps1
tests/    test_claudeye.py
```

`uv run python scripts/try_tool.py zoom samples/dashboard_4k.png A1`처럼 도구를 직접 호출하면 결과 이미지가 `out/`에 저장됩니다.

## 라이선스

[MIT](LICENSE). 백엔드 모델은 각자의 라이선스를 따릅니다. 기본 모델(RapidOCR/PP-OCR, Whisper, LLMDet, SAM 2.1, Depth Anything V2 Small, SigLIP 2)은 Apache-2.0 또는 MIT입니다. SAM 3는 별도의 SAM License(게이트)입니다.
