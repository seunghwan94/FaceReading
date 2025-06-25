**Facial Feature-Based Physiognomy Analysis**

이 프로젝트는 `face_recognition` 라이브러리를 통해 얼굴 랜드마크를 추출하고, 다양한 거리·비율·각도·면적·대칭성 지표를 계산하여 JSON 형태로 변환한 뒤, GPT API에 전달하여 관상 분석 결과를 받아오는 파이프라인을 제공합니다.

---

## 🚀 목차

1. [사전 준비](#사전-준비)
2. [설치 및 환경 구성](#설치-및-환경-구성)
3. [랜드마크 추출 및 시각화](#랜드마크-추출-및-시각화)
4. [메트릭 계산](#메트릭-계산)
5. [JSON 포맷 생성](#json-포맷-생성)
6. [GPT 프롬프트 템플릿](#gpt-프롬프트-템플릿)
7. [실행 예시](#실행-예시)
8. [주의사항 및 팁](#주의사항-및-팁)

---

## 1. 사전 준비

* Python 3.7 이상
* 가상환경(venv) 권장
* 설치 불가능 환경에서는 Mediapipe 또는 AWS Rekognition 대안 고려

---

## 2. 설치 및 환경 구성

> dlib 빌드가 필요하므로 CMake 및 Visual C++ Build Tools(Windows) 설치 필수

[CMake 공식 사이트](https://cmake.org/download/)에서 설치

```bash
# Binary distributions : 환경에 맞게 설치 후 찍어보기
cmake --version
```

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치
pip install --upgrade pip
pip install face_recognition pillow
```



---

## 3. 랜드마크 추출 및 시각화

```python
import face_recognition
from PIL import Image, ImageDraw

# 1) 이미지 로드
image = face_recognition.load_image_file("test.jpg")

# 2) 랜드마크 추출
landmarks_list = face_recognition.face_landmarks(image)

# 3) 시각화
pil_img = Image.fromarray(image)
draw = ImageDraw.Draw(pil_img)
for lm in landmarks_list:
    for feature in ["left_eye","right_eye","nose_bridge","nose_tip","top_lip","bottom_lip","chin"]:
        pts = lm[feature]
        draw.line(pts + [pts[0]], width=2)
pil_img.show()
```

---

## 4. 메트릭 계산

다음 지표를 계산합니다:

* **거리(Distances)**: 눈 사이 거리, 코 길이, 입 너비 등
* **비율(Ratios)**: 얼굴 폭/높이 대비 비율
* **각도(Angles)**: 눈 축, 턱 선 기울기 등
* **면적(Areas)**: 눈·입술 영역 면적
* **대칭성(Symmetry)**: 좌우 대칭 오차
* **고급 지표(Advanced)**: 골든비율 적합도, T존 지수, 얼굴 구역 비율

상세 코드는 `metrics_extraction.py`에 구현되어 있습니다.

---

## 5. JSON 포맷 생성

```json
{
  "face_size": { "width": 350, "height": 450 },
  "metrics": {
    "eye_distance": 0.25,
    "mouth_width": 0.40,
    "jaw_angle": -3.5,
    ...
  }
}
```

* 정규화된 비율(0\~1) 및 각도(°)로 구성
* GPT에 전달하기 용이한 스키마

---

## 6. GPT 프롬프트 템플릿

```
당신은 관상 전문가입니다.
아래 얼굴 비율 정보를 참고하여 성격·강점·약점을 2~3문장으로 설명해 주세요.

<FACE_DATA>
{
  "metrics": {
    "eye_distance": 0.25,
    "mouth_width": 0.40,
    "jaw_angle": -3.5,
    ...
  }
}
</FACE_DATA>
```

---

## 7. 실행 예시

```bash
python extract_and_analyze.py --image test.jpg --output result.json
```

* `extract_and_analyze.py` 는 랜드마크 추출, 메트릭 계산, JSON 출력
* 이후 GPT API 호출 스크립트와 연동

---

## 8. 주의사항 및 팁

* **정면 사진 권장**: 비스듬한 각도는 왜곡 발생
* **프라이버시**: 민감 데이터이므로 HTTPS/암호화 적용
* **면책 조항**: 오락용 분석임을 명시
* **대체 솔루션**: 설치 불가 시 Mediapipe 또는 AWS Rekognition 사용

---

위 README를 참고하여 프로젝트를 설정하고, 얼굴 좌표 기반으로 GPT 관상 분석 파이프라인을 구축해 보세요!
