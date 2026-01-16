# 2024 국민환경의식 설문조사 분석 보고서

## 📋 프로젝트 개요

본 프로젝트는 2024년 국민환경의식 설문조사 데이터를 분석하여 **환경의식-태도-실천 괴리**를 규명하고, 인구통계학적 특성에 따른 환경행동 결정요인과 방해요인을 심층 분석합니다.

## 🎯 연구 목적

- **핵심 질문**: "환경의식(Attitude/PN)은 높은데 왜 실천(Behavior)으로 이어지지 않는가?"
- **가설**: 집단별(소득, 성별, 연령)로 행동을 가로막는 방해 요인(Barrier)과 행동을 이끄는 촉진 요인(Facilitator)이 다를 것이다

## 📊 주요 분석 내용

### 1. 이론적 프레임워크
- **TPB (Theory of Planned Behavior)**: Attitude, SN, PBC → Behavior
- **Personal Norm (PN)**: 환경 개인규범의 역할
- **통합 모델**: TPB + PN

### 2. 분석 기법
- 신뢰도 분석 (Cronbach's α)
- 다중회귀분석 (H1, H2, H3)
- ANOVA / T-test (인구통계 차이)
- Gap 분석 (Min-Max 정규화)
- 조절효과 분석
- 9사분면 프로파일링
- 인구통계별 회귀분석 (촉진/방해 요인)

### 3. 핵심 발견
- PN의 직접 효과 **매우 약함** (β=0.049) → 의식-행동 괴리 존재 증명
- 소득별 방해요인: 저소득층 - Attitude/SN 점수 낮음
- 성별 차이: 남성 - Attitude 중심, 여성 - SN 중심
- 연령별 차이: 60대+ - 사회적 규범(SN)에 가장 민감

## 📁 파일 구조

```
eco/
├── index.html              # 웹 대시보드 (메인)
├── script.js               # 시각화 로직
├── style.css               # 스타일시트
├── generate_full_report.py # JSON 생성 스크립트
├── full_report_data.json   # 분석 결과 데이터
├── main.py                 # 기본 분석 스크립트
└── 2024data/               # 원본 데이터 (Excel, SPSS)
```

## 🚀 사용 방법

### 1. 데이터 분석 실행
```bash
python generate_full_report.py
```

### 2. 웹 대시보드 확인
`index.html` 파일을 브라우저에서 열기

### 3. 온라인 버전
GitHub Pages: [https://wookidoki.github.io/Survey-on-Environmental-Ethics-in-Korea-2024-/](https://wookidoki.github.io/Survey-on-Environmental-Ethics-in-Korea-2024-/)

## 📊 데이터 품질

- **표본 크기**: 3,040명
- **분석 표본**: 3,011명 (결측률 1.0%)
- **신뢰도**: 모든 척도 α ≥ 0.70
- **모델 적합도**: R² = 0.247 (사회과학 적정 수준)

## 🔍 데이터 투명성

본 연구는 데이터 조작(Cherry Picking) 의혹을 차단하기 위해:
- 14개 전체 행동 문항의 평균 점수 공개
- 천장 효과(Ceiling Effect) 분석 결과 공개
- 신뢰도 유지를 위해 모든 문항 분석에 포함

## 📈 시각화

- Chart.js: 막대 그래프, 라인 차트
- Plotly.js: 3D 산점도
- 반응형 웹 디자인

## 🛠️ 기술 스택

- **Python**: pandas, numpy, scipy, statsmodels
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **Libraries**: Chart.js, Plotly.js, MathJax

## 📄 라이선스

본 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 👤 Author

이성욱 (Lee Seongwook)

---

**Note**: 원본 데이터는 2024 국민환경의식조사 공공데이터를 기반으로 합니다.
