# 2024 국민환경의식 설문조사 분석

## 프로젝트 개요

2024년 국민환경의식 설문조사 데이터를 분석하여 환경의식-태도-실천 괴리를 규명하고, 인구통계학적 특성에 따른 환경행동 결정요인과 방해요인을 분석한다.

## 연구 목적

- 핵심 질문: "환경의식은 높은데 왜 실천으로 이어지지 않는가?"
- 가설: 집단별(소득, 성별, 연령)로 행동을 가로막는 방해 요인과 촉진 요인이 다를 것이다

## 분석 내용

### 이론적 프레임워크
- TPB (Theory of Planned Behavior): Attitude, SN, PBC -> Behavior
- Personal Norm (PN): 환경 개인규범의 역할
- 통합 모델: TPB + PN

### 분석 기법
- 신뢰도 분석 (Cronbach's Alpha)
- 다중회귀분석
- ANOVA / T-test
- Gap 분석 (Min-Max 정규화)
- 조절효과 분석
- 9사분면 프로파일링

### 핵심 발견
- PN의 직접 효과 매우 약함 (Beta=0.049) -> 의식-행동 괴리 존재
- 소득별 방해요인: 저소득층 - Attitude/SN 점수 낮음
- 성별 차이: 남성 - Attitude 중심, 여성 - SN 중심
- 연령별 차이: 60대+ - 사회적 규범(SN)에 가장 민감

## 파일 구조

```
├── index.html              # 웹 대시보드
├── script.js               # 시각화 로직
├── style.css               # 스타일시트
├── generate_full_report.py # JSON 생성 스크립트
├── full_report_data.json   # 분석 결과 데이터
└── main.py                 # 기본 분석 스크립트
```

## 사용 방법

### 데이터 분석 실행
```bash
python generate_full_report.py
```

### 웹 대시보드 확인
`index.html` 파일을 브라우저에서 열기

### 온라인 버전
GitHub Pages: https://wookidoki.github.io/Survey-on-Environmental-Ethics-in-Korea-2024-/

## 데이터 품질

- 표본 크기: 3,040명
- 분석 표본: 3,011명 (결측률 1.0%)
- 신뢰도: 모든 척도 Alpha >= 0.70
- 모델 적합도: R² = 0.247

## 기술 스택

- Python: pandas, numpy, scipy, statsmodels
- Frontend: HTML5, CSS3, JavaScript
- Libraries: Chart.js, Plotly.js, MathJax

## Author

이성욱 (Lee Seongwook)
