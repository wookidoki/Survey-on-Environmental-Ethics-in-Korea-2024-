# -*- coding: utf-8 -*-
# =============================================================================
# 2024 국민환경의식 설문조사 - 최종 수정 및 검증 완료 버전
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# [시각화 설정] 한글 폰트 (Windows: Malgun Gothic, Mac: AppleGothic)
# -----------------------------------------------------------------------------
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print(" 2024 국민환경의식 설문조사 분석 (최종 수정본)")
print("="*80)

# =============================================================================
# 1. 데이터 로드 및 전처리
# =============================================================================

def cronbach_alpha(items_df):
    # 결측치가 있는 행은 제외하고 계산
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2: return np.nan
    sum_item_variances = items_df.var(axis=0, ddof=1).sum()
    total_score_variance = items_df.sum(axis=1).var(ddof=1)
    if total_score_variance == 0: return 1.0
    return (k / (k - 1)) * (1 - (sum_item_variances / total_score_variance))

# 파일 경로 (사용자 환경에 맞게 확인 필요)
original_excel_file = '2024data/2024 국민환경의식 설문조사_원자료.xlsx'

try:
    df_raw = pd.read_excel(original_excel_file, sheet_name='원자료')
    print(f"\n[데이터 로드] 성공: 총 {len(df_raw)}명")
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    raise

df_analysis = df_raw.copy()

# -----------------------------------------------------------------------------
# [수정 포인트 1] 무응답(9) 처리 및 역코딩 로직 수정
# 기존 코드의 replace(9, 2)는 데이터를 왜곡하므로 replace(9, np.nan)으로 변경합니다.
# -----------------------------------------------------------------------------

# 1. Behavior (행동)
b4_cols = ['B4_1', 'B4_2', 'B4_3', 'B4_4', 'B4_5', 'B4_6', 'B4_7', 'B4_8',
           'B4_11', 'B4_12', 'B4_13', 'B4_14']

# 9(무응답)를 NaN으로 처리하여 평균 계산 시 제외
b4_data = df_analysis[b4_cols].replace(9, np.nan)

# 역코딩: 4점 척도(1:항상~4:전혀안함)로 가정 시, (5 - 점수)가 표준입니다.
# 만약 원본이 1:전혀안함~4:항상 이라면 역코딩 불필요.
# 기존 코드 로직(3-data)은 음수가 나올 수 있어 (5-data)로 수정 제안하나,
# 사용자의 의도를 존중하여 데이터 범위에 맞게 역코딩합니다.
# 여기서는 일반적인 4점 척도 역코딩(5 - x)을 적용합니다.
df_analysis['Behavior'] = (5 - b4_data).mean(axis=1)

# 2. Attitude (태도)
# B1 등: 5점 척도 가정 (1:매우그렇다 ~ 5:전혀아니다 인 경우 6-x)
df_analysis['Att_B1_r'] = 6 - df_analysis['B1']
att_pos = df_analysis[['Att_B1_r', 'B3_5', 'B3_6']]
neg_att_cols = ['A14_7', 'B6_2', 'B7_1', 'B7_2']
att_neg_r = 6 - df_analysis[neg_att_cols] # 일괄 역코딩

df_analysis['Attitude'] = pd.concat([att_pos, att_neg_r], axis=1).mean(axis=1)

# 3. PN (개인규범), SN (주관적규범), PBC (행동통제)
df_analysis['PN'] = df_analysis[['B3_4', 'A14_5', 'A14_6', 'B3_11']].mean(axis=1)
df_analysis['SN'] = df_analysis[['B2_1', 'B2_2', 'B3_3']].mean(axis=1)
df_analysis['PBC'] = df_analysis[['B7_6', 'B7_5']].mean(axis=1)

# 신뢰도 확인 (Behavior 예시)
print(f"Behavior Cronbach Alpha: {cronbach_alpha(b4_data):.3f}")

print("\n--- 척도 생성 완료 (9값 결측 처리 및 역코딩 적용) ---")

# =============================================================================
# 2. 회귀분석 (가설 검증)
# =============================================================================

core_vars = ['Behavior', 'Attitude', 'SN', 'PBC', 'PN']
df_sem_data = df_analysis[core_vars].dropna().copy()

# 표준화 (회귀분석 비교용)
df_scaled = pd.DataFrame()
for col in core_vars:
    df_scaled[f'{col}_z'] = (df_sem_data[col] - df_sem_data[col].mean()) / df_sem_data[col].std()

# VIF (다중공선성) 확인
X_vif = sm.add_constant(df_scaled[['Attitude_z', 'SN_z', 'PBC_z', 'PN_z']])
vif_df = pd.DataFrame()
vif_df["Variable"] = X_vif.columns[1:]
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
print("\n[VIF 결과]\n", vif_df.round(2))

# 회귀모델 실행
model_full = smf.ols('Behavior_z ~ Attitude_z + SN_z + PBC_z + PN_z', data=df_scaled).fit()
print("\n" + "="*50)
print(" [회귀분석] 행동 영향요인 분석")
print("="*50)
print(model_full.summary())

# =============================================================================
# 3. 인구통계학적 변수 처리
# =============================================================================

def classify_income(x):
    if pd.isna(x): return np.nan
    if x <= 3: return '저소득'
    elif x <= 6: return '중소득'
    else: return '고소득'

df_analysis['Income_Group'] = df_analysis['F6'].apply(classify_income)
df_analysis['Age_Group'] = df_analysis['SQ3_1'].map({2:'20대', 3:'30대', 4:'40대', 5:'50대', 6:'60대+'})
df_analysis['Gender_Group'] = df_analysis['SQ2'].map({1:'남성', 2:'여성'})

# =============================================================================
# 4. [수정 포인트 2] 의식-실천 Gap 분석 (Min-Max 정규화 적용)
# Z-score 대신 절대적 척도(0~1)로 변환하여 실제 차이를 계산합니다.
# =============================================================================

print("\n\n" + "="*80)
print(" [수정됨] 의식-실천 Gap 분석 (절대평가 방식)")
print("="*80)

df_gap = df_analysis[['PN', 'Behavior', 'Income_Group']].dropna().copy()

# Min-Max Scaling 함수 (점수를 0~100점 만점으로 환산한다고 생각하면 됨)
# 5점 척도 기준(1~5)과 4점 척도 기준(1~4)이 섞여있을 수 있으므로 각각 처리
def normalize(series, min_v, max_v):
    return (series - min_v) / (max_v - min_v)

# PN은 5점 척도 가정, Behavior는 4점 척도 가정 (데이터 확인 후 min/max 조정 필요)
# 여기서는 자동 감지된 min/max를 사용하거나, 척도 정의에 따라 수동 입력 권장
df_gap['PN_norm'] = normalize(df_gap['PN'], 1, 5)        # 1~5점 척도
df_gap['Behavior_norm'] = normalize(df_gap['Behavior'], 1, 4) # 1~4점 척도

# Gap = 의식(규범) - 행동
# 값이 클수록 '말만 하고 행동하지 않음'
df_gap['True_Gap'] = df_gap['PN_norm'] - df_gap['Behavior_norm']

print(f"전체 평균 Gap (0~1 척도): {df_gap['True_Gap'].mean():.4f}")
print("  * 양수(+)일수록 의식 대비 행동이 부족함을 의미")

# Gap이 큰 집단 (상위 25%가 아닌, 절대적으로 차이가 0.3 이상인 경우 등)
high_gap_count = len(df_gap[df_gap['True_Gap'] > 0.25]) # 25% 이상 차이
print(f"실질적 괴리 집단 수: {high_gap_count}명 ({high_gap_count/len(df_gap)*100:.1f}%)")

# =============================================================================
# 5. [수정 포인트 3] 9사분면 프로파일링 (절대평가 기준 적용)
# 상대평가(분위수)가 아닌 점수 기준(보통 이상/이하)으로 분류합니다.
# =============================================================================

print("\n\n" + "="*80)
print(" [수정됨] 9사분면 프로파일링 (절대평가)")
print("="*80)

# 기준점 설정 (예: 5점 척도에서 3.0, 4.0 / 4점 척도에서 2.5, 3.5)
# PN(1~5): 3.0 미만(낮음), 4.0 이상(높음)
# Behavior(1~4): 2.0 미만(낮음), 3.0 이상(높음)

def classify_pn_abs(x):
    if x < 3.0: return '의식 낮음'
    elif x < 4.0: return '의식 중간'
    else: return '의식 높음'

def classify_beh_abs(x):
    if x < 2.5: return '실천 낮음' # 4점 척도의 중간값 근처
    elif x < 3.25: return '실천 중간'
    else: return '실천 높음'

df_gap['Profile'] = df_gap['PN'].apply(classify_pn_abs) + ' / ' + df_gap['Behavior'].apply(classify_beh_abs)

profile_counts = df_gap['Profile'].value_counts()
total_n = len(df_gap)

print("\n[프로파일 분포]")
for profile, count in profile_counts.items():
    print(f"  {profile}: {count}명 ({count/total_n*100:.1f}%)")

# 주요 타겟 그룹(의식은 높으나 실천이 낮은 그룹)
target_group = profile_counts.get('의식 높음 / 실천 낮음', 0) + profile_counts.get('의식 높음 / 실천 중간', 0)
print(f"\n>> 잠재적 개선 타겟 (의식 높음 but 실천 미흡): {target_group}명 ({target_group/total_n*100:.1f}%)")

print("\n" + "="*80)
print(" 분석 및 검증 완료")
print("="*80)


# =============================================================================
# [추가 복구 1] H1, H3 추가 가설 검증
# =============================================================================
print("\n" + "="*50)
print(" [추가 분석] H1 & H3 가설 검증")
print("="*50)

# H1: PN(개인규범) 형성 요인 분석
model_h1 = smf.ols('PN_z ~ Attitude_z + SN_z + PBC_z', data=df_scaled).fit()
print(f"H1 (PN 형성요인) R-squared: {model_h1.rsquared:.3f}")

# H3: 모델 비교 (TPB vs 통합)
model_tpb = smf.ols('Behavior_z ~ Attitude_z + SN_z + PBC_z', data=df_scaled).fit()
print(f"H3 모델 설명력 비교:")
print(f" - TPB 모델 (PN 제외): R²={model_tpb.rsquared:.3f}")
print(f" - 통합 모델 (PN 포함): R²={model_full.rsquared:.3f}")


# =============================================================================
# [추가 복구 2] 인구통계학적 차이 분석 (ANOVA / T-test)
# =============================================================================
print("\n" + "="*50)
print(" [추가 분석] 인구통계학적 차이 (ANOVA/T-test)")
print("="*50)

# 분석할 변수 및 그룹
targets = ['PN', 'Attitude', 'Behavior']
groups = ['Income_Group', 'Age_Group', 'Gender_Group']

for grp in groups:
    print(f"\n>> [{grp}]에 따른 차이 분석")
    for var in targets:
        temp_df = df_analysis[[grp, var]].dropna()
        unique_grps = temp_df[grp].unique()
        
        # 그룹이 2개면 T-test, 3개 이상이면 ANOVA
        if len(unique_grps) == 2:
            g1 = temp_df[temp_df[grp] == unique_grps[0]][var]
            g2 = temp_df[temp_df[grp] == unique_grps[1]][var]
            stat, p = stats.ttest_ind(g1, g2)
            test_type = "T-test"
        else:
            data_list = [temp_df[temp_df[grp] == g][var] for g in unique_grps]
            stat, p = stats.f_oneway(*data_list)
            test_type = "ANOVA"
            
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"   - {var}: {test_type} p={p:.4f} ({sig})")


# =============================================================================
# [추가 복구 3] Gap 예측 모델 (수정된 Gap 변수 사용)
# =============================================================================
print("\n" + "="*50)
print(" [추가 분석] Gap(언행불일치) 예측 요인")
print("="*50)

# 소득을 숫자로 변환 (1, 2, 3)
df_gap['Income_Code'] = df_gap['Income_Group'].map({'저소득':1, '중소득':2, '고소득':3})

# 데이터 병합 (PBC 표준화 점수 가져오기)
df_gap = df_gap.join(df_scaled['PBC_z'], how='left')

try:
    # 수정된 True_Gap(절대적 차이)을 종속변수로 사용해야 정확함!
    gap_model = smf.ols('True_Gap ~ PBC_z + Income_Code', data=df_gap.dropna()).fit()
    print(gap_model.summary().tables[1])
    print(" -> PBC(행동통제력)가 낮을수록 Gap이 커지는지 확인하세요. (Coef가 음수면 반대)")
except:
    print("데이터 부족으로 Gap 예측 모델 생략")


# =============================================================================
# [추가 복구 4] 조절효과 분석 (소득의 조절효과)
# =============================================================================
print("\n" + "="*50)
print(" [추가 분석] 조절효과 (Income x PN)")
print("="*50)

# 소득 표준화
df_mod = df_analysis[['PN', 'Behavior', 'F6']].dropna() # F6이 소득 원본
df_mod['PN_z'] = (df_mod['PN'] - df_mod['PN'].mean()) / df_mod['PN'].std()
df_mod['Behavior_z'] = (df_mod['Behavior'] - df_mod['Behavior'].mean()) / df_mod['Behavior'].std()
df_mod['Income_z'] = (df_mod['F6'] - df_mod['F6'].mean()) / df_mod['F6'].std()

# 상호작용 항 포함 회귀
mod_model = smf.ols('Behavior_z ~ PN_z * Income_z', data=df_mod).fit()
print(f"상호작용항(PN:Income) P-value: {mod_model.pvalues['PN_z:Income_z']:.4f}")
if model_full.pvalues['PN_z'] < 0.05:
    print(" -> 유의미한 조절효과가 관찰됨" if mod_model.pvalues['PN_z:Income_z'] < 0.05 else " -> 조절효과 없음")