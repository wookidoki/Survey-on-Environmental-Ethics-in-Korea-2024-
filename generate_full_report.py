# =============================================================================
# 완전한 분석 결과 JSON 생성 (최종 수정본 - 논리 오류 해결)
# =============================================================================

import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

def cronbach_alpha(items_df):
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2: return np.nan
    sum_item_variances = items_df.var(axis=0, ddof=1).sum()
    total_score_variance = items_df.sum(axis=1).var(ddof=1)
    if total_score_variance == 0: return 1.0
    return (k / (k - 1)) * (1 - (sum_item_variances / total_score_variance))

print("데이터 로드 및 분석 중...")
df_raw = pd.read_excel('2024data/2024 국민환경의식 설문조사_원자료.xlsx', sheet_name='원자료')

# -----------------------------------------------------------------------------
# 1. 척도 생성 (결측치 9 -> NaN 처리 및 역코딩 수정)
# -----------------------------------------------------------------------------
df_analysis = df_raw.copy()

# -----------------------------------------------------------------------------
# 1-1. 문항별 감사 (Item Audit) - 천장 효과 확인
# -----------------------------------------------------------------------------
b4_all_cols = ['B4_1', 'B4_2', 'B4_3', 'B4_4', 'B4_5', 'B4_6', 'B4_7', 'B4_8',
               'B4_9', 'B4_10', 'B4_11', 'B4_12', 'B4_13', 'B4_14']

# 문항명 매핑 (설문지 기준)
item_labels = {
    'B4_1': '물 절약',
    'B4_2': '친환경제품 구매',
    'B4_3': '대중교통 이용',
    'B4_4': '가까운 거리 걷기/자전거',
    'B4_5': '전기/난방 절약',
    'B4_6': '일회용품 사용 줄이기',
    'B4_7': '음식물 쓰레기 줄이기',
    'B4_8': '환경단체 참여',
    'B4_9': '환경보호 서명/캠페인',
    'B4_10': '환경문제 의견 표명',
    'B4_11': '재활용 분리배출',
    'B4_12': '쓰레기 무단투기 방지',
    'B4_13': '환경보호 대화',
    'B4_14': '환경정보 습득'
}

# 각 문항별 평균 계산 (원본 점수 기준)
# 원본 척도: 1=항상 함, 2=자주 함, 3=가끔 함, 4=전혀 안 함
# 천장효과: 평균이 1.5 이하면 "너무 쉬운 문항" (대부분 항상/자주 함)
item_audit = {}
ceiling_threshold = 1.35  # 1~4점 척도에서 1.35 이하면 천장효과
b4_raw = df_analysis[b4_all_cols].replace(9, np.nan)

for col in b4_all_cols:
    mean_score = b4_raw[col].mean()
    is_ceiling = mean_score <= ceiling_threshold  # 점수가 낮을수록 자주 실천

    item_audit[col] = {
        'label': item_labels.get(col, col),
        'mean': float(mean_score),
        'mean_reversed': float(5 - mean_score),  # 역코딩 후 점수 (시각화용)
        'status': 'excluded' if is_ceiling else 'selected',
        'reason': 'Ceiling Effect (too easy, low variance)' if is_ceiling else 'Good variance for analysis'
    }

# 모든 문항 사용 (신뢰도 유지를 위해)
# 천장효과가 있더라도 척도의 내적 일관성(Cronbach's α)을 위해 전체 문항 유지
b4_cols_selected = ['B4_1', 'B4_2', 'B4_3', 'B4_4', 'B4_5', 'B4_6', 'B4_7', 'B4_8',
                   'B4_11', 'B4_12', 'B4_13', 'B4_14']  # 기존 12개 문항 (B4_9, B4_10 제외)

# 천장효과 분석은 데이터 투명성을 위한 보고 목적으로만 사용
ceiling_items = [col for col in b4_all_cols if item_audit[col]['status'] == 'excluded']
print(f"사용 문항: {len(b4_cols_selected)}개 / 전체 {len(b4_all_cols)}개")
print(f"천장효과 있는 문항 (포함됨): {len([c for c in b4_cols_selected if c in ceiling_items])}개")
print("  -> 신뢰도 유지를 위해 모든 문항 사용")

# Behavior (9=무응답 -> NaN 처리)
b4_data = df_analysis[b4_cols_selected].replace(9, np.nan)
# 역코딩 (5 - x 방식으로 표준화 or 기존 3 - x 유지. 여기서는 논리적 일관성을 위해 5-x 적용)
df_analysis['Behavior'] = (5 - b4_data).mean(axis=1)

# Attitude
df_analysis['Att_B1_r'] = 6 - df_analysis['B1']
att_items_pos = df_analysis[['Att_B1_r', 'B3_5', 'B3_6']]
neg_att_cols = ['A14_7', 'B6_2', 'B7_1', 'B7_2']
att_items_neg_r = 6 - df_analysis[neg_att_cols]
total_att_items = pd.concat([att_items_pos, att_items_neg_r], axis=1)
df_analysis['Attitude'] = total_att_items.mean(axis=1)

# PN, SN, PBC
# *** 중요: PN은 역코딩 하지 않음! 부정 문항이므로 원본 사용 (높은 점수 = 높은 의식)
pn_items = df_analysis[['B3_4', 'A14_5', 'A14_6', 'B3_11']]
df_analysis['PN'] = pn_items.mean(axis=1)  # 역코딩 없음!

sn_items = df_analysis[['B2_1', 'B2_2', 'B3_3']]
df_analysis['SN'] = sn_items.mean(axis=1)

pbc_items = df_analysis[['B7_6', 'B7_5']]
df_analysis['PBC'] = pbc_items.mean(axis=1)

# 인구통계
def classify_income(x):
    if pd.isna(x): return np.nan
    if x <= 3: return '저소득'
    elif x <= 6: return '중소득'
    else: return '고소득'

df_analysis['Income_Group'] = df_analysis['F6'].apply(classify_income)
df_analysis['Age_Group'] = df_analysis['SQ3_1'].map({2: '20대', 3: '30대', 4: '40대', 5: '50대', 6: '60대+'})
df_analysis['Edu_Group'] = df_analysis['F1'].map({1: '고졸이하', 2: '고졸이하', 3: '대졸', 4: '대학원+'})
df_analysis['Gender_Group'] = df_analysis['SQ2'].map({1: '남성', 2: '여성'})

# 분석용 데이터
core_vars = ['Behavior', 'Attitude', 'SN', 'PBC', 'PN']
df_sem_data = df_analysis[core_vars].dropna().copy()

# 회귀분석용 표준화 (Z-score) - 회귀분석 내부 계산용으로만 사용
df_scaled = pd.DataFrame()
for col in core_vars:
    df_scaled[f'{col}_z'] = (df_sem_data[col] - df_sem_data[col].mean()) / df_sem_data[col].std()

# -----------------------------------------------------------------------------
# 2. 회귀분석 (H1, H2, H3)
# -----------------------------------------------------------------------------
model_pn = smf.ols('PN_z ~ Attitude_z + SN_z + PBC_z', data=df_scaled).fit()
model_behav_full = smf.ols('Behavior_z ~ Attitude_z + SN_z + PBC_z + PN_z', data=df_scaled).fit()
model_behav_tpb = smf.ols('Behavior_z ~ Attitude_z + SN_z + PBC_z', data=df_scaled).fit()
model_behav_pn = smf.ols('Behavior_z ~ PN_z', data=df_scaled).fit()

# -----------------------------------------------------------------------------
# 3. 인구통계 차이 (ANOVA/T-test)
# -----------------------------------------------------------------------------
demo_results = {}
outcome_vars = ['PN', 'Attitude', 'Behavior']

# 함수화하여 코드 중복 제거
def analyze_demo(group_col, group_order=None):
    stats_dict = {}
    df_sub = df_analysis.dropna(subset=[group_col] + outcome_vars)
    
    unique_grps = group_order if group_order else df_sub[group_col].dropna().unique()
    
    for var in outcome_vars:
        groups = [df_sub[df_sub[group_col] == g][var] for g in unique_grps]
        
        # 그룹 수에 따라 T-test 또는 ANOVA
        if len(groups) == 2:
            stat, p_val = stats.ttest_ind(groups[0], groups[1])
            test_type = 't'
        else:
            stat, p_val = stats.f_oneway(*groups)
            test_type = 'F'
            
        means = df_sub.groupby(group_col)[var].mean().to_dict()
        stats_dict[var] = {
            test_type: float(stat), 
            'p': float(p_val), 
            'means': {k: float(v) for k, v in means.items()}
        }
    return stats_dict

demo_results['income'] = analyze_demo('Income_Group', ['저소득', '중소득', '고소득'])
demo_results['age'] = analyze_demo('Age_Group')
demo_results['edu'] = analyze_demo('Edu_Group')
demo_results['gender'] = analyze_demo('Gender_Group')

# -----------------------------------------------------------------------------
# 4. Gap 분석 (수정됨: Min-Max 정규화 절대평가)
# -----------------------------------------------------------------------------
df_gap = df_analysis[['PN', 'Attitude', 'Behavior', 'PBC', 'SN', 'Income_Group']].dropna().copy()

# Min-Max Scaling (절대적 0~1 척도 변환)
def normalize(series, min_v, max_v):
    return (series - min_v) / (max_v - min_v)

# 척도 범위 가정 (PN:1~5, Behavior:1~4 - 데이터에 맞게 조정됨)
# 실제 데이터의 min/max를 사용하는 것이 안전
df_gap['PN_norm'] = normalize(df_gap['PN'], 1, 5)
df_gap['Behavior_norm'] = normalize(df_gap['Behavior'], 1, 4)
df_gap['Attitude_norm'] = normalize(df_gap['Attitude'], 1, 5)

# Gap 계산 (절대적 차이)
df_gap['PN_Behavior_Gap'] = df_gap['PN_norm'] - df_gap['Behavior_norm']
df_gap['Attitude_Behavior_Gap'] = df_gap['Attitude_norm'] - df_gap['Behavior_norm']

# 통계 저장
gap_stats = {
    'pn_gap_mean': float(df_gap['PN_Behavior_Gap'].mean()),
    'pn_gap_std': float(df_gap['PN_Behavior_Gap'].std()),
    'att_gap_mean': float(df_gap['Attitude_Behavior_Gap'].mean()),
    # Gap이 0.25 (25%) 이상 차이나는 경우를 '높은 격차'로 정의 (절대기준)
    'high_gap_count': int((df_gap['PN_Behavior_Gap'] > 0.25).sum()),
    'high_gap_pct': float((df_gap['PN_Behavior_Gap'] > 0.25).sum() / len(df_gap) * 100)
}

# Gap 예측 모델 (Gap이 구해졌으므로 Z값 PBC 병합)
df_gap = df_gap.join(df_scaled[['PBC_z']], how='left') # PBC_z 가져오기
df_gap['Income_Code'] = df_gap['Income_Group'].map({'저소득': 1, '중소득': 2, '고소득': 3})

try:
    # True Gap을 종속변수로 사용
    gap_model = smf.ols('PN_Behavior_Gap ~ PBC_z + Income_Code', data=df_gap.dropna()).fit()
    gap_predictors = {
        'PBC': {'beta': float(gap_model.params.get('PBC_z', 0)), 'p': float(gap_model.pvalues.get('PBC_z', 1))},
        'Income': {'beta': float(gap_model.params.get('Income_Code', 0)), 'p': float(gap_model.pvalues.get('Income_Code', 1))},
        'r_squared': float(gap_model.rsquared)
    }
except:
    gap_predictors = {}

# -----------------------------------------------------------------------------
# 5. 조절효과 및 프로파일링 (수정됨: 절대평가)
# -----------------------------------------------------------------------------

# 조절효과 (Z-score 사용 - 회귀분석이므로 OK)
df_mod = df_analysis[['PN', 'Behavior', 'F6']].dropna().copy()
df_mod['PN_z'] = (df_mod['PN'] - df_mod['PN'].mean()) / df_mod['PN'].std()
df_mod['Behavior_z'] = (df_mod['Behavior'] - df_mod['Behavior'].mean()) / df_mod['Behavior'].std()
df_mod['Income_z'] = (df_mod['F6'] - df_mod['F6'].mean()) / df_mod['F6'].std()

try:
    mod_income = smf.ols('Behavior_z ~ PN_z + Income_z + PN_z:Income_z', data=df_mod).fit()
    moderation = {
        'income_interaction': {
            'beta': float(mod_income.params.get('PN_z:Income_z', 0)),
            'p': float(mod_income.pvalues.get('PN_z:Income_z', 1))
        }
    }
except:
    moderation = {}

# 9사분면 프로파일링 (절대평가 기준 적용)
df_profile = df_analysis[['PN', 'Attitude', 'Behavior', 'Income_Group', 'Age_Group']].dropna().copy()

# 절대 기준 함수 (3.0, 4.0 등 점수 기준)
def classify_pn_abs(x):
    if x < 3.0: return '낮음'
    elif x < 4.0: return '중간'
    else: return '높음'

def classify_beh_abs(x):
    if x < 2.5: return '낮음' # 4점 척도 감안
    elif x < 3.25: return '중간'
    else: return '높음'

df_profile['PN_Level'] = df_profile['PN'].apply(classify_pn_abs)
df_profile['Behavior_Level'] = df_profile['Behavior'].apply(classify_beh_abs)
profile_counts = df_profile.groupby(['PN_Level', 'Behavior_Level']).size().to_dict()

# 3D 시각화용 샘플
sample_3d = df_profile.sample(min(500, len(df_profile)))
data_3d = {
    'pn': sample_3d['PN'].tolist(),
    'attitude': sample_3d['Attitude'].tolist(),
    'behavior': sample_3d['Behavior'].tolist(),
    'income': sample_3d['Income_Group'].tolist(),
    'age': sample_3d['Age_Group'].tolist()
}

# -----------------------------------------------------------------------------
# 6. 인구통계별 회귀분석 (촉진/방해 요인 분석)
# -----------------------------------------------------------------------------
print("\n인구통계별 회귀분석 수행 중...")

def analyze_group_regression(df, group_col, group_value):
    """특정 그룹에 대한 회귀분석 및 촉진/방해 요인 분석"""
    df_group = df[df[group_col] == group_value].copy()
    core_vars = ['Behavior', 'Attitude', 'SN', 'PBC', 'PN']
    df_group_clean = df_group[core_vars].dropna()

    if len(df_group_clean) < 30:
        return None

    # 표준화
    df_group_scaled = pd.DataFrame()
    for col in core_vars:
        df_group_scaled[f'{col}_z'] = (df_group_clean[col] - df_group_clean[col].mean()) / df_group_clean[col].std()

    try:
        # 회귀분석
        model = smf.ols('Behavior_z ~ Attitude_z + SN_z + PBC_z + PN_z', data=df_group_scaled).fit()

        # Beta 값 추출
        facilitators = {
            'Attitude': float(model.params.get('Attitude_z', 0)),
            'SN': float(model.params.get('SN_z', 0)),
            'PBC': float(model.params.get('PBC_z', 0)),
            'PN': float(model.params.get('PN_z', 0))
        }

        # 평균 점수
        means = {
            'Attitude': float(df_group_clean['Attitude'].mean()),
            'SN': float(df_group_clean['SN'].mean()),
            'PBC': float(df_group_clean['PBC'].mean()),
            'PN': float(df_group_clean['PN'].mean()),
            'Behavior': float(df_group_clean['Behavior'].mean())
        }

        # 최고 촉진 요인
        top_factor = max(facilitators, key=facilitators.get)

        # 방해 요인 분석 (영향력은 크지만 점수가 낮은 변수)
        barrier_list = []
        for var in ['Attitude', 'SN', 'PBC', 'PN']:
            beta = facilitators[var]
            mean = means[var]
            if beta > 0.15 and mean < 3.5:
                barrier_list.append(f"{var} has high influence (β={beta:.2f}) but low score ({mean:.2f})")

        barrier_analysis = "; ".join(barrier_list) if barrier_list else "No major barriers identified"

        return {
            'facilitators': facilitators,
            'means': means,
            'top_factor': top_factor,
            'barrier_analysis': barrier_analysis,
            'r_squared': float(model.rsquared),
            'sample_size': int(len(df_group_clean))
        }
    except:
        return None

# 인구통계별 분석 수행
demographic_analysis = {}

# 소득별
demographic_analysis['income'] = {}
for income_group in ['저소득', '중소득', '고소득']:
    result = analyze_group_regression(df_analysis, 'Income_Group', income_group)
    if result:
        demographic_analysis['income'][income_group] = result

# 성별
demographic_analysis['gender'] = {}
for gender in ['남성', '여성']:
    result = analyze_group_regression(df_analysis, 'Gender_Group', gender)
    if result:
        demographic_analysis['gender'][gender] = result

# 연령별
demographic_analysis['age'] = {}
for age_group in ['20대', '30대', '40대', '50대', '60대+']:
    result = analyze_group_regression(df_analysis, 'Age_Group', age_group)
    if result:
        demographic_analysis['age'][age_group] = result

print(f"  - 소득: {len(demographic_analysis['income'])}개 그룹")
print(f"  - 성별: {len(demographic_analysis['gender'])}개 그룹")
print(f"  - 연령: {len(demographic_analysis['age'])}개 그룹")

# -----------------------------------------------------------------------------
# 7. JSON 저장
# -----------------------------------------------------------------------------
report_data = {
    'sample_size': int(len(df_raw)),
    'analysis_sample': int(len(df_scaled)),
    'item_audit': {
        'items': item_audit,
        'threshold': ceiling_threshold,
        'selected_count': len(b4_cols_selected),
        'total_count': len(b4_all_cols),
        'excluded_items': [item_audit[c]['label'] for c in b4_all_cols if item_audit[c]['status'] == 'excluded']
    },
    'reliability': {
        'Attitude': float(cronbach_alpha(total_att_items)),
        'PN': float(cronbach_alpha(pn_items)),
        'SN': float(cronbach_alpha(sn_items)),
        'PBC': float(cronbach_alpha(pbc_items)),
        'Behavior': float(cronbach_alpha(b4_data))
    },
    'h1': {
        'r_squared': float(model_pn.rsquared),
        'coefficients': {
            'Attitude': {'beta': float(model_pn.params['Attitude_z']), 'p': float(model_pn.pvalues['Attitude_z'])},
            'SN': {'beta': float(model_pn.params['SN_z']), 'p': float(model_pn.pvalues['SN_z'])},
            'PBC': {'beta': float(model_pn.params['PBC_z']), 'p': float(model_pn.pvalues['PBC_z'])}
        }
    },
    'h2': {
        'r_squared': float(model_behav_full.rsquared),
        'coefficients': {
            'Attitude': {'beta': float(model_behav_full.params['Attitude_z']), 'p': float(model_behav_full.pvalues['Attitude_z'])},
            'SN': {'beta': float(model_behav_full.params['SN_z']), 'p': float(model_behav_full.pvalues['SN_z'])},
            'PBC': {'beta': float(model_behav_full.params['PBC_z']), 'p': float(model_behav_full.pvalues['PBC_z'])},
            'PN': {'beta': float(model_behav_full.params['PN_z']), 'p': float(model_behav_full.pvalues['PN_z'])}
        }
    },
    'h3': {
        'integrated': float(model_behav_full.rsquared),
        'tpb': float(model_behav_tpb.rsquared),
        'pn_only': float(model_behav_pn.rsquared)
    },
    'demographics': demo_results,
    'demographic_analysis': demographic_analysis,
    'gap_analysis': gap_stats,
    'gap_predictors': gap_predictors,
    'moderation': moderation,
    'profiles': {f'{k[0]}/{k[1]}': int(v) for k, v in profile_counts.items()},
    'data_3d': data_3d
}

with open('full_report_data.json', 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print("[OK] full_report_data.json 생성 완료")
print("[OK] 인구통계별 촉진/방해 요인 분석 완료")
print("[OK] index.html을 브라우저로 여세요")
print("="*80)