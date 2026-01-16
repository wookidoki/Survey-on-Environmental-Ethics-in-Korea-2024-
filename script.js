let reportData = null;

// 데이터 로드 및 초기화
fetch("./full_report_data.json")
  .then((response) => response.json())
  .then((data) => {
    reportData = data;
    renderFullReport(data);
  })
  .catch((error) => {
    console.error("Error:", error);
    alert(
      "full_report_data.json을 찾을 수 없습니다. 분석 스크립트(main.py)를 먼저 실행하세요."
    );
  });

function renderFullReport(data) {
  // 0. 데이터 감사 렌더링 (Item Audit)
  renderItemAudit(data.item_audit);

  // 1. 기본 정보 렌더링
  document.getElementById("sample-size").textContent = data.sample_size;
  document.getElementById("sample-size-text").textContent = data.sample_size;
  document.getElementById("analysis-sample").textContent = data.analysis_sample;

  const avgAlpha =
    Object.values(data.reliability).reduce((a, b) => a + b, 0) /
    Object.keys(data.reliability).length;
  document.getElementById("avg-alpha").textContent = avgAlpha.toFixed(3);
  document.getElementById("model-r2").textContent =
    data.h2.r_squared.toFixed(3);

  // 2. 신뢰도 렌더링
  renderReliability(data.reliability);

  // 3. 가설 검증 (H1, H2, H3)
  renderH1(data.h1);
  renderH2(data.h2);
  renderH3(data.h3);

  // 4. 인구통계
  renderDemographics(data.demographics);

  // 5. Gap 분석
  renderGapAnalysis(data.gap_analysis, data.gap_predictors);

  // 6. 조절효과
  renderModeration(data.moderation);

  // 7. 프로파일링
  renderProfiles(data.profiles, data.analysis_sample);

  // 8. 3D 시각화
  render3D(data.data_3d);
}

// --- Helper Functions ---
function getSigText(p) {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

function getSigClass(p) {
  if (p < 0.001) return "sig-high";
  if (p < 0.01) return "sig-med";
  if (p < 0.05) return "sig-low";
  return "sig-none";
}

// --- Rendering Functions ---
function renderItemAudit(itemAudit) {
  // 기준선 표시
  document.getElementById("ceiling-threshold").textContent =
    itemAudit.threshold.toFixed(2);

  // 차트 데이터 준비
  const items = itemAudit.items;
  const itemIds = Object.keys(items).sort();
  const labels = itemIds.map((id) => items[id].label);
  const means = itemIds.map((id) => items[id].mean);


  const backgroundColors = itemIds.map((id) =>
    b4_used.includes(id) ? "#667eea" : "#95a5a6"
  );

  // 차트 생성
  new Chart(document.getElementById("item-audit-chart"), {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "평균 점수 (1=항상 함, 4=전혀 안 함)",
          data: means,
          backgroundColor: backgroundColors,
        },
      ],
    },
    options: {
      indexAxis: "y", // 가로 막대 그래프
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "행동 문항별 평균 점수 및 선정 여부",
          font: { size: 16 },
        },
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 4,
          title: {
            display: true,
            text: "평균 점수 (낮을수록 자주 실천)",
          },
        },
      },
      // 기준선 추가
      annotation: {
        annotations: [
          {
            type: "line",
            xMin: itemAudit.threshold,
            xMax: itemAudit.threshold,
            borderColor: "#e74c3c",
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
              content: "천장 효과 기준선",
              enabled: true,
              position: "end",
            },
          },
        ],
      },
    },
  });

  // 테이블 렌더링
  const table = document.querySelector("#item-audit-table tbody");
  const b4_used = [
    "B4_1",
    "B4_2",
    "B4_3",
    "B4_4",
    "B4_5",
    "B4_6",
    "B4_7",
    "B4_8",
    "B4_11",
    "B4_12",
    "B4_13",
    "B4_14",
  ]; // 실제 분석에 사용된 12개 문항

  itemIds.forEach((itemId) => {
    const item = items[itemId];
    const row = table.insertRow();
    const isUsed = b4_used.includes(itemId);
    const hasCeiling = item.status === "excluded"; // 천장효과 있음

    const usedText = isUsed ? "✓ 포함" : "✗ 제외";
    const usedClass = isUsed ? "sig-low" : "sig-none";
    const ceilingText = hasCeiling ? "있음" : "없음";
    const ceilingClass = hasCeiling ? "sig-none" : "sig-low";

    row.innerHTML = `
            <td><strong>${itemId}</strong></td>
            <td>${item.label}</td>
            <td>${item.mean.toFixed(3)}</td>
            <td>${item.mean_reversed.toFixed(3)}</td>
            <td class="significance ${usedClass}"><strong>${usedText}</strong></td>
            <td class="significance ${ceilingClass}">${ceilingText}</td>
            <td style="font-size: 0.9em">${
              isUsed
                ? hasCeiling
                  ? "천장효과 있으나 신뢰도 유지를 위해 포함"
                  : "정상적인 분산, 분석에 포함"
                : "분석 대상 아님 (B4_9, B4_10)"
            }</td>
        `;
  });
}

function renderReliability(reliability) {
  // 텍스트 업데이트
  document.getElementById("alpha-attitude").textContent =
    reliability.Attitude.toFixed(3);
  document.getElementById("alpha-sn").textContent = reliability.SN.toFixed(3);
  document.getElementById("alpha-pbc").textContent = reliability.PBC.toFixed(3);
  document.getElementById("alpha-pn").textContent = reliability.PN.toFixed(3);
  document.getElementById("alpha-behavior").textContent =
    reliability.Behavior.toFixed(3);
  document.getElementById("behavior-alpha-text").textContent =
    reliability.Behavior.toFixed(3);
  document.getElementById("pbc-alpha-text").textContent =
    reliability.PBC.toFixed(3);
  document.getElementById("pbc-alpha-method").textContent =
    reliability.PBC.toFixed(3);

  // 테이블 업데이트
  const table = document.querySelector("#reliability-table tbody");
  const itemCounts = { Attitude: 7, PN: 4, SN: 3, PBC: 2, Behavior: 12 };
  Object.entries(reliability).forEach(([key, value]) => {
    const row = table.insertRow();
    row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${itemCounts[key]}</td>
            <td><strong>${value.toFixed(3)}</strong></td>
            <td>${value >= 0.8 ? "우수" : value >= 0.7 ? "양호" : "보통"}</td>
        `;
  });
}

function renderH1(h1) {
  document.getElementById("h1-r2").textContent = h1.r_squared.toFixed(3);
  document.getElementById("h1-r2-pct").textContent = (
    h1.r_squared * 100
  ).toFixed(1);

  const table = document.querySelector("#h1-table tbody");
  Object.entries(h1.coefficients).forEach(([key, val]) => {
    const row = table.insertRow();
    row.innerHTML = `
            <td>${key} → PN</td>
            <td><strong>${val.beta.toFixed(3)}</strong></td>
            <td>-</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
        `;
  });

  new Chart(document.getElementById("h1-chart"), {
    type: "bar",
    data: {
      labels: Object.keys(h1.coefficients),
      datasets: [
        {
          label: "표준화 계수 (β)",
          data: Object.values(h1.coefficients).map((v) => v.beta),
          backgroundColor: ["#667eea", "#764ba2", "#f093fb"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "H1: Attitude, SN, PBC → PN 경로 계수",
          font: { size: 16 },
        },
      },
      scales: { y: { beginAtZero: true, max: 0.5 } },
    },
  });
}

function renderH2(h2) {
  document.getElementById("h2-r2").textContent = h2.r_squared.toFixed(3);
  document.getElementById("h2-r2-pct").textContent = (
    h2.r_squared * 100
  ).toFixed(1);
  document.getElementById("pn-beta").textContent =
    h2.coefficients.PN.beta.toFixed(3);

  const table = document.querySelector("#h2-table tbody");
  Object.entries(h2.coefficients).forEach(([key, val]) => {
    const row = table.insertRow();
    row.innerHTML = `
            <td>${key} → Behavior</td>
            <td><strong>${val.beta.toFixed(3)}</strong></td>
            <td>-</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
        `;
  });

  new Chart(document.getElementById("h2-chart"), {
    type: "bar",
    data: {
      labels: Object.keys(h2.coefficients),
      datasets: [
        {
          label: "표준화 계수 (β)",
          data: Object.values(h2.coefficients).map((v) => v.beta),
          backgroundColor: ["#667eea", "#764ba2", "#f093fb", "#4facfe"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "H2: 4 Factors → Behavior 경로 계수",
          font: { size: 16 },
        },
      },
      scales: { y: { beginAtZero: true, max: 0.4 } },
    },
  });
}

function renderH3(h3) {
  const table = document.getElementById("h3-table");
  const tbody =
    table.querySelector("tbody") ||
    table.appendChild(document.createElement("tbody"));
  tbody.innerHTML = `
        <tr>
            <td>통합 모델 (4 factors)</td>
            <td><strong>${h3.integrated.toFixed(3)}</strong></td>
            <td>${(h3.integrated * 100).toFixed(1)}%</td>
            <td>-</td>
        </tr>
        <tr>
            <td>TPB 모델 (3 factors)</td>
            <td><strong>${h3.tpb.toFixed(3)}</strong></td>
            <td>${(h3.tpb * 100).toFixed(1)}%</td>
            <td>${(((h3.tpb - h3.integrated) / h3.integrated) * 100).toFixed(
              1
            )}%</td>
        </tr>
        <tr>
            <td>PN 단독 모델</td>
            <td><strong>${h3.pn_only.toFixed(3)}</strong></td>
            <td>${(h3.pn_only * 100).toFixed(1)}%</td>
            <td>${(
              ((h3.pn_only - h3.integrated) / h3.integrated) *
              100
            ).toFixed(1)}%</td>
        </tr>
    `;

  new Chart(document.getElementById("h3-chart"), {
    type: "bar",
    data: {
      labels: ["통합 모델", "TPB 모델", "PN 단독"],
      datasets: [
        {
          label: "R² (설명력)",
          data: [h3.integrated, h3.tpb, h3.pn_only],
          backgroundColor: ["#667eea", "#764ba2", "#f093fb"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "H3: 모델 설명력 비교",
          font: { size: 16 },
        },
      },
      scales: { y: { beginAtZero: true, max: 0.3 } },
    },
  });
}

function renderDemographics(demo) {
  // 1. 소득
  const incomeTable = document.querySelector("#income-table tbody");
  Object.entries(demo.income).forEach(([key, val]) => {
    const row = incomeTable.insertRow();
    row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${val.F.toFixed(2)}</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
            <td>${val.means["저소득"]?.toFixed(2) || "-"}</td>
            <td>${val.means["중소득"]?.toFixed(2) || "-"}</td>
            <td>${val.means["고소득"]?.toFixed(2) || "-"}</td>
        `;
  });

  new Chart(document.getElementById("income-chart"), {
    type: "bar",
    data: {
      labels: ["저소득", "중소득", "고소득"],
      datasets: ["PN", "Attitude", "Behavior"].map((v, i) => ({
        label: v,
        data: Object.values(demo.income[v].means),
        backgroundColor: ["#667eea", "#764ba2", "#f093fb"][i],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "소득 수준별 평균 비교",
          font: { size: 16 },
        },
      },
    },
  });

  // 2. 연령
  const ageTable = document.querySelector("#age-table tbody");
  Object.entries(demo.age).forEach(([key, val]) => {
    const row = ageTable.insertRow();
    row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${val.F.toFixed(2)}</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
        `;
  });

  new Chart(document.getElementById("age-chart"), {
    type: "line",
    data: {
      labels: Object.keys(demo.age.PN.means),
      datasets: ["PN", "Attitude", "Behavior"].map((v, i) => ({
        label: v,
        data: Object.values(demo.age[v].means),
        borderColor: ["#667eea", "#764ba2", "#f093fb"][i],
        backgroundColor: ["#667eea", "#764ba2", "#f093fb"][i],
        tension: 0.1,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "연령대별 평균 비교",
          font: { size: 16 },
        },
      },
    },
  });

  // 3. 학력
  const eduTable = document.querySelector("#edu-table tbody");
  Object.entries(demo.edu).forEach(([key, val]) => {
    const row = eduTable.insertRow();
    row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${val.F.toFixed(2)}</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
        `;
  });

  new Chart(document.getElementById("edu-chart"), {
    type: "bar",
    data: {
      labels: Object.keys(demo.edu.PN.means),
      datasets: ["PN", "Attitude", "Behavior"].map((v, i) => ({
        label: v,
        data: Object.values(demo.edu[v].means),
        backgroundColor: ["#667eea", "#764ba2", "#f093fb"][i],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: "학력별 평균 비교", font: { size: 16 } },
      },
    },
  });

  // 4. 성별
  const genderTable = document.querySelector("#gender-table tbody");
  Object.entries(demo.gender).forEach(([key, val]) => {
    const row = genderTable.insertRow();
    row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${val.t.toFixed(2)}</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
            <td>${val.means["남성"].toFixed(2)}</td>
            <td>${val.means["여성"].toFixed(2)}</td>
        `;
  });

  new Chart(document.getElementById("gender-chart"), {
    type: "bar",
    data: {
      labels: ["남성", "여성"],
      datasets: ["PN", "Attitude", "Behavior"].map((v, i) => ({
        label: v,
        data: Object.values(demo.gender[v].means),
        backgroundColor: ["#667eea", "#764ba2", "#f093fb"][i],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: "성별 평균 비교", font: { size: 16 } },
      },
    },
  });
}

function renderGapAnalysis(gap, predictors) {
  document.getElementById("gap-mean").textContent = gap.pn_gap_mean.toFixed(3);
  document.getElementById("high-gap-count").textContent =
    gap.high_gap_count + "명";
  document.getElementById("high-gap-pct").textContent =
    gap.high_gap_pct.toFixed(1) + "%";

  if (predictors && Object.keys(predictors).length > 0) {
    const table = document.querySelector("#gap-predictors-table tbody");
    Object.entries(predictors).forEach(([key, val]) => {
      if (key === "r_squared") return;
      const row = table.insertRow();
      const interpretation =
        key === "PBC"
          ? val.beta > 0
            ? "PBC 높을수록 Gap 작음 (실천 촉진)"
            : "PBC 낮을수록 Gap 큼 (방해요소)"
          : val.beta > 0
          ? "소득 높을수록 Gap 작음"
          : "소득 낮을수록 Gap 큼 (방해요소)";
      row.innerHTML = `
                <td><strong>${key}</strong></td>
                <td>${val.beta.toFixed(3)}</td>
                <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
                <td class="significance ${getSigClass(val.p)}">${getSigText(
        val.p
      )}</td>
                <td>${interpretation}</td>
            `;
    });
  }
}

function renderModeration(moderation) {
  if (moderation && moderation.income_interaction) {
    const table = document.querySelector("#moderation-table tbody");
    const val = moderation.income_interaction;
    const row = table.insertRow();
    const interpretation =
      val.p < 0.05
        ? val.beta > 0
          ? "고소득층에서 PN→Behavior 경로 강화"
          : "저소득층에서 PN→Behavior 경로 약화 (방해요소)"
        : "소득의 조절효과 없음 (소득과 무관하게 경로 동일)";
    row.innerHTML = `
            <td><strong>소득 (PN × Income)</strong></td>
            <td>${val.beta.toFixed(3)}</td>
            <td>${val.p < 0.0001 ? "<0.0001" : val.p.toFixed(4)}</td>
            <td class="significance ${getSigClass(val.p)}">${getSigText(
      val.p
    )}</td>
            <td>${interpretation}</td>
        `;
  }
}

function renderProfiles(profiles, total) {
  const table = document.querySelector("#profile-table tbody");
  const profileDescriptions = {
    "높음/높음": "이상적 집단 (환경 리더)",
    "높음/중간": "의식은 높으나 실천 보통",
    "높음/낮음": "괴리 집단 (의식만 높음)",
    "중간/높음": "실천은 높으나 의식 보통",
    "중간/중간": "평균 집단",
    "중간/낮음": "의식·실천 모두 보통 이하",
    "낮음/높음": "무의식 실천 (습관 또는 외부 압력)",
    "낮음/중간": "의식·실천 모두 낮음",
    "낮음/낮음": "전반적 낮음 집단",
  };

  Object.entries(profiles)
    .sort((a, b) => b[1] - a[1])
    .forEach(([key, val]) => {
      const row = table.insertRow();
      const pct = ((val / total) * 100).toFixed(1);
      row.innerHTML = `
            <td><strong>${key}</strong></td>
            <td>${val}명</td>
            <td><strong>${pct}%</strong></td>
            <td>${profileDescriptions[key] || "-"}</td>
        `;
    });

  const sortedProfiles = Object.entries(profiles).sort((a, b) => b[1] - a[1]);
  new Chart(document.getElementById("profile-chart"), {
    type: "bar",
    data: {
      labels: sortedProfiles.map((p) => p[0]),
      datasets: [
        {
          label: "인원",
          data: sortedProfiles.map((p) => p[1]),
          backgroundColor: "#667eea",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "9사분면 프로파일 분포",
          font: { size: 16 },
        },
      },
    },
  });
}

function render3D(data3d) {
  const colorScale = {
    저소득: "#e74c3c",
    중소득: "#f39c12",
    고소득: "#27ae60",
  };

  const trace = {
    x: data3d.pn,
    y: data3d.attitude,
    z: data3d.behavior,
    mode: "markers",
    marker: {
      size: 4,
      color: data3d.income.map((i) => colorScale[i] || "#95a5a6"),
      opacity: 0.6,
    },
    type: "scatter3d",
    text: data3d.income.map(
      (inc, i) =>
        `소득: ${inc}<br>연령: ${data3d.age[i]}<br>PN: ${data3d.pn[i].toFixed(
          2
        )}<br>Attitude: ${data3d.attitude[i].toFixed(
          2
        )}<br>Behavior: ${data3d.behavior[i].toFixed(2)}`
    ),
    hoverinfo: "text",
  };

  const layout = {
    scene: {
      xaxis: { title: "PN (환경의식)" },
      yaxis: { title: "Attitude (태도)" },
      zaxis: { title: "Behavior (실천)" },
    },
    margin: { l: 0, r: 0, b: 0, t: 0 },
  };

  Plotly.newPlot("plot-3d", [trace], layout);
}
