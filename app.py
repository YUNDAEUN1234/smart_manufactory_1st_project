"""
총괄생산계획 (Aggregate Production Planning) 최적화 대시보드
------------------------------------------------------------
원예장비 제조업체 사례를 기반으로 한 인터랙티브 의사결정 도구입니다.
수요·비용·생산능력 파라미터를 자유롭게 조정하며
LP 최적화 / 평준화 / 추종 세 전략을 즉시 비교할 수 있습니다.

- 모델   : PuLP + CBC 솔버 (Streamlit Cloud에서 추가 설치 불필요)
- 시각화 : Plotly (반응형, 한글 친화적)
- 전략   : LP 최적 / 평준화(Level) / 추종(Chase) 자동 비교
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st

# ============================================================
# 1. 페이지 / 테마 설정
# ============================================================
st.set_page_config(
    page_title="총괄생산계획 최적화 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 화이트 + 블루 계열 디자인 시스템 (AI 티 안 나는 차분한 톤)
PRIMARY = "#2563eb"     # 메인 블루
PRIMARY_LIGHT = "#60a5fa"
PRIMARY_PALE = "#dbeafe"
ACCENT = "#f59e0b"      # 주의/외주
DANGER = "#ef4444"      # 부재고
SUCCESS = "#10b981"     # 고용
NEUTRAL = "#64748b"     # 수요(레퍼런스)

PALETTE = ["#2563eb", "#60a5fa", "#93c5fd", "#f59e0b",
           "#ef4444", "#10b981", "#a855f7", "#64748b"]

st.markdown(
    """
    <style>
        /* 전체 여백 */
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
        /* 사이드바 */
        section[data-testid="stSidebar"] {background:#f8fafc; border-right:1px solid #e2e8f0;}
        /* KPI 카드 */
        [data-testid="stMetric"] {
            background:#ffffff;
            padding: 14px 18px;
            border-radius: 10px;
            border:1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }
        [data-testid="stMetricValue"] {color:#1e3a8a; font-weight:700;}
        [data-testid="stMetricLabel"] {color:#475569;}
        /* 헤더 */
        h1 {color:#0f172a; font-weight:800; letter-spacing:-0.02em;}
        h2 {color:#1e3a8a; font-weight:700; border-bottom:2px solid #dbeafe; padding-bottom:.35rem; margin-top:1.5rem;}
        h3 {color:#1e40af; font-weight:600;}
        /* 버튼 */
        .stButton>button, .stDownloadButton>button {
            background:#2563eb; color:white; border:0; border-radius:8px;
            padding:.55rem 1.1rem; font-weight:600;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {background:#1d4ed8;}
        /* 캡션 / 헬프 */
        .small-note {color:#64748b; font-size:.85rem;}
        /* 데이터 테이블 헤더 */
        .stDataFrame thead tr th {background:#f1f5f9 !important; color:#1e3a8a !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. 최적화 함수
# ============================================================
@st.cache_data(show_spinner=False)
def solve_app(demand, params, strategy="optimal", integer=False):
    """
    APP LP/IP 최적화.

    Parameters
    ----------
    demand   : list[float]   월별 수요
    params   : dict           비용·능력·초기조건 파라미터
    strategy : 'optimal' | 'level' | 'chase'
    integer  : True면 IP, False면 LP
    """
    n = len(demand)
    T = list(range(1, n + 1))
    T_full = list(range(0, n + 1))
    cat = pulp.LpInteger if integer else pulp.LpContinuous

    m = pulp.LpProblem(f"APP_{strategy}", pulp.LpMinimize)

    # 결정변수
    W = pulp.LpVariable.dicts("W", T_full, lowBound=0, cat=cat)
    H = pulp.LpVariable.dicts("H", T, lowBound=0, cat=cat)
    L = pulp.LpVariable.dicts("L", T, lowBound=0, cat=cat)
    P = pulp.LpVariable.dicts("P", T, lowBound=0, cat=cat)
    I = pulp.LpVariable.dicts("I", T_full, lowBound=0, cat=cat)
    S = pulp.LpVariable.dicts("S", T_full, lowBound=0, cat=cat)
    C = pulp.LpVariable.dicts("C", T, lowBound=0, cat=cat)
    O = pulp.LpVariable.dicts("O", T, lowBound=0)  # 시간은 연속

    # 파생 계수
    reg_labor = params["reg_wage"] * params["work_hours"] * params["work_days"]
    units_per_worker = params["work_days"] * params["work_hours"] / params["std_time"]
    units_per_ot_hr = 1.0 / params["std_time"]

    # 목적함수
    m += pulp.lpSum(
        reg_labor * W[t]
        + params["ot_wage"] * O[t]
        + params["hire_cost"] * H[t]
        + params["layoff_cost"] * L[t]
        + params["inv_cost"] * I[t]
        + params["backorder_cost"] * S[t]
        + params["material_cost"] * P[t]
        + params["subcontract_cost"] * C[t]
        for t in T
    )

    # 초기/최종 조건
    m += W[0] == params["init_workforce"]
    m += I[0] == params["init_inventory"]
    m += S[0] == 0
    m += I[n] >= params["final_inventory"]
    m += S[n] == 0

    # 시점별 제약
    for t in T:
        m += W[t] == W[t - 1] + H[t] - L[t]                                    # 인력 균형
        m += P[t] <= units_per_worker * W[t] + units_per_ot_hr * O[t]          # 생산능력
        m += I[t] == I[t - 1] + P[t] + C[t] - demand[t - 1] - S[t - 1] + S[t]  # 재고 균형
        m += O[t] <= params["ot_limit"] * W[t]                                 # 초과근무 한도

    # ── 전략별 추가 제약 ─────────────────────────────────────
    if strategy == "level":
        # 평준화: 인력 변화 금지 (초기 인력 그대로 유지)
        for t in T:
            m += H[t] == 0
            m += L[t] == 0
    elif strategy == "chase":
        # 추종: 재고를 최종재고 수준 이하로 유지 → 생산이 수요를 추종
        for t in T:
            m += I[t] <= params["final_inventory"]

    # 풀이
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=20)
    m.solve(solver)
    if pulp.LpStatus[m.status] != "Optimal":
        return None

    val = lambda x: float(pulp.value(x) or 0)
    res = {
        "W": [val(W[t]) for t in T_full],
        "H": [0.0] + [val(H[t]) for t in T],
        "L": [0.0] + [val(L[t]) for t in T],
        "P": [0.0] + [val(P[t]) for t in T],
        "I": [val(I[t]) for t in T_full],
        "S": [val(S[t]) for t in T_full],
        "C": [0.0] + [val(C[t]) for t in T],
        "O": [0.0] + [val(O[t]) for t in T],
        "demand": [0.0] + list(demand),
        "total_cost": float(pulp.value(m.objective)),
        "status": pulp.LpStatus[m.status],
    }

    # 비용 항목 분해
    res["cost_breakdown"] = {
        "정규임금":   sum(reg_labor * res["W"][t] for t in T),
        "초과근무":   sum(params["ot_wage"] * res["O"][t] for t in T),
        "고용비용":   sum(params["hire_cost"] * res["H"][t] for t in T),
        "해고비용":   sum(params["layoff_cost"] * res["L"][t] for t in T),
        "재고유지":   sum(params["inv_cost"] * res["I"][t] for t in T),
        "재고부족":   sum(params["backorder_cost"] * res["S"][t] for t in T),
        "재료비":     sum(params["material_cost"] * res["P"][t] for t in T),
        "하청추가비": sum(params["subcontract_cost"] * res["C"][t] for t in T),
    }
    return res


# ============================================================
# 3. 사이드바 입력
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ 입력 파라미터")
    st.caption("값을 변경하면 모든 차트가 즉시 다시 계산됩니다.")

    # ── 계획 기간 & 수요 ────────────────────────────────────
    with st.expander("📅 계획 기간 · 수요", expanded=True):
        n_periods = st.slider("계획 기간 (개월)", 3, 24, 6)
        st.markdown('<p class="small-note">월별 수요(개) — 표를 직접 편집하세요</p>', unsafe_allow_html=True)
        # 기본값(1~6월) + 외삽
        default = [1600, 3000, 3200, 3800, 2200, 2200]
        if n_periods <= 6:
            base = default[:n_periods]
        else:
            base = default + [int(np.mean(default))] * (n_periods - 6)
        demand_df = pd.DataFrame(
            {"월": [f"{i+1}월" for i in range(n_periods)], "수요(개)": base}
        )
        edited = st.data_editor(
            demand_df,
            hide_index=True,
            use_container_width=True,
            key=f"demand_editor_{n_periods}",
            column_config={
                "월": st.column_config.TextColumn(disabled=True),
                "수요(개)": st.column_config.NumberColumn(min_value=0, step=100),
            },
        )
        demand = edited["수요(개)"].astype(float).tolist()

    # ── 초기조건 ────────────────────────────────────────────
    with st.expander("📦 초기 / 종료 조건", expanded=True):
        init_inventory = st.number_input("초기 재고 $I_0$ (개)", 0, 1_000_000, 1000, 100)
        final_inventory = st.number_input("최종 재고 $I_T$ ≥ (개)", 0, 1_000_000, 500, 100)
        init_workforce = st.number_input("초기 작업자 $W_0$ (명)", 0, 10_000, 80, 1)

    # ── 생산 능력 ───────────────────────────────────────────
    with st.expander("⏱️ 생산 능력", expanded=False):
        work_days = st.number_input("월 작업일 (일)", 1, 31, 20)
        work_hours = st.number_input("일 작업시간 (시)", 1, 24, 8)
        std_time = st.number_input("표준 작업시간 (시간/개)", 0.1, 100.0, 4.0, 0.1)
        ot_limit = st.number_input("초과근무 제한 (시간/인/월)", 0, 200, 10)

    # ── 비용 (천원) ─────────────────────────────────────────
    with st.expander("💰 비용 (단위: 천원)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            reg_wage = st.number_input("정규임금/시간", 0.0, 100.0, 4.0, 0.5)
            hire_cost = st.number_input("고용비용/인", 0, 10_000, 300)
            inv_cost = st.number_input("재고유지비/개·월", 0.0, 100.0, 2.0, 0.5)
            material_cost = st.number_input("재료비/개", 0.0, 10_000.0, 10.0, 1.0)
        with c2:
            ot_wage = st.number_input("초과근무임금/시간", 0.0, 100.0, 6.0, 0.5)
            layoff_cost = st.number_input("해고비용/인", 0, 10_000, 500)
            backorder_cost = st.number_input("부재고비용/개·월", 0.0, 100.0, 5.0, 0.5)
            subcontract_cost = st.number_input("하청 추가비/개", 0.0, 10_000.0, 30.0, 1.0)

    # ── 전략 / 모드 ─────────────────────────────────────────
    with st.expander("🎯 전략 · 모델 옵션", expanded=True):
        strategy_label = {
            "optimal": "🤖 LP 최적화 (혼합 전략)",
            "level":   "🟦 평준화 (Level) — 인력 일정",
            "chase":   "🟧 추종 (Chase) — 재고 최소",
        }
        strategy = st.radio(
            "주 전략",
            options=list(strategy_label.keys()),
            format_func=lambda x: strategy_label[x],
            index=0,
        )
        integer_mode = st.checkbox(
            "정수 계획 (IP 모드)",
            value=False,
            help="작업자/생산량을 정수로 강제합니다.",
        )
        compare_all = st.checkbox("📊 전략별 비교 패널 표시", value=True)

# 입력 패키징
params = dict(
    reg_wage=reg_wage, ot_wage=ot_wage,
    hire_cost=hire_cost, layoff_cost=layoff_cost,
    inv_cost=inv_cost, backorder_cost=backorder_cost,
    material_cost=material_cost, subcontract_cost=subcontract_cost,
    work_days=work_days, work_hours=work_hours,
    std_time=std_time, ot_limit=ot_limit,
    init_workforce=init_workforce, init_inventory=init_inventory,
    final_inventory=final_inventory,
)


# ============================================================
# 4. 메인 헤더
# ============================================================
st.markdown("# 🏭 총괄생산계획 최적화 대시보드")
st.markdown(
    f"<p class='small-note'>"
    f"원예장비 제조업체 사례 · LP/IP 기반 의사결정 지원 · "
    f"전략 = <b>{strategy_label[strategy]}</b> · 모드 = <b>{'정수(IP)' if integer_mode else '연속(LP)'}</b>"
    f"</p>",
    unsafe_allow_html=True,
)

# ============================================================
# 5. 최적화 실행
# ============================================================
with st.spinner("최적화 풀이 중..."):
    result = solve_app(demand, params, strategy=strategy, integer=integer_mode)

if result is None:
    st.error(
        "⚠️ **해를 찾을 수 없습니다 (Infeasible).**\n\n"
        "선택한 전략과 파라미터로는 모든 제약을 만족할 수 없습니다. "
        "예: '평준화' 전략에서 초기 인력만으로 수요를 감당할 수 없거나, "
        "'추종' 전략에서 최종재고 목표가 너무 높을 때 발생합니다.\n\n"
        "▸ 초기 작업자 수를 늘리거나\n▸ 최종재고 목표를 낮추거나\n▸ 다른 전략을 선택해보세요."
    )
    st.stop()

# 시간축 라벨
months_full = [f"{i}월" for i in range(0, n_periods + 1)]  # 0~T
months_p = [f"{i+1}월" for i in range(n_periods)]          # 1~T
T_idx = list(range(1, n_periods + 1))


# ============================================================
# 6. KPI 카드
# ============================================================
total_cost_thou = result["total_cost"]
total_demand = sum(demand)
total_prod = sum(result["P"][1:])
total_subc = sum(result["C"][1:])
avg_workers = float(np.mean(result["W"][1:]))
total_hire = sum(result["H"][1:])
total_layoff = sum(result["L"][1:])
total_ot = sum(result["O"][1:])
total_short = sum(result["S"][1:])

# 매출/이익(참고): 판가 40천원/개 가정
revenue = 40.0 * total_demand  # 천원 (단가 40천원/개 가정)
profit = revenue - total_cost_thou

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 총 비용", f"{total_cost_thou:,.0f} 천원",
          f"₩ {total_cost_thou*1000:,.0f}")
k2.metric("📈 추정 이익", f"{profit:,.0f} 천원",
          f"매출 가정 {revenue:,.0f} 천원", delta_color="normal")
k3.metric("📦 수요 / 공급",
          f"{int(total_demand):,} / {total_prod + total_subc:,.0f} 개",
          f"외주 {int(total_subc):,} · 부재고 {int(total_short):,}")
k4.metric("👥 평균 인력", f"{avg_workers:.1f} 명",
          f"고용 {total_hire:.0f} · 해고 {total_layoff:.0f}")
k5.metric("⏰ 총 초과근무", f"{total_ot:,.0f} 시간",
          f"평균 {total_ot / n_periods:.1f} 시간/월")

# 진단 메시지
diag = []
if total_short > 0:
    diag.append(f"⚠️ 부재고 발생: {total_short:.0f}개 — 수요 일부가 지연 충족됩니다.")
if total_subc > 0:
    diag.append(f"📤 외주 사용: {total_subc:.0f}개 — 자체 생산능력을 초과한 분량입니다.")
if total_ot / max(avg_workers * n_periods, 1) > 8:
    diag.append("⚠️ 초과근무 강도가 높음 — 인력 충원 검토를 권장합니다.")
if not diag:
    diag.append("✅ 정상 운영 — 부재고/외주 없이 자체 정규생산으로 수요를 충족합니다.")
st.info("  ".join(diag))


# ============================================================
# 7. 핵심 차트 4개 (2×2 그리드)
# ============================================================
st.markdown("## 📊 운영 대시보드")

LAYOUT_KW = dict(
    template="plotly_white",
    height=380,
    margin=dict(l=10, r=10, t=40, b=60),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    font=dict(family="-apple-system, 'Segoe UI', sans-serif", size=12, color="#0f172a"),
)

row1c1, row1c2 = st.columns(2)

# (1) 수요 vs 생산
with row1c1:
    st.markdown("#### 📈 수요 대비 공급")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months_p, y=demand, name="수요 (D)",
        marker_color=NEUTRAL, opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=months_p, y=result["P"][1:],
        name="자체 생산 (P)", mode="lines+markers",
        line=dict(color=PRIMARY, width=3), marker=dict(size=9),
    ))
    total_supply = [result["P"][t] + result["C"][t] for t in T_idx]
    fig.add_trace(go.Scatter(
        x=months_p, y=total_supply,
        name="총 공급 (P+C)", mode="lines+markers",
        line=dict(color=ACCENT, width=2, dash="dash"), marker=dict(size=7),
    ))
    fig.update_layout(yaxis_title="수량 (개)", **LAYOUT_KW)
    st.plotly_chart(fig, use_container_width=True)

# (2) 재고 / 부재고
with row1c2:
    st.markdown("#### 📦 재고 / 부재고 추이")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months_full, y=result["I"], name="재고 (I)",
        fill="tozeroy", line=dict(color=PRIMARY, width=2.5),
        fillcolor="rgba(37,99,235,0.18)",
    ))
    if any(s > 0.01 for s in result["S"]):
        fig.add_trace(go.Scatter(
            x=months_full, y=[-s for s in result["S"]],
            name="부재고 (S)", fill="tozeroy",
            line=dict(color=DANGER, width=2.5),
            fillcolor="rgba(239,68,68,0.20)",
        ))
    fig.add_hline(
        y=params["final_inventory"], line_dash="dot", line_color=ACCENT,
        annotation_text=f"최종재고 목표 {params['final_inventory']}",
        annotation_position="top right",
    )
    fig.update_layout(yaxis_title="수량 (개)", **LAYOUT_KW)
    st.plotly_chart(fig, use_container_width=True)

row2c1, row2c2 = st.columns(2)

# (3) 공급 구성 (정규/초과/외주) — Stacked bar
with row2c1:
    st.markdown("#### 🏭 공급 원천별 구성")
    units_per_worker = work_days * work_hours / std_time
    ot_units = [result["O"][t] / std_time for t in T_idx]
    reg_units = [max(0.0, result["P"][t] - ot_units[i]) for i, t in enumerate(T_idx)]
    sub_units = [result["C"][t] for t in T_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months_p, y=reg_units,
                         name="정규시간 생산", marker_color=PRIMARY))
    fig.add_trace(go.Bar(x=months_p, y=ot_units,
                         name="초과시간 생산", marker_color=PRIMARY_LIGHT))
    fig.add_trace(go.Bar(x=months_p, y=sub_units,
                         name="외주 (Subcontract)", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=months_p, y=demand,
                             name="수요(참고)", mode="lines+markers",
                             line=dict(color=DANGER, width=2, dash="dot"),
                             marker=dict(size=6)))
    fig.update_layout(barmode="stack", yaxis_title="수량 (개)", **LAYOUT_KW)
    st.plotly_chart(fig, use_container_width=True)

# (4) 비용 구성
with row2c2:
    st.markdown("#### 💰 비용 항목별 비중")
    cb = {k: v for k, v in result["cost_breakdown"].items() if v > 0.01}
    fig = go.Figure(data=[go.Pie(
        labels=list(cb.keys()),
        values=list(cb.values()),
        hole=0.55,
        marker=dict(colors=PALETTE, line=dict(color="white", width=2)),
        textposition="outside",
        textinfo="label+percent",
    )])
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{total_cost_thou:,.0f}</b><br><span style='font-size:11px;color:#64748b'>천원</span>",
            x=0.5, y=0.5, font=dict(size=18, color=PRIMARY), showarrow=False,
        )],
        height=380, margin=dict(l=10, r=10, t=40, b=20),
        showlegend=False,
        font=dict(family="-apple-system, 'Segoe UI', sans-serif", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 8. 인력 운영 차트 (full width)
# ============================================================
st.markdown("#### 👥 인력 운영 계획")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=months_full, y=result["W"], name="작업자 수 (W)",
    mode="lines+markers",
    line=dict(color=PRIMARY, width=3),
    marker=dict(size=11, line=dict(color="white", width=2)),
))
fig.add_trace(go.Bar(
    x=months_p, y=result["H"][1:], name="신규 고용 (H)",
    marker_color=SUCCESS, opacity=0.85, yaxis="y2",
))
fig.add_trace(go.Bar(
    x=months_p, y=[-l for l in result["L"][1:]], name="해고 (L)",
    marker_color=DANGER, opacity=0.85, yaxis="y2",
))
fig.update_layout(
    template="plotly_white", height=350,
    margin=dict(l=10, r=10, t=20, b=60),
    hovermode="x unified",
    yaxis=dict(title="작업자 수 (명)", side="left"),
    yaxis2=dict(title="고용/해고 (명)", overlaying="y", side="right",
                showgrid=False, zeroline=True, zerolinecolor="#cbd5e1"),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    font=dict(family="-apple-system, 'Segoe UI', sans-serif", size=12),
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 9. 전략 비교
# ============================================================
if compare_all:
    st.markdown("## 🔍 전략별 비교 분석")

    strategies_meta = {
        "optimal": ("🤖 LP 최적화", PRIMARY),
        "level":   ("🟦 평준화",     PRIMARY_LIGHT),
        "chase":   ("🟧 추종",       ACCENT),
    }
    comparison = {}
    for s, (name, _) in strategies_meta.items():
        r = solve_app(demand, params, strategy=s, integer=integer_mode)
        comparison[s] = (name, r)

    # 9-1) 비용 요약 테이블
    rows = []
    for s, (name, r) in comparison.items():
        if r is None:
            rows.append({"전략": name, "총비용": "❌ 해 없음 (Infeasible)"})
            continue
        b = r["cost_breakdown"]
        rows.append({
            "전략": name + (" ★" if s == strategy else ""),
            "총비용 (천원)": round(r["total_cost"]),
            "정규임금": round(b["정규임금"]),
            "초과근무": round(b["초과근무"]),
            "고용+해고": round(b["고용비용"] + b["해고비용"]),
            "재고유지+부재고": round(b["재고유지"] + b["재고부족"]),
            "외주 추가비": round(b["하청추가비"]),
            "재료비": round(b["재료비"]),
            "총 부재고(개)": round(sum(r["S"][1:])),
            "총 외주(개)": round(sum(r["C"][1:])),
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # 9-2) 추천 전략 자동 선택 (실현가능한 것 중 최저 비용)
    feasible = {s: r for s, (_, r) in comparison.items() if r is not None}
    if feasible:
        best_s, best_r = min(feasible.items(), key=lambda x: x[1]["total_cost"])
        best_name = strategies_meta[best_s][0]
        delta_vs_curr = (
            comparison[strategy][1]["total_cost"] - best_r["total_cost"]
            if comparison[strategy][1] is not None else None
        )
        if best_s == strategy:
            st.success(
                f"💡 **추천 전략: {best_name}** (현재 선택과 동일) — "
                f"가능한 전략 중 비용이 가장 낮습니다 ({best_r['total_cost']:,.0f} 천원)."
            )
        else:
            msg = f"💡 **추천 전략: {best_name}** — 총비용 {best_r['total_cost']:,.0f} 천원"
            if delta_vs_curr is not None and delta_vs_curr > 0:
                msg += f" (현재 선택 대비 **{delta_vs_curr:,.0f} 천원 절감**)"
            st.success(msg)

    # 9-3) 전략별 비용 항목 비교 (그룹 바)
    fig = go.Figure()
    items = ["정규임금", "초과근무", "고용비용", "해고비용",
             "재고유지", "재고부족", "재료비", "하청추가비"]
    for s, (name, r) in comparison.items():
        if r is None:
            continue
        fig.add_trace(go.Bar(
            name=name, x=items,
            y=[r["cost_breakdown"][k] for k in items],
            marker_color=strategies_meta[s][1],
        ))
    fig.update_layout(
        template="plotly_white", height=420, barmode="group",
        margin=dict(l=10, r=10, t=20, b=40),
        yaxis_title="비용 (천원)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center"),
        font=dict(family="-apple-system, 'Segoe UI', sans-serif", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 9-4) 전략별 인력 / 재고 추이
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.markdown("**전략별 작업자 수 추이**")
        fig = go.Figure()
        for s, (name, r) in comparison.items():
            if r is None: continue
            fig.add_trace(go.Scatter(
                x=months_full, y=r["W"], name=name,
                mode="lines+markers",
                line=dict(width=2.5, color=strategies_meta[s][1]),
            ))
        fig.update_layout(
            template="plotly_white", height=320,
            margin=dict(l=10, r=10, t=10, b=40),
            yaxis_title="작업자 수 (명)",
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with cmp2:
        st.markdown("**전략별 재고(±부재고) 추이**")
        fig = go.Figure()
        for s, (name, r) in comparison.items():
            if r is None: continue
            net_inv = [r["I"][t] - r["S"][t] for t in range(len(r["I"]))]
            fig.add_trace(go.Scatter(
                x=months_full, y=net_inv, name=name,
                mode="lines+markers",
                line=dict(width=2.5, color=strategies_meta[s][1]),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        fig.update_layout(
            template="plotly_white", height=320,
            margin=dict(l=10, r=10, t=10, b=40),
            yaxis_title="순재고 (개)",
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. 상세 데이터 + CSV 다운로드
# ============================================================
st.markdown("## 📋 상세 결과")

result_df = pd.DataFrame({
    "월": months_full,
    "수요 D": [0] + [int(d) for d in demand],
    "작업자 W": [round(x, 2) for x in result["W"]],
    "고용 H": [round(x, 2) for x in result["H"]],
    "해고 L": [round(x, 2) for x in result["L"]],
    "생산 P": [round(x, 2) for x in result["P"]],
    "재고 I": [round(x, 2) for x in result["I"]],
    "부재고 S": [round(x, 2) for x in result["S"]],
    "외주 C": [round(x, 2) for x in result["C"]],
    "초과시간 O": [round(x, 2) for x in result["O"]],
})
st.dataframe(result_df, hide_index=True, use_container_width=True)

dl1, dl2 = st.columns([1, 5])
with dl1:
    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"production_plan_{strategy}.csv",
        mime="text/csv",
    )


# ============================================================
# 11. 수리적 모델 / 입력 요약 (Expander)
# ============================================================
with st.expander("📐 수리적 모델 보기 (LP / IP 정식)", expanded=False):
    st.markdown(
        r"""
##### 결정변수
- $W_t$: $t$월 작업자 수 / $H_t$: 신규 고용 / $L_t$: 해고
- $P_t$: 생산량 / $I_t$: 기말 재고 / $S_t$: 부재고
- $C_t$: 외주량 / $O_t$: 초과근무 시간

##### 목적함수 (단위: 천원)
$$\min Z = \sum_{t}\bigl(640\,W_t + 6\,O_t + 300\,H_t + 500\,L_t
+ 2\,I_t + 5\,S_t + 10\,P_t + 30\,C_t\bigr)$$

##### 제약조건
- **인력 균형**:  $W_t = W_{t-1} + H_t - L_t$
- **생산능력**:  $P_t \le 40\,W_t + O_t/4$
- **재고 균형**:  $I_t = I_{t-1} + P_t + C_t - D_t - S_{t-1} + S_t$
- **초과근무 한도**:  $O_t \le 10\,W_t$
- **초기/최종 조건**:  $W_0=80,\; I_0=1000,\; I_T \ge 500,\; S_0=S_T=0$
- **비음수**:  모든 변수 $\ge 0$ (IP 모드일 경우 정수)

##### 전략별 추가 제약
| 전략 | 추가 제약 | 의미 |
|---|---|---|
| 🤖 LP 최적화 | (없음) | 모든 옵션을 동시에 고려해 비용 최소화 |
| 🟦 평준화 | $H_t = L_t = 0,\; \forall t$ | 인력 변화 없이 일정한 작업자로 운영 |
| 🟧 추종 | $I_t \le I_T^{target},\; \forall t$ | 재고를 최소화 → 생산이 수요를 추종 |
        """
    )

with st.expander("ℹ️ 입력 파라미터 요약", expanded=False):
    summary_df = pd.DataFrame({
        "항목": [
            "계획 기간", "초기 작업자", "초기 재고", "최종 재고 ≥",
            "월 작업일", "일 작업시간", "표준작업시간/개", "초과근무 한도",
            "정규임금", "초과근무임금", "고용비", "해고비",
            "재고유지비", "부재고비용", "재료비", "하청 추가비",
        ],
        "값": [
            f"{n_periods} 개월", f"{init_workforce} 명",
            f"{init_inventory:,} 개", f"{final_inventory:,} 개",
            f"{work_days} 일/월", f"{work_hours} 시간/일",
            f"{std_time} 시간/개", f"{ot_limit} 시간/인·월",
            f"{reg_wage} 천원/시", f"{ot_wage} 천원/시",
            f"{hire_cost:,} 천원/인", f"{layoff_cost:,} 천원/인",
            f"{inv_cost} 천원/개·월", f"{backorder_cost} 천원/개·월",
            f"{material_cost} 천원/개", f"{subcontract_cost} 천원/개",
        ],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ============================================================
# 12. 푸터
# ============================================================
st.markdown(
    """
    <div style='text-align:center; padding:1.5rem 0 .5rem 0; color:#94a3b8; font-size:.85rem;'>
        © 2026 총괄생산계획 최적화 대시보드 · Streamlit + PuLP + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
