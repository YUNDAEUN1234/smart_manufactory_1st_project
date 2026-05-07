"""
총괄생산계획 (Aggregate Production Planning) 최적화 대시보드
------------------------------------------------------------
원예장비 제조업체 사례를 기반으로 한 인터랙티브 의사결정 도구.
수요·비용·생산능력 파라미터를 실시간으로 조정하며
LP / IP 최적해를 즉시 확인할 수 있습니다.

스택: Streamlit · PuLP (CBC 솔버) · Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import streamlit as st


# ────────────────────────────────────────────────────────────
# 상수: 디자인 토큰
# ────────────────────────────────────────────────────────────

PRIMARY       = "#2563eb"   # 메인 블루
PRIMARY_LIGHT = "#60a5fa"   # 연블루 (초과근무)
ACCENT        = "#f59e0b"   # 주황 (외주 / 경고)
DANGER        = "#ef4444"   # 빨강 (부재고 / 해고)
SUCCESS       = "#10b981"   # 초록 (고용)
NEUTRAL       = "#64748b"   # 회색 (수요 참고 막대)

PALETTE = [PRIMARY, PRIMARY_LIGHT, "#93c5fd", ACCENT,
           DANGER, SUCCESS, "#a855f7", NEUTRAL]

FONT_FAMILY = "-apple-system, 'Segoe UI', sans-serif"




# ────────────────────────────────────────────────────────────
# 페이지 설정 & 글로벌 CSS
# ────────────────────────────────────────────────────────────

def setup_page() -> None:
    """Streamlit 페이지 기본 설정 및 글로벌 스타일 주입."""
    st.set_page_config(
        page_title="총괄생산계획 최적화 대시보드",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }
            section[data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }
            [data-testid="stMetric"] {
                background: #ffffff;
                padding: 14px 18px;
                border-radius: 10px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 2px rgba(15,23,42,0.04);
            }
            [data-testid="stMetricValue"] { color: #1e3a8a; font-weight: 700; }
            [data-testid="stMetricLabel"] { color: #475569; }
            h1 { color: #0f172a; font-weight: 800; letter-spacing: -0.02em; }
            h2 {
                color: #1e3a8a; font-weight: 700;
                border-bottom: 2px solid #dbeafe;
                padding-bottom: .35rem; margin-top: 1.5rem;
            }
            h3 { color: #1e40af; font-weight: 600; }
            .stButton>button, .stDownloadButton>button {
                background: #2563eb; color: white; border: 0;
                border-radius: 8px; padding: .55rem 1.1rem; font-weight: 600;
            }
            .stButton>button:hover,
            .stDownloadButton>button:hover { background: #1d4ed8; }
            .small-note { color: #64748b; font-size: .85rem; }
            .stDataFrame thead tr th {
                background: #f1f5f9 !important;
                color: #1e3a8a !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────
# 최적화 엔진
# ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def solve_app(
    demand: tuple[float, ...],
    params: dict,
    strategy: str = "optimal",
    integer: bool = False,
) -> dict | None:
    """
    총괄생산계획 LP/IP 최적화.

    Parameters
    ----------
    demand   : 월별 수요 (tuple — st.cache_data의 hash key로 사용)
    params   : 비용·능력·초기조건 딕셔너리
    strategy : 'optimal' | 'level' | 'chase'
    integer  : True → 정수 계획(IP), False → 연속 LP

    Returns
    -------
    결과 딕셔너리 또는 None (infeasible)
    """
    n      = len(demand)
    T      = list(range(1, n + 1))   # 1 … n
    T_full = list(range(0, n + 1))   # 0 … n  (초기 포함)
    cat    = pulp.LpInteger if integer else pulp.LpContinuous

    model = pulp.LpProblem(f"APP_{strategy}", pulp.LpMinimize)

    # ── 결정변수 ────────────────────────────────────────────
    W = pulp.LpVariable.dicts("W", T_full, lowBound=0, cat=cat)   # 작업자 수
    H = pulp.LpVariable.dicts("H", T,      lowBound=0, cat=cat)   # 신규 고용
    L = pulp.LpVariable.dicts("L", T,      lowBound=0, cat=cat)   # 해고
    P = pulp.LpVariable.dicts("P", T,      lowBound=0, cat=cat)   # 생산량
    I = pulp.LpVariable.dicts("I", T_full, lowBound=0, cat=cat)   # 기말 재고
    S = pulp.LpVariable.dicts("S", T_full, lowBound=0, cat=cat)   # 부재고
    C = pulp.LpVariable.dicts("C", T,      lowBound=0, cat=cat)   # 외주량
    O = pulp.LpVariable.dicts("O", T,      lowBound=0)            # 초과근무 시간 (항상 연속)

    # ── 파생 계수 ───────────────────────────────────────────
    reg_labor_per_worker = (                                       # 1인당 월 정규임금 (천원)
        params["reg_wage"] * params["work_hours"] * params["work_days"]
    )
    units_per_worker = (                                           # 1인 정규시간 생산량 (개/월)
        params["work_days"] * params["work_hours"] / params["std_time"]
    )
    units_per_ot_hr = 1.0 / params["std_time"]                    # 초과근무 1시간당 생산량

    # ── 목적함수: 총비용 최소화 ─────────────────────────────
    model += pulp.lpSum(
        reg_labor_per_worker         * W[t]
        + params["ot_wage"]          * O[t]
        + params["hire_cost"]        * H[t]
        + params["layoff_cost"]      * L[t]
        + params["inv_cost"]         * I[t]
        + params["backorder_cost"]   * S[t]
        + params["material_cost"]    * P[t]
        + params["subcontract_cost"] * C[t]
        for t in T
    )

    # ── 초기 / 최종 조건 ────────────────────────────────────
    model += W[0] == params["init_workforce"]
    model += I[0] == params["init_inventory"]
    model += S[0] == 0
    model += I[n] >= params["final_inventory"]
    model += S[n] == 0

    # ── 시점별 제약 ─────────────────────────────────────────
    for t in T:
        model += W[t] == W[t-1] + H[t] - L[t]                                # 인력 균형
        model += P[t] <= units_per_worker * W[t] + units_per_ot_hr * O[t]    # 생산능력 상한
        model += I[t] == I[t-1] + P[t] + C[t] - demand[t-1] - S[t-1] + S[t] # 재고 균형
        model += O[t] <= params["ot_limit"] * W[t]                            # 초과근무 한도

    # ── 전략별 추가 제약 ────────────────────────────────────
    if strategy == "level":
        # 평준화: 고용·해고 금지 → 인력 수준 고정
        for t in T:
            model += H[t] == 0
            model += L[t] == 0
    elif strategy == "chase":
        # 추종: 재고를 최종재고 목표 이하로 억제 → 생산이 수요를 따라감
        for t in T:
            model += I[t] <= params["final_inventory"]

    # ── 풀이 ────────────────────────────────────────────────
    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    val = lambda x: float(pulp.value(x) or 0)

    result = {
        "W":          [val(W[t]) for t in T_full],
        "H":          [0.0] + [val(H[t]) for t in T],
        "L":          [0.0] + [val(L[t]) for t in T],
        "P":          [0.0] + [val(P[t]) for t in T],
        "I":          [val(I[t]) for t in T_full],
        "S":          [val(S[t]) for t in T_full],
        "C":          [0.0] + [val(C[t]) for t in T],
        "O":          [0.0] + [val(O[t]) for t in T],
        "demand":     [0.0] + list(demand),
        "total_cost": float(pulp.value(model.objective)),
        "status":     pulp.LpStatus[model.status],
    }

    # 비용 항목별 분해 (도넛 차트 · 비교 테이블용)
    result["cost_breakdown"] = {
        "정규임금":   sum(reg_labor_per_worker       * result["W"][t] for t in T),
        "초과근무":   sum(params["ot_wage"]           * result["O"][t] for t in T),
        "고용비용":   sum(params["hire_cost"]         * result["H"][t] for t in T),
        "해고비용":   sum(params["layoff_cost"]       * result["L"][t] for t in T),
        "재고유지":   sum(params["inv_cost"]          * result["I"][t] for t in T),
        "재고부족":   sum(params["backorder_cost"]    * result["S"][t] for t in T),
        "재료비":     sum(params["material_cost"]     * result["P"][t] for t in T),
        "하청추가비": sum(params["subcontract_cost"]  * result["C"][t] for t in T),
    }
    return result


# ────────────────────────────────────────────────────────────
# 사이드바 입력
# ────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[list[float], dict, bool]:
    """
    사이드바 입력 패널을 렌더링하고 입력값을 반환.

    Returns
    -------
    demand       : 월별 수요 리스트
    params       : 최적화 파라미터 딕셔너리
    integer_mode : IP 모드 여부
    """
    with st.sidebar:
        st.markdown("## ⚙️ 입력 파라미터")
        st.caption("값을 변경하면 모든 차트가 즉시 다시 계산됩니다.")

        # ── 계획 기간 & 수요 ──────────────────────────────
        with st.expander("📅 계획 기간 · 수요", expanded=True):
            n_periods = st.slider("계획 기간 (개월)", 3, 24, 6)
            st.markdown(
                '<p class="small-note">월별 수요(개) — 표를 직접 편집하세요</p>',
                unsafe_allow_html=True,
            )
            # 기본값 6개월치, 초과 기간은 평균으로 외삽
            default_demand = [1600, 3000, 3200, 3800, 2200, 2200]
            base = (
                default_demand[:n_periods]
                if n_periods <= 6
                else default_demand + [int(np.mean(default_demand))] * (n_periods - 6)
            )
            demand_df = pd.DataFrame(
                {"월": [f"{i+1}월" for i in range(n_periods)], "수요(개)": base}
            )
            edited = st.data_editor(
                demand_df,
                hide_index=True,
                use_container_width=True,
                key=f"demand_editor_{n_periods}",
                column_config={
                    "월":       st.column_config.TextColumn(disabled=True),
                    "수요(개)": st.column_config.NumberColumn(min_value=0, step=100),
                },
            )
            demand = edited["수요(개)"].astype(float).tolist()

        # ── 초기 / 종료 조건 ──────────────────────────────
        with st.expander("📦 초기 / 종료 조건", expanded=True):
            init_inventory  = st.number_input("초기 재고 $I_0$ (개)",   0, 1_000_000, 1000, 100)
            final_inventory = st.number_input("최종 재고 $I_T$ ≥ (개)", 0, 1_000_000,  500, 100)
            init_workforce  = st.number_input("초기 작업자 $W_0$ (명)", 0,    10_000,   80,   1)

        # ── 생산 능력 ─────────────────────────────────────
        with st.expander("⏱️ 생산 능력", expanded=False):
            work_days  = st.number_input("월 작업일 (일)",             1,  31, 20)
            work_hours = st.number_input("일 작업시간 (시)",           1,  24,  8)
            std_time   = st.number_input("표준 작업시간 (시간/개)",  0.1, 100.0, 4.0, 0.1)
            ot_limit   = st.number_input("초과근무 제한 (시간/인/월)", 0, 200,   10)

        # ── 비용 (단위: 천원) ─────────────────────────────
        with st.expander("💰 비용 (단위: 천원)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                reg_wage      = st.number_input("정규임금/시간",    0.0, 100.0,    4.0, 0.5)
                hire_cost     = st.number_input("고용비용/인",      0,  10_000,    300)
                inv_cost      = st.number_input("재고유지비/개·월", 0.0, 100.0,    2.0, 0.5)
                material_cost = st.number_input("재료비/개",        0.0, 10_000.0, 10.0, 1.0)
            with c2:
                ot_wage          = st.number_input("초과근무임금/시간", 0.0, 100.0,    6.0, 0.5)
                layoff_cost      = st.number_input("해고비용/인",       0,  10_000,    500)
                backorder_cost   = st.number_input("부재고비용/개·월",  0.0, 100.0,    5.0, 0.5)
                subcontract_cost = st.number_input("하청 추가비/개",    0.0, 10_000.0, 30.0, 1.0)

        # ── 전략 · 모델 옵션 ──────────────────────────────
        with st.expander("🎯 모델 옵션", expanded=True):
            integer_mode = st.checkbox(
                "정수 계획 (IP 모드)",
                value=False,
                help="작업자·생산량을 정수로 강제합니다.",
            )

    params = dict(
        reg_wage=reg_wage,           ot_wage=ot_wage,
        hire_cost=hire_cost,         layoff_cost=layoff_cost,
        inv_cost=inv_cost,           backorder_cost=backorder_cost,
        material_cost=material_cost, subcontract_cost=subcontract_cost,
        work_days=work_days,         work_hours=work_hours,
        std_time=std_time,           ot_limit=ot_limit,
        init_workforce=init_workforce,
        init_inventory=init_inventory,
        final_inventory=final_inventory,
    )
    return demand, params, integer_mode


# ────────────────────────────────────────────────────────────
# KPI 카드 & 진단 메시지
# ────────────────────────────────────────────────────────────

def render_kpi(result: dict, demand: list[float], n_periods: int) -> dict:
    """
    상단 KPI 카드 4개를 렌더링.

    Returns
    -------
    agg : 집계값 딕셔너리 (이후 차트 함수에서 재사용)
    """
    total_cost   = result["total_cost"]
    total_subc   = sum(result["C"][1:])
    total_short  = sum(result["S"][1:])
    avg_workers  = float(np.mean(result["W"][1:]))
    total_hire   = sum(result["H"][1:])
    total_layoff = sum(result["L"][1:])
    total_ot     = sum(result["O"][1:])
    total_demand = sum(demand)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 총 비용",
              f"{total_cost:,.0f} 천원",
              f"₩ {total_cost * 1000:,.0f}")
    k2.metric("📦 수요 / 공급",
              f"{int(total_demand):,} / {sum(result['P'][1:]) + total_subc:,.0f} 개",
              f"외주 {int(total_subc):,} · 부재고 {int(total_short):,}")
    k3.metric("👥 평균 인력",
              f"{avg_workers:.1f} 명",
              f"고용 {total_hire:.0f} · 해고 {total_layoff:.0f}")
    k4.metric("⏰ 총 초과근무",
              f"{total_ot:,.0f} 시간",
              f"평균 {total_ot / n_periods:.1f} 시간/월")

    return dict(
        total_cost=total_cost,   total_subc=total_subc,
        total_short=total_short, avg_workers=avg_workers,
        total_hire=total_hire,   total_layoff=total_layoff,
        total_ot=total_ot,
    )


# ────────────────────────────────────────────────────────────
# 운영 대시보드 차트
# ────────────────────────────────────────────────────────────

# 공통 레이아웃 기본값 (각 차트에서 **LAYOUT_KW 로 언팩해 사용)
LAYOUT_KW = dict(
    template="plotly_white",
    height=380,
    margin=dict(l=10, r=10, t=40, b=60),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    font=dict(family=FONT_FAMILY, size=12, color="#0f172a"),
)


def chart_supply_vs_demand(
    result: dict, demand: list[float],
    months_p: list[str], T_idx: list[int], total_subc: float,
) -> go.Figure:
    """수요 대비 자체생산 + 외주 총공급 비교 차트."""
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
    # 외주가 있을 때만 총공급 선 추가 (없으면 자체생산과 완전히 겹쳐 지저분해짐)
    if total_subc > 0.01:
        total_supply = [result["P"][t] + result["C"][t] for t in T_idx]
        fig.add_trace(go.Scatter(
            x=months_p, y=total_supply,
            name="총 공급 (P+C)", mode="lines+markers",
            line=dict(color=ACCENT, width=2, dash="dash"), marker=dict(size=7),
        ))

    fig.update_layout(yaxis_title="수량 (개)", **LAYOUT_KW)
    return fig


def chart_inventory(
    result: dict, months_full: list[str], final_inventory: float,
) -> go.Figure:
    """재고 / 부재고 추이 (0 위 = 재고 비축, 0 아래 = 백오더)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months_full, y=result["I"], name="재고 (I)",
        fill="tozeroy", line=dict(color=PRIMARY, width=2.5),
        fillcolor="rgba(37,99,235,0.18)",
    ))
    # 실제 부재고가 발생한 경우에만 표시
    if any(s > 0.01 for s in result["S"]):
        fig.add_trace(go.Scatter(
            x=months_full, y=[-s for s in result["S"]],
            name="부재고 (S)", fill="tozeroy",
            line=dict(color=DANGER, width=2.5),
            fillcolor="rgba(239,68,68,0.20)",
        ))
    fig.add_hline(
        y=final_inventory, line_dash="dot", line_color=ACCENT,
        annotation_text=f"최종재고 목표 {final_inventory}",
        annotation_position="top right",
    )

    fig.update_layout(yaxis_title="수량 (개)", **LAYOUT_KW)
    return fig


def chart_supply_breakdown(
    result: dict, demand: list[float],
    months_p: list[str], T_idx: list[int], std_time: float,
) -> go.Figure:
    """공급 원천별 구성 스택 바 (정규 / 초과 / 외주)."""
    ot_units  = [result["O"][t] / std_time for t in T_idx]
    reg_units = [max(0.0, result["P"][t] - ot_units[i]) for i, t in enumerate(T_idx)]
    sub_units = [result["C"][t] for t in T_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months_p, y=reg_units,
                         name="정규시간 생산", marker_color=PRIMARY))
    # 실제로 사용된 채널만 표시 (0인 항목은 범례에서 제외)
    if any(v > 0.01 for v in ot_units):
        fig.add_trace(go.Bar(x=months_p, y=ot_units,
                             name="초과시간 생산", marker_color=PRIMARY_LIGHT))
    if any(v > 0.01 for v in sub_units):
        fig.add_trace(go.Bar(x=months_p, y=sub_units,
                             name="외주 (Subcontract)", marker_color=ACCENT))
    fig.add_trace(go.Scatter(
        x=months_p, y=demand, name="수요(참고)", mode="lines+markers",
        line=dict(color=DANGER, width=2, dash="dot"), marker=dict(size=6),
    ))

    fig.update_layout(
        barmode="stack", yaxis_title="수량 (개)",
        **{**LAYOUT_KW, "margin": dict(l=10, r=10, t=40, b=80)},
    )
    return fig


def chart_cost_donut(result: dict, total_cost: float) -> go.Figure:
    """비용 항목별 비중 도넛 차트 (0원 항목 제외)."""
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
            text=(
                f"<b>{total_cost:,.0f}</b>"
                f"<br><span style='font-size:11px;color:#64748b'>천원</span>"
            ),
            x=0.5, y=0.5, font=dict(size=18, color=PRIMARY), showarrow=False,
        )],
        height=380, margin=dict(l=10, r=10, t=40, b=20),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=11),
    )
    return fig


def chart_workforce(
    result: dict, months_full: list[str], months_p: list[str],
) -> go.Figure:
    """인력 운영 계획 (작업자 수 추이 + 고용/해고 막대, 이중 y축)."""
    hl_vals = result["H"][1:] + result["L"][1:]
    hl_max  = max(max(hl_vals) if hl_vals else 1, 1)

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
        yaxis2=dict(
            title="고용/해고 (명)", overlaying="y", side="right",
            showgrid=False, zeroline=True, zerolinecolor="#cbd5e1",
            range=[-hl_max * 1.4, hl_max * 1.4],  # 대칭 고정 — 이상치에 끌려가지 않게
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        font=dict(family=FONT_FAMILY, size=12),
    )
    return fig


def render_operation_charts(
    result: dict, demand: list[float], params: dict,
    months_full: list[str], months_p: list[str],
    T_idx: list[int], total_subc: float, total_cost: float,
) -> None:
    """운영 대시보드 차트 4개(2×2 그리드) + 인력 운영 차트를 렌더링."""
    st.markdown("## 📊 운영 대시보드")

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.markdown("#### 📈 수요 대비 공급")
        st.plotly_chart(
            chart_supply_vs_demand(result, demand, months_p, T_idx, total_subc),
            use_container_width=True,
        )
    with row1c2:
        st.markdown("#### 📦 재고 / 부재고 추이")
        st.plotly_chart(
            chart_inventory(result, months_full, params["final_inventory"]),
            use_container_width=True,
        )

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.markdown("#### 🏭 공급 원천별 구성")
        st.plotly_chart(
            chart_supply_breakdown(result, demand, months_p, T_idx, params["std_time"]),
            use_container_width=True,
        )
    with row2c2:
        st.markdown("#### 💰 비용 항목별 비중")
        st.plotly_chart(
            chart_cost_donut(result, total_cost),
            use_container_width=True,
        )

    st.markdown("#### 👥 인력 운영 계획")
    st.plotly_chart(
        chart_workforce(result, months_full, months_p),
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────────
# 상세 결과 테이블 & CSV 다운로드
# ────────────────────────────────────────────────────────────

def render_detail_table(
    result: dict, demand: list[float],
    months_full: list[str],
) -> None:
    """결정변수 시점별 상세 결과 테이블과 CSV 다운로드 버튼."""
    st.markdown("## 📋 상세 결과")

    result_df = pd.DataFrame({
        "월":         months_full,
        "수요 D":     [0] + [int(d) for d in demand],
        "작업자 W":   [round(x, 2) for x in result["W"]],
        "고용 H":     [round(x, 2) for x in result["H"]],
        "해고 L":     [round(x, 2) for x in result["L"]],
        "생산 P":     [round(x, 2) for x in result["P"]],
        "재고 I":     [round(x, 2) for x in result["I"]],
        "부재고 S":   [round(x, 2) for x in result["S"]],
        "외주 C":     [round(x, 2) for x in result["C"]],
        "초과시간 O": [round(x, 2) for x in result["O"]],
    })
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    dl_col, _ = st.columns([1, 5])
    with dl_col:
        st.download_button(
            "📥 결과 CSV 다운로드",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="production_plan.csv",
            mime="text/csv",
        )


# ────────────────────────────────────────────────────────────
# 수리 모델 & 파라미터 요약 (Expander)
# ────────────────────────────────────────────────────────────

def render_model_expanders() -> None:
    """LP/IP 수식 정리 expander."""

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
- **인력 균형**      : $W_t = W_{t-1} + H_t - L_t$
- **생산능력**       : $P_t \le 40\,W_t + O_t/4$
- **재고 균형**      : $I_t = I_{t-1} + P_t + C_t - D_t - S_{t-1} + S_t$
- **초과근무 한도**  : $O_t \le 10\,W_t$
- **초기/최종 조건** : $W_0=80,\; I_0=1000,\; I_T \ge 500,\; S_0=S_T=0$
- **비음수**         : 모든 변수 $\ge 0$ (IP 모드 시 정수)

##### 전략 옵션
- **LP 모드** : 모든 변수 연속값 허용 → 소수점 인력도 가능
- **IP 모드** : 작업자·생산량을 정수로 강제 → 현실적인 계획
            """
        )


# ────────────────────────────────────────────────────────────
# 메인 실행 흐름
# ────────────────────────────────────────────────────────────

def main() -> None:
    setup_page()

    # 1. 사이드바에서 입력 받기
    demand, params, integer_mode = render_sidebar()

    # 2. 페이지 헤더
    st.markdown("# 🏭 총괄생산계획 최적화 대시보드")
    st.markdown(
        f"<p class='small-note'>"
        f"원예장비 제조업체 사례 · LP/IP 기반 의사결정 지원 · "
        f"모드 = <b>{'정수(IP)' if integer_mode else '연속(LP)'}</b>"
        f"</p>",
        unsafe_allow_html=True,
    )

    # 3. 최적화 실행 (st.cache_data 적용 — 동일 입력 재계산 없음)
    with st.spinner("최적화 풀이 중..."):
        result = solve_app(tuple(demand), params, strategy="optimal", integer=integer_mode)

    if result is None:
        st.error(
            "⚠️ **해를 찾을 수 없습니다 (Infeasible).**\n\n"
            "파라미터 조합으로는 모든 제약을 동시에 만족할 수 없습니다.\n\n"
            "▸ 초기 작업자 수를 늘리거나\n"
            "▸ 최종재고 목표를 낮추거나\n"
            "▸ 수요를 조정해보세요."
        )
        st.stop()

    # 시간축 라벨
    n_periods   = len(demand)
    T_idx       = list(range(1, n_periods + 1))
    months_full = [f"{i}월"   for i in range(n_periods + 1)]  # 0월 ~ T월
    months_p    = [f"{i+1}월" for i in range(n_periods)]      # 1월 ~ T월

    # 4. KPI 카드 & 진단 메시지
    agg = render_kpi(result, demand, n_periods)

    # 5. 운영 대시보드 차트
    render_operation_charts(
        result, demand, params,
        months_full, months_p, T_idx,
        agg["total_subc"], agg["total_cost"],
    )

    # 6. 상세 결과 테이블 & CSV 다운로드
    render_detail_table(result, demand, months_full)

    # 7. 수리 모델 & 파라미터 요약
    render_model_expanders()

    # 8. 푸터
    st.markdown(
        "<div style='text-align:center; padding:1.5rem 0 .5rem 0;"
        " color:#94a3b8; font-size:.85rem;'>"
        "© 2026 총괄생산계획 최적화 대시보드 · Streamlit + PuLP + Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


main()
