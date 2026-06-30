"""
Dashboard del FULL GRID — barrido sistemático para recuperación de IDs.

Lee los CSV agregados del grid completo (uno por optimizador) y monta una serie
de visualizaciones *explicativas* pensadas para la memoria del TFM: cada gráfico
va acompañado de qué se está mirando y qué se concluye.

Grid de referencia (bypass2, `grid_raw/`):
    4 optimizadores (UMDA, EMNA, EGNA, KEDA)
    × 2 muestreos (non_symmetric, symmetric)
    × 5 fitness (binary, margin, softmax, regret, entropy)
    × 4 niveles de % de reglas de decisión (5, 10, 20, 40)
    = 160 configuraciones, 5 repeticiones cada una, T=1, stop=top50.

Uso:
    streamlit run dashboard_fullgrid.py
"""

import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Full Grid Explorer", page_icon="🧮", layout="wide")

st.title("🧮 Full Grid — Recuperación de IDs por barrido sistemático")
st.caption(
    "Comparativa de **optimizadores EDA**, **funciones de fitness**, **% de reglas "
    "de decisión** y **modo de muestreo**. Cada bloque explica qué se mira y qué se concluye."
)

# ─── DESCUBRIMIENTO DE CSVs ────────────────────────────────────────────────────
# El full grid balanceado vive en example/<red>/grid_raw (la versión más reciente)
# con fallback a grid_full. Cada optimizador es un CSV grid_search_results_*.csv.

_PREFERRED_DIRS = [
    "example/bypass2/grid_raw",
    "example/bypass2/grid_full",
]
_CROSSCHECK_DIRS = [
    "example/nhlv1/grid_full_nhlv1",
    "example/bypass2/grid_full_nhlv1",
]


@st.cache_data
def _load_dir(d: str, net_label: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "grid_search_results_*.csv"))):
        opt = os.path.basename(f).replace("grid_search_results_", "").replace(".csv", "")
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["optimizer"] = opt.upper()
        df["net"] = net_label
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _first_existing(dirs):
    for d in dirs:
        if glob.glob(os.path.join(d, "grid_search_results_*.csv")):
            return d
    return None


_main_dir = _first_existing(_PREFERRED_DIRS)
if _main_dir is None:
    st.error(
        "No se encontraron CSVs del full grid en "
        f"{_PREFERRED_DIRS}. Ejecuta `grid_search.py` por optimizador primero."
    )
    st.stop()

df = _load_dir(_main_dir, "bypass2")
_cross_dir = _first_existing(_CROSSCHECK_DIRS)
df_cross = _load_dir(_cross_dir, "nhlv1") if _cross_dir else pd.DataFrame()

st.caption(f"📂 Grid principal: `{_main_dir}` · {len(df)} configuraciones · "
           f"{df['optimizer'].nunique()} optimizadores."
           + (f"  ·  Cross-check: `{_cross_dir}`" if _cross_dir else ""))

# Orden y paleta consistentes para los optimizadores en todos los gráficos.
_OPT_ORDER = ["UMDA", "EMNA", "EGNA", "KEDA"]
_OPT_ORDER = [o for o in _OPT_ORDER if o in df["optimizer"].unique()] + \
             [o for o in df["optimizer"].unique() if o not in _OPT_ORDER]
_PALETTE = px.colors.qualitative.Set2

# ─── FILTROS ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filtros")
    st.caption("Acotan todos los gráficos a la vez. Por defecto: todo el grid.")
    f_opt = st.multiselect("Optimizador", _OPT_ORDER, default=_OPT_ORDER)
    f_fit = st.multiselect("Fitness", sorted(df["fitness_type"].unique()),
                           default=sorted(df["fitness_type"].unique()))
    f_samp = st.multiselect("Muestreo", sorted(df["sampling_mode"].unique()),
                            default=sorted(df["sampling_mode"].unique()))
    f_pct = st.multiselect("% Reglas de decisión",
                           sorted(df["n_decision_rules_pct"].unique()),
                           default=sorted(df["n_decision_rules_pct"].unique()))

d = df[
    df["optimizer"].isin(f_opt)
    & df["fitness_type"].isin(f_fit)
    & df["sampling_mode"].isin(f_samp)
    & df["n_decision_rules_pct"].isin(f_pct)
].copy()

if d.empty:
    st.warning("Ningún dato con los filtros actuales.")
    st.stop()

# ─── KPIs ─────────────────────────────────────────────────────────────────────
best = d.loc[d["accuracy_mean"].idxmax()]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Configuraciones", len(d))
k2.metric("Mejor accuracy (media)", f"{best['accuracy_mean']:.1f}%",
          f"{best['optimizer']} · {best['fitness_type']} · {best['n_decision_rules_pct']}%")
k3.metric("Accuracy máx. global", f"{d['accuracy_max'].max():.0f}%")
k4.metric("Gen. media de parada", f"{d['stop_gen_mean'].mean():.1f}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 1 · RANKING DE OPTIMIZADORES
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · ¿Qué optimizador recupera mejor?")
st.markdown(
    "Accuracy media de cada optimizador **promediando todas sus configuraciones** "
    "(fitness × muestreo × % reglas). La barra de error es la desviación entre configuraciones."
)

rank = (
    d.groupby("optimizer")["accuracy_mean"]
    .agg(media="mean", desv="std", maximo="max")
    .reindex([o for o in _OPT_ORDER if o in d["optimizer"].unique()])
    .reset_index()
)
fig1 = px.bar(
    rank, x="optimizer", y="media", error_y="desv", text="media",
    color="optimizer", color_discrete_sequence=_PALETTE,
    category_orders={"optimizer": _OPT_ORDER},
    labels={"media": "Accuracy media (%)", "optimizer": "Optimizador"},
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig1.update_layout(showlegend=False, height=380, yaxis_range=[75, 102])
st.plotly_chart(fig1, use_container_width=True)
st.success(
    "**Conclusión.** KEDA domina con claridad (~97% de media, hasta 100%), muy por "
    "encima de EMNA/UMDA/EGNA (~86–88%). Es el único que recupera la red **perfectamente**. "
    "El precio: necesita una población mucho mayor (ver bloque de coste)."
)

# ══════════════════════════════════════════════════════════════════════════════
# 2 · EFECTO DEL % DE REGLAS DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · ¿Cuántas reglas de decisión hacen falta?")
st.markdown(
    "Accuracy frente al **% de reglas de decisión** que ve el optimizador. Cuantas más "
    "reglas, más información para reconstruir los IDs — pero también más coste."
)
by_pct = (
    d.groupby(["n_decision_rules_pct", "optimizer"], as_index=False)["accuracy_mean"].mean()
)
fig2 = px.line(
    by_pct, x="n_decision_rules_pct", y="accuracy_mean", color="optimizer",
    markers=True, color_discrete_sequence=_PALETTE,
    category_orders={"optimizer": _OPT_ORDER},
    labels={"n_decision_rules_pct": "% reglas de decisión",
            "accuracy_mean": "Accuracy media (%)", "optimizer": "Optimizador"},
)
fig2.update_layout(height=400)
st.plotly_chart(fig2, use_container_width=True)
st.success(
    "**Conclusión.** La accuracy crece de forma **monótona** con el % de reglas "
    "(global: 5%→86%, 10%→88%, 20%→91%, 40%→94%). El efecto es grande en los "
    "optimizadores débiles (EGNA gana ~8 puntos del 5% al 40%) y casi nulo en KEDA, "
    "que ya parte de ~93% al 5% y satura cerca del 100%. **Más reglas compensan un "
    "optimizador peor**, pero KEDA llega lejos incluso con pocas."
)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · MAPA DE CALOR — INTERACCIÓN optimizador × % reglas
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 · Mapa de calor: optimizador × % reglas")
st.markdown("La misma interacción del bloque anterior, en rejilla. Útil para localizar la celda ganadora.")
heat = d.pivot_table("accuracy_mean", "optimizer", "n_decision_rules_pct", aggfunc="mean")
heat = heat.reindex([o for o in _OPT_ORDER if o in heat.index])
fig3 = px.imshow(
    heat, text_auto=".1f", aspect="auto", color_continuous_scale="Tealgrn",
    labels={"x": "% reglas de decisión", "y": "Optimizador", "color": "Accuracy %"},
)
fig3.update_layout(height=360)
st.plotly_chart(fig3, use_container_width=True)
st.success(
    "**Conclusión.** La esquina KEDA × 40% reglas roza el 100%. KEDA gana en **todas** "
    "las columnas, así que la elección de optimizador pesa más que el nº de reglas."
)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · ¿IMPORTA LA FUNCIÓN DE FITNESS?
# ══════════════════════════════════════════════════════════════════════════════
st.header("4 · ¿Importa la función de fitness?")
st.markdown(
    "Accuracy media por tipo de fitness (sobre todos los optimizadores y configuraciones). "
    "Si las barras están parejas, complicar el fitness no aporta."
)
by_fit = (
    d.groupby("fitness_type", as_index=False)["accuracy_mean"].mean()
    .sort_values("accuracy_mean", ascending=False)
)
fig4 = px.bar(
    by_fit, x="fitness_type", y="accuracy_mean", text="accuracy_mean",
    color="fitness_type", color_discrete_sequence=_PALETTE,
    labels={"fitness_type": "Fitness", "accuracy_mean": "Accuracy media (%)"},
)
fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig4.update_layout(showlegend=False, height=360,
                   yaxis_range=[float(by_fit["accuracy_mean"].min()) - 3,
                                float(by_fit["accuracy_mean"].max()) + 2])
st.plotly_chart(fig4, use_container_width=True)
st.success(
    "**Conclusión.** Apenas importa: todas las funciones caen entre ~88.8% y ~90.3% "
    "(menos de 1.5 puntos de diferencia). El `binary`, el más simple, casi iguala al "
    "mejor → **no compensa complicar el fitness**. Único matiz: en KEDA, `softmax`/`entropy` "
    "rinden algo mejor cuando hay pocas reglas."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · MUESTREO simétrico vs no-simétrico
# ══════════════════════════════════════════════════════════════════════════════
st.header("5 · Muestreo simétrico vs no-simétrico")
st.markdown("Comparación directa del modo de muestreo, por optimizador.")
by_samp = d.groupby(["optimizer", "sampling_mode"], as_index=False)["accuracy_mean"].mean()
fig5 = px.bar(
    by_samp, x="optimizer", y="accuracy_mean", color="sampling_mode",
    barmode="group", color_discrete_sequence=_PALETTE,
    category_orders={"optimizer": _OPT_ORDER},
    labels={"optimizer": "Optimizador", "accuracy_mean": "Accuracy media (%)",
            "sampling_mode": "Muestreo"},
)
fig5.update_layout(height=360, yaxis_range=[80, 102])
st.plotly_chart(fig5, use_container_width=True)
st.success(
    "**Conclusión.** Diferencia despreciable (global 90.0% simétrico vs 89.4% no-simétrico). "
    "El modo de muestreo **no es un factor relevante** en esta red."
)

# ══════════════════════════════════════════════════════════════════════════════
# 6 · TRADE-OFF coste vs accuracy
# ══════════════════════════════════════════════════════════════════════════════
st.header("6 · Coste computacional vs accuracy")
st.markdown(
    "Cada punto es una configuración: **CPU total (s)** en X, **accuracy** en Y. "
    "Arriba-izquierda = barato y preciso (frontera de Pareto)."
)
if "cpu_total_mean" in d.columns:
    fig6 = px.scatter(
        d, x="cpu_total_mean", y="accuracy_mean", color="optimizer",
        symbol="optimizer", color_discrete_sequence=_PALETTE,
        category_orders={"optimizer": _OPT_ORDER},
        hover_data=["fitness_type", "n_decision_rules_pct", "sampling_mode"],
        labels={"cpu_total_mean": "CPU total medio (s)",
                "accuracy_mean": "Accuracy media (%)", "optimizer": "Optimizador"},
    )
    fig6.update_traces(marker=dict(size=11, opacity=0.8))
    fig6.update_layout(height=440)
    st.plotly_chart(fig6, use_container_width=True)
    st.success(
        "**Conclusión.** Dos familias claras: **UMDA y EMNA son baratísimos** (~0.5 s) con "
        "techo ~88%; **KEDA y EGNA son caros** (~4–5 s). KEDA es el único caro que **vale "
        "la pena** porque su accuracy (≈97–100%) justifica el coste; **EGNA es caro y "
        "mediocre** (~86%) → la peor relación coste/beneficio. Si la CPU es crítica, "
        "UMDA/EMNA al 40% dan ~92–93% a 1/10 del coste de KEDA."
    )
else:
    st.info("No hay columnas de CPU en estos CSVs.")

# ══════════════════════════════════════════════════════════════════════════════
# 7 · DESGLOSE DEL GANADOR (KEDA): fitness × % reglas
# ══════════════════════════════════════════════════════════════════════════════
if "KEDA" in d["optimizer"].unique():
    st.header("7 · El ganador en detalle — KEDA: fitness × % reglas")
    st.markdown("Dónde KEDA llega antes al 100%, cruzando fitness y % de reglas.")
    k = d[d["optimizer"] == "KEDA"]
    heat_k = k.pivot_table("accuracy_mean", "fitness_type", "n_decision_rules_pct", aggfunc="mean")
    fig7 = px.imshow(
        heat_k, text_auto=".1f", aspect="auto", color_continuous_scale="Tealgrn",
        labels={"x": "% reglas", "y": "Fitness", "color": "Accuracy %"},
    )
    fig7.update_layout(height=360)
    st.plotly_chart(fig7, use_container_width=True)
    st.success(
        "**Conclusión.** Con KEDA basta el **20%** de reglas para superar el 99% con casi "
        "cualquier fitness; al 40% varias combinaciones tocan el 100% exacto. No necesita "
        "fitness sofisticado: `binary` al 40% ya da ~99.7%."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 8 · CROSS-CHECK en red grande (nhlv1)
# ══════════════════════════════════════════════════════════════════════════════
if not df_cross.empty:
    st.header("8 · Cross-check en red grande (nhlv1)")
    st.markdown(
        "La misma idea en una red mayor (67 reglas, modo `utility_only`). Aquí el grid "
        "completo solo está disponible para algunos optimizadores; sirve de validación cualitativa."
    )
    cc = df_cross.groupby(["optimizer", "n_decision_rules_pct"], as_index=False)["accuracy_mean"].mean()
    fig8 = px.line(
        cc, x="n_decision_rules_pct", y="accuracy_mean", color="optimizer",
        markers=True, color_discrete_sequence=_PALETTE,
        labels={"n_decision_rules_pct": "% reglas de decisión",
                "accuracy_mean": "Accuracy media (%)", "optimizer": "Optimizador"},
    )
    fig8.update_layout(height=380)
    st.plotly_chart(fig8, use_container_width=True)
    st.info(
        "**Lectura.** En la red grande el techo de los EDA simples (UMDA) baja a ~85–86% y "
        "el efecto de añadir reglas se aplana: una red más compleja **necesita el optimizador "
        "fuerte (KEDA)** o un % de reglas alto para acercarse a la recuperación total."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 9 · TABLA — TOP CONFIGURACIONES
# ══════════════════════════════════════════════════════════════════════════════
st.header("9 · Top configuraciones")
cols = ["optimizer", "sampling_mode", "fitness_type", "n_decision_rules_pct",
        "accuracy_mean", "accuracy_std", "accuracy_max", "stop_gen_mean", "cpu_total_mean"]
cols = [c for c in cols if c in d.columns]
top = d.sort_values("accuracy_mean", ascending=False)[cols].head(20)
st.dataframe(
    top, use_container_width=True, hide_index=True,
    column_config={
        "optimizer": "Optimizador", "sampling_mode": "Muestreo", "fitness_type": "Fitness",
        "n_decision_rules_pct": st.column_config.NumberColumn("% Reglas", format="%d%%"),
        "accuracy_mean": st.column_config.NumberColumn("Accuracy media", format="%.1f%%"),
        "accuracy_std": st.column_config.NumberColumn("σ", format="%.2f"),
        "accuracy_max": st.column_config.NumberColumn("Accuracy máx", format="%.0f%%"),
        "stop_gen_mean": st.column_config.NumberColumn("Gen. parada", format="%.1f"),
        "cpu_total_mean": st.column_config.NumberColumn("CPU total (s)", format="%.2f"),
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("📌 Conclusiones globales")
st.markdown(
    """
**1. El optimizador es el factor decisivo.** KEDA recupera la red prácticamente al 100%
(~97% de media) frente al ~86–88% de UMDA, EMNA y EGNA. La elección de optimizador
pesa más que cualquier otro hiperparámetro del grid.

**2. Más reglas de decisión siempre ayudan, con rendimientos decrecientes.** La accuracy
sube de forma monótona (86%→88%→91%→94% al pasar de 5% a 40% de reglas). El efecto es
fuerte en los optimizadores débiles y casi nulo en KEDA, que ya está saturado.

**3. La función de fitness es casi irrelevante.** Menos de 1.5 puntos separan al mejor del
peor fitness. `binary`, el más simple, basta → no hay que complicar la función objetivo.

**4. El modo de muestreo no importa.** Simétrico y no-simétrico empatan (~0.6 puntos).

**5. Coste: KEDA es el único caro que compensa.** UMDA/EMNA son ~10× más baratos pero
topan en ~88%; EGNA es caro y mediocre (peor relación coste/beneficio). KEDA paga su
coste en accuracy. Su sobrecoste viene de necesitar población grande (sg=400) para que
la estimación de densidad por kernel no degenere.

**6. Recomendación práctica.**
- *Máxima precisión:* **KEDA + 20–40% de reglas + `binary` + cualquier muestreo** → ~99–100%.
- *CPU limitada:* **UMDA o EMNA al 40%** → ~92–93% a una décima parte del coste.
- En **redes grandes (nhlv1)** los EDA simples se quedan en ~85%: ahí KEDA o un % de reglas
  alto se vuelven imprescindibles.
"""
)
