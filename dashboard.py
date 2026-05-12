import streamlit as st
import pandas as pd
import altair as alt
import glob
import os

st.set_page_config(page_title="EDA Grid Search Explorer", page_icon="📊", layout="wide")

st.title("📊 EDA Grid Search Explorer")
st.markdown("Visualización interactiva de los resultados del barrido paramétrico para la recuperación de IDs.")

# 1. Localizar los datasets disponibles
base_folders = [os.path.dirname(f) for f in glob.glob("example/*/grid_search_results.csv")]
base_folders += [os.path.dirname(f) for f in glob.glob("grid_search_results.csv")] # Por si está en la raíz
base_folders = list(set(base_folders))

if not base_folders:
    st.warning("No se encontraron resultados de Grid Search. Ejecuta `python grid_search.py` primero.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("📁 Origen de Datos")
selected_folder = st.sidebar.selectbox("Selecciona la carpeta", base_folders)

results_file = os.path.join(selected_folder, "grid_search_results.csv")
curves_file = os.path.join(selected_folder, "grid_search_curves.csv")

@st.cache_data
def load_data(res_path, curv_path):
    df_res = pd.read_csv(res_path)
    df_curv = pd.read_csv(curv_path) if os.path.exists(curv_path) else None
    return df_res, df_curv

df_results, df_curves = load_data(results_file, curves_file)

st.sidebar.header("🔍 Filtros de Exploración")
selected_fitness = st.sidebar.multiselect("Tipo de Fitness", df_results["fitness_type"].unique(), default=df_results["fitness_type"].unique())
selected_stop = st.sidebar.multiselect("Modo de Parada", df_results["stop_mode"].unique(), default=df_results["stop_mode"].unique())
selected_rules = st.sidebar.multiselect("% Reglas Entrenadas", df_results["n_decision_rules_pct"].unique(), default=df_results["n_decision_rules_pct"].unique())

filtered_results = df_results[
    (df_results["fitness_type"].isin(selected_fitness)) &
    (df_results["stop_mode"].isin(selected_stop)) &
    (df_results["n_decision_rules_pct"].isin(selected_rules))
]

if filtered_results.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada.")
    st.stop()

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Experimentos Mostrados", len(filtered_results))
best_acc = filtered_results["best_accuracy_mean"].max()
col2.metric("Mejor Accuracy Medio", f"{best_acc:.2f}%")
best_fit = filtered_results["best_fitness_mean"].min()
col3.metric("Mejor Fitness Medio", f"{best_fit:.4e}")
avg_gen = filtered_results["stop_gen_mean"].mean()
col4.metric("Generación Media de Parada", f"{avg_gen:.1f}")

st.divider()

# --- SCATTER PLOT GLOBAL ---
st.subheader("🎯 Resumen de Precisión vs Generación de Parada")
scatter_chart = alt.Chart(filtered_results).mark_circle(size=100).encode(
    x=alt.X('stop_gen_mean:Q', title='Generación Media de Parada'),
    y=alt.Y('best_accuracy_mean:Q', title='Precisión Media (%)', scale=alt.Scale(domain=[filtered_results['best_accuracy_mean'].min()-5, 100])),
    color=alt.Color('fitness_type:N', title='Fitness Type', scale=alt.Scale(scheme='category10')),
    shape=alt.Shape('stop_mode:N', title='Stop Mode'),
    size=alt.Size('n_decision_rules_pct:O', title='% Reglas'),
    tooltip=[
        alt.Tooltip('fitness_type:N', title='Fitness'),
        alt.Tooltip('stop_mode:N', title='Stop'),
        alt.Tooltip('n_decision_rules_pct:O', title='% Reglas'),
        alt.Tooltip('best_accuracy_mean:Q', title='Accuracy Medio (%)'),
        alt.Tooltip('best_accuracy_max:Q', title='Accuracy Max (%)'),
        alt.Tooltip('stop_gen_mean:Q', title='Generación Parada'),
    ]
).properties(height=400).interactive()

st.altair_chart(scatter_chart, width="stretch")

# --- ANALISIS DE CURVAS ---
if df_curves is not None:
    st.divider()
    st.subheader("📈 Curvas de Evolución")
    
    # Filtrar las curvas usando los mismos filtros globales
    filtered_curves = df_curves[
        (df_curves["fitness_type"].isin(selected_fitness)) &
        (df_curves["stop_mode"].isin(selected_stop)) &
        (df_curves["n_decision_rules_pct"].isin(selected_rules))
    ].copy()
    
    if not filtered_curves.empty:
        # Crear un nombre único para cada configuración
        filtered_curves["config_name"] = (
            filtered_curves["fitness_type"] + " | " + 
            filtered_curves["stop_mode"] + " | " + 
            filtered_curves["n_decision_rules_pct"].astype(str) + "%"
        )
        
        tab1, tab2 = st.tabs(["Comparación Global (Todas las filtradas)", "Detalle Individual (Con dispersión)"])
        
        with tab1:
            st.markdown("Compara todas las configuraciones que tienes activas en el filtro.")
            
            # Selector interactivo para resaltar líneas
            highlight = alt.selection_point(on='mouseover', fields=['config_name'], nearest=True)
            
            base_acc = alt.Chart(filtered_curves).encode(
                x=alt.X('generation:Q', title='Generación'),
                y=alt.Y('mean_accuracy:Q', title='Accuracy Medio (%)'),
                color=alt.Color('config_name:N', title='Configuración', legend=alt.Legend(orient='bottom', columns=3)),
                tooltip=['config_name', 'generation', 'mean_accuracy', 'mean_fitness']
            )
            
            lines_acc = base_acc.mark_line(strokeWidth=3).encode(
                opacity=alt.condition(highlight, alt.value(1), alt.value(0.2))
            )
            
            points_acc = base_acc.mark_circle().encode(
                opacity=alt.value(0)
            ).add_params(highlight)
            
            st.altair_chart((lines_acc + points_acc).interactive(), width="stretch")
            
            base_fit = alt.Chart(filtered_curves).encode(
                x=alt.X('generation:Q', title='Generación'),
                y=alt.Y('mean_fitness:Q', title='Fitness Medio (Penalización)'),
                color=alt.Color('config_name:N', title='Configuración', legend=None),
                tooltip=['config_name', 'generation', 'mean_fitness']
            )
            
            lines_fit = base_fit.mark_line(strokeWidth=3).encode(
                opacity=alt.condition(highlight, alt.value(1), alt.value(0.2))
            )
            
            points_fit = base_fit.mark_circle().encode(
                opacity=alt.value(0)
            ).add_params(highlight)
            
            st.altair_chart((lines_fit + points_fit).interactive(), width="stretch")

        with tab2:
            st.markdown("Selecciona una configuración específica para ver la zona de dispersión (±1 desviación estándar).")
            sel_col1, sel_col2, sel_col3 = st.columns(3)
            c_fit = sel_col1.selectbox("Fitness", filtered_results["fitness_type"].unique())
            c_stop = sel_col2.selectbox("Stop Mode", filtered_results[filtered_results["fitness_type"] == c_fit]["stop_mode"].unique())
            c_rules = sel_col3.selectbox("% Reglas", filtered_results[(filtered_results["fitness_type"] == c_fit) & (filtered_results["stop_mode"] == c_stop)]["n_decision_rules_pct"].unique())
            
            curve_data = filtered_curves[
                (filtered_curves["fitness_type"] == c_fit) & 
                (filtered_curves["stop_mode"] == c_stop) & 
                (filtered_curves["n_decision_rules_pct"] == c_rules)
            ].copy()
            
            if not curve_data.empty:
                curve_data['acc_lower'] = curve_data['mean_accuracy'] - curve_data['accuracy_std']
                curve_data['acc_upper'] = curve_data['mean_accuracy'] + curve_data['accuracy_std']
                curve_data['acc_lower'] = curve_data['acc_lower'].clip(lower=0)
                curve_data['acc_upper'] = curve_data['acc_upper'].clip(upper=100)
                
                line_acc = alt.Chart(curve_data).mark_line(color='#1f77b4', strokeWidth=3).encode(
                    x=alt.X('generation:Q', title='Generación'),
                    y=alt.Y('mean_accuracy:Q', title='Accuracy (%)'),
                    tooltip=['generation', 'mean_accuracy', 'accuracy_std']
                )
                band_acc = alt.Chart(curve_data).mark_area(opacity=0.2, color='#1f77b4').encode(
                    x='generation:Q',
                    y='acc_lower:Q',
                    y2='acc_upper:Q'
                )
                
                curve_data['fit_lower'] = curve_data['mean_fitness'] - curve_data['fitness_std']
                curve_data['fit_upper'] = curve_data['mean_fitness'] + curve_data['fitness_std']
                
                line_fit = alt.Chart(curve_data).mark_line(color='#ff7f0e', strokeWidth=3).encode(
                    x=alt.X('generation:Q', title='Generación'),
                    y=alt.Y('mean_fitness:Q', title='Fitness (Penalización)'),
                    tooltip=['generation', 'mean_fitness', 'fitness_std']
                )
                band_fit = alt.Chart(curve_data).mark_area(opacity=0.2, color='#ff7f0e').encode(
                    x='generation:Q',
                    y='fit_lower:Q',
                    y2='fit_upper:Q'
                )
                
                ch1, ch2 = st.columns(2)
                ch1.altair_chart((band_acc + line_acc).interactive(), width="stretch")
                ch2.altair_chart((band_fit + line_fit).interactive(), width="stretch")
            else:
                st.info("No hay curvas disponibles para esta configuración exacta.")
    else:
        st.info("No hay curvas disponibles para los filtros seleccionados.")

# --- TABLA DE DATOS ---
st.divider()
st.subheader("📋 Tabla de Resultados Agregados")
st.dataframe(
    filtered_results.sort_values(by="best_accuracy_mean", ascending=False),
    width="stretch",
    hide_index=True
)
