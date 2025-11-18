"""
Aplicación principal de Streamlit para análisis de transacciones de supermercado.
"""

import sys
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import plotly.express as px

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.metrics import MetricsCalculator
from src.visualizations import Visualizer
from src.clustering import CustomerSegmentation

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def load_processed_data():
    """Carga los datos procesados desde CSV."""
    try:
        transactions = pd.read_csv('data/processed/transactions_expanded.csv')
        customer_metrics = pd.read_csv('data/processed/customer_metrics.csv')
        product_metrics = pd.read_csv('data/processed/product_metrics.csv')
        transaction_metrics = pd.read_csv('data/processed/transaction_metrics.csv')
        
        # Convertir fechas
        transactions['date'] = pd.to_datetime(transactions['date'])
        transaction_metrics['date'] = pd.to_datetime(transaction_metrics['date'])
        customer_metrics['last_purchase_date'] = pd.to_datetime(customer_metrics['last_purchase_date'])
        
        return transactions, customer_metrics, product_metrics, transaction_metrics
    except FileNotFoundError:
        st.error("⚠️ No se encontraron datos procesados. Ejecuta primero 'test_data_pipeline.py'")
        st.stop()


def render_sidebar_filters(transactions):
    """Renderiza los filtros en el sidebar."""
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de fecha
    min_date = transactions['date'].min().date()
    max_date = transactions['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Rango de Fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtro de tienda
    stores = ['Todas'] + sorted(transactions['store_id'].unique().tolist())
    selected_store = st.sidebar.selectbox("Tienda", stores)
    
    # Filtro de categoría
    categories = ['Todas'] + sorted(transactions['category_name'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Categoría", categories)
    
    return date_range, selected_store, selected_category


def apply_filters(df, date_range, store, category):
    """Aplica los filtros al DataFrame."""
    filtered = df.copy()
    
    # Filtro de fecha
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['date'].dt.date >= start_date) & 
            (filtered['date'].dt.date <= end_date)
        ]
    
    # Filtro de tienda
    if store != 'Todas':
        filtered = filtered[filtered['store_id'] == int(store)]
    
    # Filtro de categoría
    if category != 'Todas':
        filtered = filtered[filtered['category_name'] == category]
    
    return filtered


def render_dashboard(transactions, customer_metrics, product_metrics):
    """Renderiza el Dashboard Ejecutivo."""
    st.header("📊 Dashboard Ejecutivo")
    st.markdown("---")
    
    # Calcular métricas
    calc = MetricsCalculator()
    kpis = calc.calculate_kpis(transactions)
    
    # Mostrar KPIs principales en 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Unidades",
            value=f"{kpis['total_units']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🛒 Transacciones",
            value=f"{kpis['total_transactions']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="👥 Clientes Únicos",
            value=f"{kpis['total_customers']:,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🏷️ Productos Únicos",
            value=f"{kpis['total_products']:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Métricas adicionales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📈 Avg. Productos/Canasta",
            value=f"{kpis['avg_basket_size']:.2f}"
        )
    
    with col2:
        st.metric(
            label="📊 Avg. Unidades/Transacción",
            value=f"{kpis['avg_units_per_transaction']:.2f}"
        )
    
    with col3:
        days = kpis['date_range']['days']
        st.metric(
            label="📅 Días Analizados",
            value=f"{days}"
        )
    
    st.markdown("---")
    
    # Visualizaciones en 2 columnas
    viz = Visualizer()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Productos")
        fig = viz.plot_top_products(product_metrics, n=10)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("👤 Top 10 Clientes")
        fig = viz.plot_top_customers(customer_metrics, n=10)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Segunda fila de visualizaciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 10 Categorías")
        category_perf = calc.calculate_category_performance(transactions)
        fig = viz.plot_category_distribution(category_perf, n=10, chart_type='bar')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📅 Transacciones por Día de Semana")
        fig = viz.plot_heatmap_day_hour(transactions)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Serie temporal completa
    st.subheader("📈 Evolución Temporal de Ventas")
    
    freq_option = st.radio(
        "Frecuencia",
        options=['Diaria', 'Semanal', 'Mensual'],
        horizontal=True
    )
    
    freq_map = {'Diaria': 'D', 'Semanal': 'W', 'Mensual': 'M'}
    fig = viz.plot_time_series(transactions, freq=freq_map[freq_option], metric='quantity')
    st.plotly_chart(fig, width='stretch')


def render_temporal_analysis(transactions):
    """Renderiza el análisis temporal."""
    st.header("⏰ Análisis Temporal")
    st.markdown("---")
    
    calc = MetricsCalculator()
    viz = Visualizer()
    
    # Serie temporal con diferentes métricas
    st.subheader("📈 Evolución de Métricas")
    
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox(
            "Métrica",
            options=['Unidades Vendidas', 'Transacciones', 'Clientes Únicos']
        )
    with col2:
        freq = st.selectbox(
            "Frecuencia",
            options=['Diaria', 'Semanal', 'Mensual']
        )
    
    metric_map = {
        'Unidades Vendidas': 'quantity',
        'Transacciones': 'transactions',
        'Clientes Únicos': 'customers'
    }
    freq_map = {'Diaria': 'D', 'Semanal': 'W', 'Mensual': 'M'}
    
    fig = viz.plot_time_series(
        transactions,
        freq=freq_map[freq],
        metric=metric_map[metric]
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Patrones temporales
    st.subheader("📊 Patrones Temporales")
    
    patterns = calc.calculate_temporal_patterns(transactions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Por Día de Semana**")
        fig = viz.plot_heatmap_day_hour(transactions)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.write("**Por Mes**")
        month_data = patterns['by_month'].copy()
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        month_data['month_name'] = month_data['month'].apply(lambda x: month_names[int(x)-1])
        
        fig = px.bar(
            month_data,
            x='month_name',
            y='quantity',
            title='Ventas por Mes',
            labels={'quantity': 'Unidades', 'month_name': 'Mes'}
        )
        st.plotly_chart(fig, width='stretch')


def render_distributions_analysis(customer_metrics, product_metrics):
    """Renderiza el análisis de distribuciones."""
    st.header("📊 Análisis de Distribuciones")
    st.markdown("---")
    
    viz = Visualizer()
    
    # Análisis de clientes
    st.subheader("👥 Distribuciones por Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = viz.plot_boxplot(
            customer_metrics,
            column='frequency',
            title='Distribución de Frecuencia de Compra'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = viz.plot_histogram(
            customer_metrics,
            column='total_quantity',
            bins=50,
            title='Distribución de Cantidad Total por Cliente'
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Análisis de productos
    st.subheader("🏷️ Distribuciones por Producto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = viz.plot_boxplot(
            product_metrics,
            column='total_quantity',
            title='Distribución de Ventas por Producto'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = viz.plot_histogram(
            product_metrics,
            column='unique_customers',
            bins=30,
            title='Distribución de Alcance por Producto'
        )
        st.plotly_chart(fig, width='stretch')


def render_correlations_analysis(customer_metrics):
    """Renderiza el análisis de correlaciones."""
    st.header("🔗 Análisis de Correlaciones")
    st.markdown("---")
    
    viz = Visualizer()
    
    # Matriz de correlación
    st.subheader("📊 Matriz de Correlación")
    
    fig = viz.plot_correlation_heatmap(customer_metrics)
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("📈 Relaciones entre Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "Variable X",
            options=['frequency', 'recency', 'total_quantity', 'unique_products']
        )
    
    with col2:
        y_var = st.selectbox(
            "Variable Y",
            options=['total_quantity', 'unique_products', 'frequency', 'recency']
        )
    
    fig = viz.plot_scatter_2d(
        customer_metrics,
        x=x_var,
        y=y_var,
        title=f'{y_var} vs {x_var}'
    )
    st.plotly_chart(fig, width='stretch')


@st.cache_data
def prepare_clustering_data(customer_metrics):
    """Prepara datos para clustering (cached - solo se ejecuta una vez)."""
    segmenter = CustomerSegmentation()
    _, X_scaled = segmenter.prepare_features(customer_metrics)
    return X_scaled, segmenter.features_used


@st.cache_data
def calculate_optimal_k(X_scaled, min_k=2, max_k=8):
    """Calcula K óptimo (cached - solo se ejecuta una vez por rango)."""
    segmenter = CustomerSegmentation()
    # Restaurar las features escaladas
    segmenter.scaled_features = X_scaled
    optimization_results = segmenter.find_optimal_k(X_scaled, min_k=min_k, max_k=max_k)
    return optimization_results


@st.cache_data
def train_clustering_model(_X_scaled, n_clusters):
    """Entrena el modelo de clustering (cached por número de clusters)."""
    segmenter = CustomerSegmentation()
    segmenter.scaled_features = _X_scaled
    labels = segmenter.fit_kmeans(_X_scaled, n_clusters)
    return labels


def render_customer_segmentation(customer_metrics):
    """Renderiza la página de segmentación de clientes."""
    st.header("👥 Segmentación de Clientes")
    st.markdown("---")
    
    st.markdown("""
    La segmentación de clientes utiliza **K-Means Clustering** basado en:
    - 📊 **Frecuencia**: Número de transacciones
    - 🛍️ **Productos únicos**: Diversidad de productos comprados
    - 📦 **Volumen total**: Cantidad total de unidades
    - 🏷️ **Categorías únicas**: Diversidad de categorías
    """)
    
    st.markdown("---")
    
    # Preparar datos (se ejecuta una sola vez y se cachea)
    X_scaled, features_used = prepare_clustering_data(customer_metrics)
    
    col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
    with col_info1:
        st.success(f"✅ Datos preparados: {len(customer_metrics):,} clientes con {len(features_used)} features")
    with col_info2:
        st.info("💾 Usando caché")
    with col_info3:
        if st.button("🔄 Limpiar caché", help="Limpiar todos los resultados cacheados y recalcular"):
            st.cache_data.clear()
            st.rerun()
    
    # Paso 1: Encontrar K óptimo
    st.subheader("🔍 Paso 1: Determinar Número Óptimo de Clusters")
    
    with st.expander("Ver análisis de optimización", expanded=False):
        st.info("ℹ️ Este análisis se calcula una sola vez y se mantiene en caché. No se recalculará al cambiar el número de clusters.")
        
        # Esta optimización se cachea y no se vuelve a calcular
        optimization_results = calculate_optimal_k(X_scaled, min_k=2, max_k=8)
        
        # Crear visualización (necesitamos un segmenter temporal solo para el plot)
        segmenter_temp = CustomerSegmentation()
        fig = segmenter_temp.plot_elbow_method(optimization_results)
        st.plotly_chart(fig, width='stretch')
        
        st.success(f"💡 **K óptimo recomendado**: {optimization_results['recommended_k']} clusters (basado en Silhouette Score)")
    
    st.markdown("---")
    
    # Paso 2: Seleccionar número de clusters
    st.subheader("⚙️ Paso 2: Configurar Segmentación")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        n_clusters = st.slider(
            "Número de Clusters",
            min_value=2,
            max_value=10,
            value=optimization_results['recommended_k'],
            help="Selecciona el número de segmentos de clientes"
        )
    
    with col2:
        viz_type = st.radio(
            "Visualización",
            options=['2D', '3D'],
            horizontal=True
        )
    
    st.markdown("---")
    
    # Paso 3: Entrenar y visualizar
    st.subheader("📊 Paso 3: Resultados de Segmentación")
    
    # Entrenar modelo (se cachea por n_clusters, no se recalcula si ya existe)
    labels = train_clustering_model(X_scaled, n_clusters)
    
    # Crear segmenter para funciones de visualización
    segmenter = CustomerSegmentation()
    segmenter.scaled_features = X_scaled
    segmenter.features_used = features_used
    
    # Reducir dimensionalidad para visualización
    n_components = 3 if viz_type == '3D' else 2
    X_reduced = segmenter.reduce_dimensions(X_scaled, n_components=n_components)
    
    # Perfilar clusters
    profile = segmenter.profile_clusters(customer_metrics, labels)
    cluster_names = segmenter.name_clusters(profile)
    
    # Mostrar métricas de calidad
    col_success1, col_success2 = st.columns([3, 1])
    with col_success1:
        st.success(f"✅ Modelo entrenado exitosamente con {n_clusters} clusters")
    with col_success2:
        st.info("💾 Resultado cacheado")
    
    # Visualización de clusters
    st.markdown("### 🎨 Visualización de Clusters")
    
    if viz_type == '2D':
        fig = segmenter.plot_clusters_2d(X_reduced, labels, cluster_names)
    else:
        fig = segmenter.plot_clusters_3d(X_reduced, labels, cluster_names)
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Paso 4: Perfil de clusters
    st.subheader("📋 Paso 4: Perfil de Cada Segmento")
    
    for cluster_id in sorted(profile['cluster'].unique()):
        cluster_data = profile[profile['cluster'] == cluster_id].iloc[0]
        cluster_info = cluster_names[cluster_id]
        
        with st.expander(f"🔵 Cluster {cluster_id}: {cluster_info['name']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tamaño", f"{int(cluster_data['size']):,}")
                st.metric("Porcentaje", f"{cluster_data['percentage']:.1f}%")
            
            with col2:
                st.metric("Frecuencia Promedio", f"{cluster_data['avg_frequency']:.1f}")
                st.metric("Productos Promedio", f"{cluster_data['avg_products']:.1f}")
            
            with col3:
                st.metric("Volumen Total", f"{int(cluster_data['total_volume']):,}")
                st.metric("Categorías Promedio", f"{cluster_data['avg_categories']:.1f}")
            
            st.markdown(f"**📝 Descripción**: {cluster_info['description']}")
            st.markdown(f"**🎯 Estrategia Recomendada**: {cluster_info['strategy']}")
    
    st.markdown("---")
    
    # Paso 5: Tabla comparativa
    st.subheader("📊 Paso 5: Comparación de Segmentos")
    
    # Agregar nombres a la tabla
    profile_display = profile.copy()
    profile_display['nombre'] = profile_display['cluster'].map(
        lambda x: cluster_names[x]['name']
    )
    
    # Seleccionar columnas relevantes
    display_cols = [
        'cluster', 'nombre', 'size', 'percentage',
        'avg_frequency', 'avg_quantity', 'avg_products', 'avg_categories'
    ]
    
    st.dataframe(
        profile_display[display_cols].style.format({
            'size': '{:,.0f}',
            'percentage': '{:.1f}%',
            'avg_frequency': '{:.1f}',
            'avg_quantity': '{:.1f}',
            'avg_products': '{:.1f}',
            'avg_categories': '{:.1f}'
        }),
        width='stretch'
    ) 
    st.markdown("---")
    
    # Paso 6: Descargar resultados
    st.subheader("💾 Paso 6: Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agregar labels al dataframe de clientes
        customer_segments = customer_metrics.copy()
        customer_segments['cluster'] = labels
        customer_segments['cluster_name'] = customer_segments['cluster'].map(
            lambda x: cluster_names[x]['name']
        )
        
        csv = customer_segments.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Clientes Segmentados (CSV)",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_profile = profile_display.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Perfil de Segmentos (CSV)",
            data=csv_profile,
            file_name="cluster_profiles.csv",
            mime="text/csv"
        )


def main():
    """Punto de entrada principal de la aplicación."""
    st.set_page_config(
        page_title="Análisis de Transacciones",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛒 Análisis de Transacciones de Supermercado")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        transactions, customer_metrics, product_metrics, transaction_metrics = load_processed_data()
    
    # Sidebar con navegación y filtros
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Selecciona una página:",
        [
            "Dashboard Ejecutivo",
            "Análisis Temporal",
            "Análisis de Distribuciones",
            "Análisis de Correlaciones",
            "Segmentación de Clientes",
            "Sistema de Recomendación",
            "Carga de Nuevos Datos"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Filtros (solo para algunas páginas)
    if page in ["Dashboard Ejecutivo", "Análisis Temporal"]:
        date_range, selected_store, selected_category = render_sidebar_filters(transactions)
        transactions_filtered = apply_filters(transactions, date_range, selected_store, selected_category)
    else:
        transactions_filtered = transactions
    
    # Renderizar página seleccionada
    if page == "Dashboard Ejecutivo":
        render_dashboard(transactions_filtered, customer_metrics, product_metrics)
    
    elif page == "Análisis Temporal":
        render_temporal_analysis(transactions_filtered)
    
    elif page == "Análisis de Distribuciones":
        render_distributions_analysis(customer_metrics, product_metrics)
    
    elif page == "Análisis de Correlaciones":
        render_correlations_analysis(customer_metrics)
    
    elif page == "Segmentación de Clientes":
        render_customer_segmentation(customer_metrics)
    
    else:
        st.info(f"⚠️ Página **{page}** en desarrollo")
        st.write("Esta funcionalidad se implementará próximamente.")


if __name__ == "__main__":
    main()
