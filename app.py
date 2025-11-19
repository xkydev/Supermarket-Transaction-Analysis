"""
Aplicación principal de Streamlit para análisis de transacciones de supermercado.
"""

import sys
from pathlib import Path
import logging
from typing import Optional
import shutil

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
from src.recommender import RecommenderSystem
from config import Paths, ClusteringConfig, VisualizationConfig, RecommenderConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES CACHEADAS PARA SISTEMA DE RECOMENDACIONES
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="🔗 Generando reglas de asociación...")
def build_cached_association_rules(
    _transactions_df,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_transactions: Optional[int] = None
):
    """Construye y cachea las reglas de asociación de Apriori.
    
    Args:
        _transactions_df: DataFrame de transacciones
        min_support: Soporte mínimo
        min_confidence: Confianza mínima
        min_lift: Lift mínimo
        max_transactions: Máximo de transacciones a analizar
    
    Returns:
        DataFrame con reglas de asociación
    """
    recommender = RecommenderSystem()
    rules = recommender.build_association_rules(
        transactions_df=_transactions_df,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_transactions=max_transactions
    )
    return rules


@st.cache_data(ttl=1800, show_spinner="🎯 Calculando recomendaciones...")
def get_cached_customer_recommendations(
    customer_id: int,
    _transactions_df,
    top_n: int = 10,
    top_k_similar: int = 1000,
    min_similarity: float = 0.1
):
    """Obtiene recomendaciones para un cliente usando caché.
    
    Args:
        customer_id: ID del cliente
        _transactions_df: DataFrame de transacciones
        top_n: Número de recomendaciones
        top_k_similar: Número de clientes similares a considerar
        min_similarity: Similitud mínima
    
    Returns:
        DataFrame con recomendaciones
    """
    recommender = RecommenderSystem()
    recommendations = recommender.get_customer_recommendations(
        customer_id=customer_id,
        transactions_df=_transactions_df,
        top_n=top_n,
        top_k_similar=top_k_similar,
        min_similarity=min_similarity
    )
    return recommendations


@st.cache_data(ttl=1800, show_spinner="🎯 Calculando recomendaciones de productos...")
def get_cached_product_recommendations(
    product_id: int,
    _association_rules,
    _transactions_df,
    top_n: int = 10
):
    """Obtiene recomendaciones para un producto usando reglas cacheadas.
    
    Args:
        product_id: ID del producto
        _association_rules: DataFrame con reglas de asociación
        _transactions_df: DataFrame de transacciones
        top_n: Número de recomendaciones
    
    Returns:
        DataFrame con recomendaciones
    """
    recommender = RecommenderSystem()
    # Filtrar reglas donde el producto está en el antecedente
    product_rules = _association_rules[
        _association_rules['antecedents'].apply(lambda x: product_id in x)
    ].copy()
    
    if product_rules.empty:
        return pd.DataFrame()
    
    # Extraer productos recomendados del consecuente
    recommendations = []
    for _, rule in product_rules.iterrows():
        for consequent_id in rule['consequents']:
            if consequent_id != product_id:
                recommendations.append({
                    'product_id': consequent_id,
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift']
                })
    
    if not recommendations:
        return pd.DataFrame()
    
    # Convertir a DataFrame y agregar por producto
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.groupby('product_id').agg({
        'support': 'mean',
        'confidence': 'max',
        'lift': 'max'
    }).reset_index()
    
    # Ordenar por lift y tomar top N
    recommendations_df = recommendations_df.sort_values('lift', ascending=False).head(top_n)
    
    # Enriquecer con información del producto
    # Verificar qué columnas están disponibles
    available_columns = ['product_id']
    if 'product_name' in _transactions_df.columns:
        available_columns.append('product_name')
    if 'category_name' in _transactions_df.columns:
        available_columns.append('category_name')
    
    product_info = _transactions_df[available_columns].drop_duplicates(subset=['product_id'])
    recommendations_df = recommendations_df.merge(product_info, on='product_id', how='left')
    
    # Generar nombres si no existen
    if 'product_name' not in recommendations_df.columns:
        recommendations_df['product_name'] = recommendations_df['product_id'].apply(lambda x: f"Producto {x}")
    if 'category_name' not in recommendations_df.columns:
        recommendations_df['category_name'] = "Sin categoría"
    
    # Calcular score (combinación de lift, confidence y support)
    recommendations_df['score'] = (
        recommendations_df['lift'] * 0.5 +
        recommendations_df['confidence'] * 100 * 0.3 +
        recommendations_df['support'] * 100 * 0.2
    )
    
    return recommendations_df


@st.cache_data
def load_processed_data():
    """Carga los datos procesados desde CSV."""
    transactions = pd.read_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv')
    customer_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv')
    product_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'product_metrics.csv')
    transaction_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'transaction_metrics.csv')
    
    # Convertir fechas
    transactions['date'] = pd.to_datetime(transactions['date'])
    transaction_metrics['date'] = pd.to_datetime(transaction_metrics['date'])
    customer_metrics['last_purchase_date'] = pd.to_datetime(customer_metrics['last_purchase_date'])
    
    return transactions, customer_metrics, product_metrics, transaction_metrics


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
    _, x_scaled = segmenter.prepare_features(customer_metrics)
    return x_scaled, segmenter.features_used


@st.cache_data
def calculate_optimal_k(x_scaled, min_k=2, max_k=8):
    """Calcula K óptimo (cached - solo se ejecuta una vez por rango)."""
    segmenter = CustomerSegmentation()
    # Restaurar las features escaladas
    segmenter.scaled_features = x_scaled
    optimization_results = segmenter.find_optimal_k(x_scaled, min_k=min_k, max_k=max_k)
    return optimization_results


@st.cache_data
def train_clustering_model(_x_scaled, n_clusters):
    """Entrena el modelo de clustering (cached por número de clusters)."""
    segmenter = CustomerSegmentation()
    segmenter.scaled_features = _x_scaled
    labels = segmenter.fit_kmeans(_x_scaled, n_clusters)
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
    x_scaled, features_used = prepare_clustering_data(customer_metrics)
    
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
        optimization_results = calculate_optimal_k(x_scaled, min_k=ClusteringConfig.MIN_CLUSTERS, max_k=ClusteringConfig.MAX_CLUSTERS)
        
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
            min_value=ClusteringConfig.MIN_CLUSTERS,
            max_value=ClusteringConfig.MAX_CLUSTERS,
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
    labels = train_clustering_model(x_scaled, n_clusters)
    
    # Crear segmenter para funciones de visualización
    segmenter = CustomerSegmentation()
    segmenter.scaled_features = x_scaled
    segmenter.features_used = features_used
    
    # Reducir dimensionalidad para visualización
    n_components = 3 if viz_type == '3D' else 2
    x_reduced = segmenter.reduce_dimensions(x_scaled, n_components=n_components)
    
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
        fig = segmenter.plot_clusters_2d(x_reduced, labels, cluster_names)
    else:
        fig = segmenter.plot_clusters_3d(x_reduced, labels, cluster_names)
    
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


def render_data_upload():
    """Renderiza la página de Carga de Nuevos Datos."""
    st.header("📤 Carga de Nuevos Datos")
    
    st.markdown("""
    Esta funcionalidad permite cargar nuevos archivos de transacciones para actualizar el análisis.
    Los datos se validarán automáticamente antes de ser procesados.
    """)
    
    st.markdown("---")
    
    # Información sobre el formato esperado
    with st.expander("ℹ️ Formato de Datos Esperado", expanded=False):
        st.markdown("""
        **Archivos soportados:**
        
        1. **Archivos de Transacciones** (formato: `XXX_Tran.csv`)
           - Columnas requeridas: `date`, `store_id`, `customer_id`, `products`
           - Formato de fecha: `YYYY-MM-DD` (ejemplo: 2013-01-01)
           - Formato de productos: IDs separados por espacios (ejemplo: "20 3 1 5")
           - El nombre del archivo debe indicar el ID de tienda (ejemplo: 102_Tran.csv)
        
        2. **Categories.csv** (opcional - para actualizar categorías)
           - Columnas: `category_id`, `category_name`
        
        3. **ProductCategory.csv** (opcional - para actualizar relaciones)
           - Columnas: `product_id`, `category_id`
        
        **Nota:** Los archivos se validarán antes de procesarse.
        """)
    
    st.markdown("---")
    
    # Tabs para diferentes tipos de carga
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cargar Transacciones", 
        "🏷️ Cargar Categorías",
        "🔗 Cargar Producto-Categoría",
        "📈 Ver Estado Actual"
    ])
    
    with tab1:
        st.subheader("Cargar Nuevo Archivo de Transacciones")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV",
            type=['csv'],
            help="Archivo de transacciones en formato XXX_Tran.csv"
        )
        
        if uploaded_file is not None:
            # Mostrar información del archivo
            st.info(f"📁 Archivo: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
            
            try:
                # Leer el archivo
                import io
                file_content = uploaded_file.getvalue().decode('utf-8')
                
                # Detectar delimitador
                delimiter = ','
                if '|' in file_content.split('\n')[0]:
                    delimiter = '|'
                
                # Intentar leer con encabezados primero
                uploaded_df = pd.read_csv(io.StringIO(file_content), sep=delimiter)
                
                # Validar estructura
                st.markdown("### ✅ Paso 1: Validación de Estructura")
                
                required_columns = ['date', 'store_id', 'customer_id', 'products']
                
                # Detectar si el archivo no tiene encabezados (columnas numéricas 0, 1, 2, 3)
                # o si tiene columnas pero no coinciden con las esperadas
                if (uploaded_df.columns.tolist() == [0, 1, 2, 3] or 
                    all(str(col).isdigit() for col in uploaded_df.columns) or
                    not any(col in uploaded_df.columns for col in required_columns)):
                    
                    st.warning("⚠️ Archivo sin encabezados detectado. Asignando nombres de columnas...")
                    
                    # Verificar que tenga exactamente 4 columnas
                    if len(uploaded_df.columns) != 4:
                        st.error(f"❌ El archivo debe tener exactamente 4 columnas, pero tiene {len(uploaded_df.columns)}")
                        st.info("Formato esperado: date, store_id, customer_id, products")
                        st.stop()
                    
                    # Asignar nombres de columnas correctos
                    uploaded_df.columns = required_columns
                    st.success(f"✅ Nombres asignados automáticamente. Delimitador detectado: '{delimiter}'")
                else:
                    # Verificar columnas faltantes
                    missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
                    
                    if missing_columns:
                        st.error(f"❌ Columnas faltantes: {', '.join(missing_columns)}")
                        st.info(f"📋 Columnas encontradas: {', '.join(uploaded_df.columns.tolist())}")
                        st.info(f"📋 Columnas requeridas: {', '.join(required_columns)}")
                        st.stop()
                    else:
                        st.success("✅ Todas las columnas requeridas están presentes")
                
                # Mostrar preview
                st.markdown("### 📋 Paso 2: Preview de Datos")
                st.dataframe(uploaded_df.head(10), width='stretch')
                
                # Mostrar estadísticas básicas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filas", f"{len(uploaded_df):,}")
                with col2:
                    st.metric("Clientes Únicos", f"{uploaded_df['customer_id'].nunique():,}")
                with col3:
                    st.metric("Tiendas", f"{uploaded_df['store_id'].nunique()}")
                with col4:
                    # Contar productos únicos
                    all_products = set()
                    for products_str in uploaded_df['products'].dropna():
                        all_products.update(products_str.split())
                    st.metric("Productos Únicos", f"{len(all_products):,}")
                
                st.markdown("---")
                
                # Validaciones adicionales
                st.markdown("### 🔍 Paso 3: Validaciones Adicionales")
                
                validations = []
                
                # Validar fechas
                try:
                    pd.to_datetime(uploaded_df['date'])
                    validations.append(("Formato de fechas", True, "Fechas válidas"))
                except:
                    validations.append(("Formato de fechas", False, "Error en formato de fechas"))
                
                # Validar valores nulos
                null_counts = uploaded_df[required_columns].isnull().sum()
                has_nulls = null_counts.sum() > 0
                validations.append((
                    "Valores nulos",
                    not has_nulls,
                    "Sin valores nulos" if not has_nulls else f"Encontrados valores nulos: {null_counts[null_counts > 0].to_dict()}"
                ))
                
                # Validar IDs positivos
                positive_ids = (uploaded_df['customer_id'] > 0).all() and (uploaded_df['store_id'] > 0).all()
                validations.append((
                    "IDs válidos",
                    positive_ids,
                    "Todos los IDs son positivos" if positive_ids else "Algunos IDs son negativos o cero"
                ))
                
                # Mostrar resultados de validación
                for validation_name, is_valid, message in validations:
                    if is_valid:
                        st.success(f"✅ **{validation_name}**: {message}")
                    else:
                        st.error(f"❌ **{validation_name}**: {message}")
                
                all_valid = all(v[1] for v in validations)
                
                st.markdown("---")
                
                # Opciones de procesamiento
                st.markdown("### ⚙️ Paso 4: Opciones de Procesamiento")
                
                col1, col2 = st.columns(2)
                with col1:
                    process_mode = st.radio(
                        "Modo de procesamiento:",
                        ["Agregar a datos existentes", "Reemplazar datos existentes"],
                        help="Agregar: añade a los datos actuales. Reemplazar: sobrescribe completamente."
                    )
                
                with col2:
                    recalculate_all = st.checkbox(
                        "Recalcular todas las métricas",
                        value=True,
                        help="Recalcula customer_metrics, product_metrics, etc."
                    )
                
                st.markdown("---")
                
                # Botón de procesamiento
                if all_valid:
                    if st.button("🚀 Procesar y Actualizar Datos", type="primary", use_container_width=True):
                        with st.spinner("Procesando datos..."):
                            try:
                                # Crear progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Paso 1: Cargar datos existentes
                                status_text.text("Cargando datos existentes...")
                                progress_bar.progress(10)
                                
                                loader = DataLoader()
                                processor = DataProcessor()
                                
                                # Paso 2: Procesar nuevo archivo
                                status_text.text("Procesando nuevo archivo...")
                                progress_bar.progress(30)
                                
                                # Expandir transacciones del nuevo archivo
                                status_text.text("Expandiendo transacciones...")
                                new_transactions = processor.expand_transactions(uploaded_df)
                                
                                progress_bar.progress(40)
                                
                                # Enriquecer con categorías
                                status_text.text("Enriqueciendo con categorías...")
                                product_category = loader.load_product_category()
                                categories = loader.load_categories()
                                new_transactions = processor.enrich_with_categories(
                                    new_transactions, 
                                    product_category, 
                                    categories
                                )
                                
                                progress_bar.progress(55)
                                
                                # Agregar features temporales
                                status_text.text("Agregando features temporales...")
                                new_transactions = processor.add_temporal_features(new_transactions)
                                
                                progress_bar.progress(70)
                                
                                # Combinar con datos existentes si es modo agregar
                                if process_mode == "Agregar a datos existentes":
                                    status_text.text("Combinando con datos existentes...")
                                    existing_transactions = pd.read_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv')
                                    combined_transactions = pd.concat([existing_transactions, new_transactions], ignore_index=True)
                                else:
                                    combined_transactions = new_transactions
                                
                                progress_bar.progress(80)
                                
                                # Guardar datos actualizados
                                status_text.text("Guardando datos actualizados...")
                                combined_transactions.to_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv', index=False)
                                
                                progress_bar.progress(90)
                                
                                # Recalcular métricas si se solicita
                                if recalculate_all:
                                    status_text.text("Recalculando métricas...")
                                    
                                    customer_metrics = processor.calculate_customer_metrics(combined_transactions)
                                    customer_metrics.to_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv', index=False)
                                    
                                    product_metrics = processor.calculate_product_metrics(combined_transactions)
                                    product_metrics.to_csv(Paths.DATA_PROCESSED / 'product_metrics.csv', index=False)
                                    
                                    transaction_metrics = processor.calculate_transaction_metrics(combined_transactions)
                                    transaction_metrics.to_csv(Paths.DATA_PROCESSED / 'transaction_metrics.csv', index=False)
                                
                                progress_bar.progress(100)
                                status_text.text("✅ Procesamiento completado!")
                                
                                # Limpiar caché
                                st.cache_data.clear()
                                
                                # Mostrar comparación antes/después
                                st.markdown("---")
                                st.markdown("### 📊 Resumen de Actualización")
                                
                                # Calcular deltas
                                if process_mode == "Agregar a datos existentes":
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        new_rows = len(new_transactions)
                                        st.metric(
                                            "Transacciones Agregadas",
                                            f"{new_rows:,}",
                                            delta=f"+{new_rows:,}"
                                        )
                                    
                                    with col2:
                                        new_customers = new_transactions['customer_id'].nunique()
                                        st.metric(
                                            "Nuevos Clientes",
                                            f"{new_customers:,}",
                                            delta=f"+{new_customers:,}"
                                        )
                                    
                                    with col3:
                                        new_products = new_transactions['product_id'].nunique()
                                        st.metric(
                                            "Nuevos Productos",
                                            f"{new_products:,}",
                                            delta=f"+{new_products:,}"
                                        )
                                
                                st.success("✅ Datos actualizados exitosamente. Recarga la página para ver los cambios.")
                                
                                # Botón para recargar
                                if st.button("🔄 Recargar Aplicación"):
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error al procesar datos: {str(e)}")
                                logger.error(f"Error en carga de datos: {e}", exc_info=True)
                else:
                    st.warning("⚠️ Por favor corrige los errores de validación antes de procesar.")
                    
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {str(e)}")
                logger.error(f"Error en lectura de archivo: {e}", exc_info=True)
    
    with tab2:
        st.subheader("Cargar Archivo de Categorías")
        
        st.markdown("""
        Sube un archivo `Categories.csv` para actualizar el catálogo de categorías.
        
        **Formato esperado:**
        - `category_id`: ID numérico de la categoría
        - `category_name`: Nombre descriptivo de la categoría
        """)
        
        categories_file = st.file_uploader(
            "Selecciona archivo Categories.csv",
            type=['csv'],
            key='categories_uploader',
            help="Archivo con categorías de productos"
        )
        
        if categories_file is not None:
            st.info(f"📁 Archivo: **{categories_file.name}** ({categories_file.size / 1024:.2f} KB)")
            
            try:
                import io
                file_content = categories_file.getvalue().decode('utf-8')
                
                # Intentar detectar delimitador
                delimiter = ','
                if '|' in file_content.split('\n')[0]:
                    delimiter = '|'
                
                # Intentar leer con encabezados primero
                categories_df = pd.read_csv(io.StringIO(file_content), sep=delimiter)
                
                st.markdown("### ✅ Validación de Estructura")
                
                required_columns = ['category_id', 'category_name']
                
                # Detectar si no tiene encabezados
                if (categories_df.columns.tolist() == [0, 1] or 
                    all(str(col).isdigit() for col in categories_df.columns) or
                    not any(col in categories_df.columns for col in required_columns)):
                    
                    st.warning("⚠️ Archivo sin encabezados detectado. Asignando nombres de columnas...")
                    
                    # Verificar que tenga exactamente 2 columnas
                    if len(categories_df.columns) != 2:
                        st.error(f"❌ El archivo debe tener exactamente 2 columnas, pero tiene {len(categories_df.columns)}")
                        st.info("Formato esperado: category_id, category_name")
                        st.stop()
                    
                    # Asignar nombres de columnas correctos
                    categories_df.columns = required_columns
                    st.success(f"✅ Nombres asignados automáticamente. Delimitador detectado: '{delimiter}'")
                else:
                    # Verificar columnas faltantes
                    missing_columns = [col for col in required_columns if col not in categories_df.columns]
                    
                    if missing_columns:
                        st.error(f"❌ Columnas faltantes: {', '.join(missing_columns)}")
                        st.info(f"📋 Columnas requeridas: {', '.join(required_columns)}")
                        st.stop()
                
                st.success("✅ Estructura válida")
                
                # Preview
                st.markdown("### 📋 Preview de Datos")
                st.dataframe(categories_df.head(20), width='stretch')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Categorías", f"{len(categories_df):,}")
                with col2:
                    st.metric("Categorías Únicas", f"{categories_df['category_id'].nunique():,}")
                
                # Validaciones
                st.markdown("### ✅ Validaciones")
                validation_passed = True
                
                # 1. IDs duplicados
                duplicates = categories_df['category_id'].duplicated().sum()
                if duplicates > 0:
                    st.error(f"❌ Se encontraron {duplicates} IDs de categoría duplicados")
                    validation_passed = False
                else:
                    st.success("✅ No hay IDs duplicados")
                
                # 2. Valores nulos
                nulls = categories_df.isnull().sum().sum()
                if nulls > 0:
                    st.error(f"❌ Se encontraron {nulls} valores nulos")
                    validation_passed = False
                else:
                    st.success("✅ No hay valores nulos")
                
                # 3. IDs positivos
                if (categories_df['category_id'] <= 0).any():
                    st.error("❌ Algunos category_id no son positivos")
                    validation_passed = False
                else:
                    st.success("✅ Todos los IDs son positivos")
                
                if validation_passed:
                    st.markdown("---")
                    
                    # Modo de actualización
                    update_mode = st.radio(
                        "Modo de actualización",
                        ["Agregar nuevas categorías", "Reemplazar todas las categorías"],
                        help="Agregar: combina con categorías existentes. Reemplazar: elimina las actuales."
                    )
                    
                    if st.button("🚀 Procesar Categorías", type="primary"):
                        with st.spinner("Procesando..."):
                            try:
                                if update_mode == "Agregar nuevas categorías":
                                    # Cargar categorías existentes
                                    existing_categories = pd.read_csv(Paths.DATA_RAW / 'Categories.csv')
                                    
                                    # Combinar (las nuevas sobrescriben las existentes con mismo ID)
                                    combined_categories = pd.concat([existing_categories, categories_df])
                                    combined_categories = combined_categories.drop_duplicates(
                                        subset=['category_id'], 
                                        keep='last'
                                    ).sort_values('category_id')
                                else:
                                    combined_categories = categories_df.sort_values('category_id')
                                
                                # Guardar
                                combined_categories.to_csv(Paths.DATA_RAW / 'Categories.csv', index=False)
                                
                                st.success("✅ Categorías actualizadas exitosamente!")
                                st.info(f"📊 Total de categorías: {len(combined_categories):,}")
                                
                                # Mostrar cambios
                                if update_mode == "Agregar nuevas categorías":
                                    new_count = len(combined_categories) - len(existing_categories)
                                    if new_count > 0:
                                        st.success(f"➕ {new_count} nuevas categorías agregadas")
                                    updated_count = len(categories_df) - new_count
                                    if updated_count > 0:
                                        st.info(f"🔄 {updated_count} categorías actualizadas")
                                
                                st.warning("⚠️ **Importante:** Debes reprocesar las transacciones para que los cambios se reflejen en el sistema.")
                                
                                if st.button("🔄 Recargar Aplicación"):
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error al procesar: {str(e)}")
                                logger.error(f"Error en carga de categorías: {e}", exc_info=True)
                else:
                    st.warning("⚠️ Por favor corrige los errores de validación.")
                    
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {str(e)}")
                logger.error(f"Error en lectura de categorías: {e}", exc_info=True)
    
    with tab3:
        st.subheader("Cargar Archivo de Producto-Categoría")
        
        st.markdown("""
        Sube un archivo `ProductCategory.csv` para actualizar las relaciones producto-categoría.
        
        **Formato esperado:**
        - `product_id`: ID numérico del producto
        - `category_id`: ID numérico de la categoría
        """)
        
        product_category_file = st.file_uploader(
            "Selecciona archivo ProductCategory.csv",
            type=['csv'],
            key='product_category_uploader',
            help="Archivo con relaciones producto-categoría"
        )
        
        if product_category_file is not None:
            st.info(f"📁 Archivo: **{product_category_file.name}** ({product_category_file.size / 1024:.2f} KB)")
            
            try:
                import io
                file_content = product_category_file.getvalue().decode('utf-8')
                
                # Intentar detectar delimitador
                delimiter = ','
                if '|' in file_content.split('\n')[0]:
                    delimiter = '|'
                
                # Intentar leer con encabezados primero
                product_category_df = pd.read_csv(io.StringIO(file_content), sep=delimiter)
                
                st.markdown("### ✅ Validación de Estructura")
                
                required_columns = ['product_id', 'category_id']
                
                # Detectar si no tiene encabezados
                if (product_category_df.columns.tolist() == [0, 1] or 
                    all(str(col).isdigit() for col in product_category_df.columns) or
                    not any(col in product_category_df.columns for col in required_columns)):
                    
                    st.warning("⚠️ Archivo sin encabezados detectado. Asignando nombres de columnas...")
                    
                    # Verificar que tenga exactamente 2 columnas
                    if len(product_category_df.columns) != 2:
                        st.error(f"❌ El archivo debe tener exactamente 2 columnas, pero tiene {len(product_category_df.columns)}")
                        st.info("Formato esperado: product_id, category_id")
                        st.stop()
                    
                    # Asignar nombres de columnas correctos
                    product_category_df.columns = required_columns
                    st.success(f"✅ Nombres asignados automáticamente. Delimitador detectado: '{delimiter}'")
                else:
                    # Verificar columnas faltantes
                    missing_columns = [col for col in required_columns if col not in product_category_df.columns]
                    
                    if missing_columns:
                        st.error(f"❌ Columnas faltantes: {', '.join(missing_columns)}")
                        st.info(f"📋 Columnas requeridas: {', '.join(required_columns)}")
                        st.stop()
                
                st.success("✅ Estructura válida")
                
                # Preview
                st.markdown("### 📋 Preview de Datos")
                st.dataframe(product_category_df.head(20), width='stretch')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Relaciones", f"{len(product_category_df):,}")
                with col2:
                    st.metric("Productos Únicos", f"{product_category_df['product_id'].nunique():,}")
                
                # Validaciones
                st.markdown("### ✅ Validaciones")
                validation_passed = True
                
                # 1. IDs duplicados
                duplicates = product_category_df['product_id'].duplicated().sum()
                if duplicates > 0:
                    st.warning(f"⚠️ Se encontraron {duplicates} product_id duplicados (se mantendrá la última asignación)")
                
                # 2. Valores nulos
                nulls = product_category_df.isnull().sum().sum()
                if nulls > 0:
                    st.error(f"❌ Se encontraron {nulls} valores nulos")
                    validation_passed = False
                else:
                    st.success("✅ No hay valores nulos")
                
                # 3. IDs positivos
                if (product_category_df['product_id'] <= 0).any() or (product_category_df['category_id'] <= 0).any():
                    st.error("❌ Algunos IDs no son positivos")
                    validation_passed = False
                else:
                    st.success("✅ Todos los IDs son positivos")
                
                # 4. Verificar categorías existentes
                try:
                    # Leer categorías existentes (formato con pipe y sin encabezados)
                    existing_categories = pd.read_csv(
                        Paths.DATA_RAW / 'Categories.csv', 
                        sep='|', 
                        header=None, 
                        names=['category_id', 'category_name']
                    )
                    
                    unknown_categories = set(product_category_df['category_id']) - set(existing_categories['category_id'])
                    if unknown_categories:
                        st.warning(f"⚠️ {len(unknown_categories)} categorías no existen en Categories.csv: {sorted(list(unknown_categories)[:10])}")
                        st.info("💡 Considera subir primero el archivo de categorías actualizado")
                    else:
                        st.success("✅ Todas las categorías existen")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo verificar categorías: {str(e)}")
                
                if validation_passed:
                    st.markdown("---")
                    
                    # Modo de actualización
                    update_mode = st.radio(
                        "Modo de actualización",
                        ["Agregar/Actualizar productos", "Reemplazar todas las relaciones"],
                        help="Agregar: combina con productos existentes. Reemplazar: elimina las actuales."
                    )
                    
                    if st.button("🚀 Procesar Producto-Categoría", type="primary"):
                        with st.spinner("Procesando..."):
                            try:
                                if update_mode == "Agregar/Actualizar productos":
                                    # Cargar relaciones existentes
                                    existing_pc = pd.read_csv(Paths.DATA_RAW / 'ProductCategory.csv')
                                    
                                    # Combinar (las nuevas sobrescriben las existentes con mismo product_id)
                                    combined_pc = pd.concat([existing_pc, product_category_df])
                                    combined_pc = combined_pc.drop_duplicates(
                                        subset=['product_id'], 
                                        keep='last'
                                    ).sort_values('product_id')
                                else:
                                    combined_pc = product_category_df.sort_values('product_id')
                                
                                # Guardar
                                combined_pc.to_csv(Paths.DATA_RAW / 'ProductCategory.csv', index=False)
                                
                                st.success("✅ Relaciones producto-categoría actualizadas exitosamente!")
                                st.info(f"📊 Total de productos: {len(combined_pc):,}")
                                
                                # Mostrar cambios
                                if update_mode == "Agregar/Actualizar productos":
                                    new_count = len(combined_pc) - len(existing_pc)
                                    if new_count > 0:
                                        st.success(f"➕ {new_count} nuevos productos agregados")
                                    updated_count = len(product_category_df) - new_count
                                    if updated_count > 0:
                                        st.info(f"🔄 {updated_count} productos actualizados")
                                
                                st.warning("⚠️ **Importante:** Debes reprocesar las transacciones para que los cambios se reflejen en el sistema.")
                                
                                if st.button("🔄 Recargar Aplicación"):
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error al procesar: {str(e)}")
                                logger.error(f"Error en carga de producto-categoría: {e}", exc_info=True)
                else:
                    st.warning("⚠️ Por favor corrige los errores de validación.")
                    
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {str(e)}")
                logger.error(f"Error en lectura de producto-categoría: {e}", exc_info=True)
    
    with tab4:
        st.subheader("Estado Actual de los Datos")
        
        try:
            # Cargar datos actuales
            transactions_current = pd.read_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv')
            
            st.markdown("### 📊 Resumen de Datos Actuales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transacciones", f"{len(transactions_current):,}")
            
            with col2:
                st.metric("Clientes Únicos", f"{transactions_current['customer_id'].nunique():,}")
            
            with col3:
                st.metric("Productos Únicos", f"{transactions_current['product_id'].nunique():,}")
            
            with col4:
                st.metric("Tiendas", f"{transactions_current['store_id'].nunique()}")
            
            st.markdown("---")
            
            # Información de fechas
            transactions_current['date'] = pd.to_datetime(transactions_current['date'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fecha Mínima", transactions_current['date'].min().strftime('%Y-%m-%d'))
            
            with col2:
                st.metric("Fecha Máxima", transactions_current['date'].max().strftime('%Y-%m-%d'))
            
            with col3:
                days_span = (transactions_current['date'].max() - transactions_current['date'].min()).days
                st.metric("Días de Datos", f"{days_span:,}")
            
            st.markdown("---")
            
            # Tamaño de archivos
            st.markdown("### 💾 Almacenamiento")
            
            processed_files = {
                'transactions_expanded.csv': 'Transacciones Expandidas',
                'customer_metrics.csv': 'Métricas de Clientes',
                'product_metrics.csv': 'Métricas de Productos',
                'transaction_metrics.csv': 'Métricas de Transacciones'
            }
            
            file_sizes = []
            for filename, description in processed_files.items():
                file_path = Paths.DATA_PROCESSED / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    file_sizes.append({
                        'Archivo': description,
                        'Tamaño (MB)': f"{size_mb:.2f}"
                    })
            
            if file_sizes:
                st.dataframe(pd.DataFrame(file_sizes), width='stretch', hide_index=True)
            
            st.markdown("---")
            
            # Botones de acción
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Limpiar Caché de la Aplicación", use_container_width=True):
                    st.cache_data.clear()
                    st.success("✅ Caché limpiado exitosamente")
            
            with col2:
                if st.button("🔄 Resetear a Datos Originales", type="primary", use_container_width=True):
                    with st.spinner("⏳ Restaurando y procesando datos originales..."):
                        try:
                            # Eliminar archivos procesados
                            processed_dir = Paths.DATA_PROCESSED
                            
                            if processed_dir.exists():
                                shutil.rmtree(processed_dir)
                            processed_dir.mkdir(parents=True)
                            
                            # Ejecutar pipeline de procesamiento
                            loader = DataLoader()
                            processor = DataProcessor()
                            
                            # Cargar datos raw
                            transactions_raw = loader.load_transactions()
                            categories = loader.load_categories()
                            product_category = loader.load_product_category()
                            
                            # Procesar todos los datos usando process_all
                            processed_data = processor.process_all(transactions_raw, product_category, categories)
                            
                            # Guardar datos procesados
                            processed_data['transactions_expanded'].to_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv', index=False)
                            processed_data['customer_metrics'].to_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv', index=False)
                            processed_data['product_metrics'].to_csv(Paths.DATA_PROCESSED / 'product_metrics.csv', index=False)
                            processed_data['transaction_metrics'].to_csv(Paths.DATA_PROCESSED / 'transaction_metrics.csv', index=False)
                            
                            # Limpiar caché
                            st.cache_data.clear()
                            
                            st.success("✅ Datos reseteados y procesados correctamente!")
                            st.info("💡 Recarga la aplicación con F5 para ver los cambios")
                            
                        except Exception as e:
                            st.error(f"❌ Error al resetear datos: {str(e)}")
                            logger.error(f"Error en reseteo de datos: {e}", exc_info=True)
        
        except FileNotFoundError:
            st.warning("⚠️ No se encontraron datos procesados. Carga algunos datos primero.")


def render_recommendations(transactions):
    """Renderiza la página de Sistema de Recomendación."""
    st.header("🎯 Sistema de Recomendación")
    
    # Información sobre caché
    with st.expander("ℹ️ Información sobre el Sistema de Caché", expanded=False):
        st.markdown("""
        **🚀 Optimización de Rendimiento:**
        
        Este sistema utiliza caché inteligente para acelerar las recomendaciones:
        
        - **🔗 Reglas de Asociación**: Se cachean según los parámetros de configuración (TTL: 1 hora)  
        - **🎯 Recomendaciones por Cliente**: Se cachean por cliente y parámetros (TTL: 30 minutos)
        - **🎯 Recomendaciones por Producto**: Se cachean por producto y parámetros (TTL: 30 minutos)
        
        **Beneficios:**
        - ⚡ Primera consulta: 2-5 minutos (dependiendo de parámetros)
        - ⚡ Consultas subsecuentes: < 1 segundo
        - 💾 Limpieza automática del caché después del tiempo de vida (TTL)
        
        **Nota:** Si cambias los parámetros de configuración, se regenerará el caché correspondiente.
        """)
        
        if st.button("🗑️ Limpiar Caché del Sistema de Recomendaciones"):
            # Limpiar funciones específicas del caché
            build_cached_association_rules.clear()
            get_cached_customer_recommendations.clear()
            get_cached_product_recommendations.clear()
            st.success("✅ Caché limpiado exitosamente. Las próximas consultas recalcularán los datos.")
            st.rerun()
    
    st.markdown("---")
    
    # Selector de tipo de recomendación
    st.subheader("Selecciona el tipo de recomendación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rec_type = st.radio(
            "Tipo de Recomendación",
            options=["Por Cliente", "Por Producto"],
            horizontal=True
        )
    
    st.markdown("---")
    
    # Inicializar recomendador
    recommender = RecommenderSystem()
    
    if rec_type == "Por Cliente":
        st.subheader("🛍️ Recomendaciones Basadas en Cliente")
        st.info(
            "💡 Este sistema utiliza **Filtrado Colaborativo** para recomendar productos. "
            "Analiza clientes con patrones de compra similares y sugiere productos que ellos han comprado."
        )
        
        # Input de cliente
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            customer_id = st.number_input(
                "ID del Cliente",
                min_value=1,
                value=1000,
                help="Ingresa el ID del cliente para obtener recomendaciones"
            )
        
        with col2:
            top_n = st.number_input(
                "Top N",
                min_value=1,
                max_value=50,
                value=RecommenderConfig.TOP_N,
                help="Número de productos a recomendar"
            )
        
        with col3:
            min_similarity = st.slider(
                "Similaridad Mínima",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Umbral de similaridad entre clientes"
            )
        
        # Botón para generar recomendaciones
        if st.button("🔍 Generar Recomendaciones", type="primary", key="rec_customer"):
            with st.spinner("Analizando clientes similares y generando recomendaciones..."):
                # Obtener estadísticas del cliente
                customer_stats = recommender.get_customer_statistics(customer_id, transactions)
                
                if not customer_stats:
                    st.error(f"❌ Cliente {customer_id} no encontrado en los datos.")
                else:
                    # Mostrar información del cliente
                    st.markdown("### 📊 Perfil del Cliente")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Transacciones", f"{customer_stats['total_transactions']:,}")
                    with col2:
                        st.metric("Productos Únicos", f"{customer_stats['unique_products']:,}")
                    with col3:
                        st.metric("Unidades Totales", f"{customer_stats['total_quantity']:,}")
                    with col4:
                        st.metric("Categorías", f"{customer_stats['unique_categories']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Categoría Favorita:** {customer_stats['favorite_category']}")
                    with col2:
                        st.info(f"**Productos por Transacción:** {customer_stats['avg_products_per_transaction']:.1f}")
                    
                    st.markdown("---")
                    
                    # Generar recomendaciones usando caché
                    recommendations = get_cached_customer_recommendations(
                        customer_id=customer_id,
                        _transactions_df=transactions,
                        top_n=top_n,
                        top_k_similar=1000,
                        min_similarity=min_similarity
                    )
                    
                    if recommendations.empty:
                        st.warning("⚠️ No se encontraron productos para recomendar. Intenta ajustar los parámetros.")
                    else:
                        st.markdown(f"### 🎁 Top {len(recommendations)} Productos Recomendados")
                        st.success(f"✅ Se encontraron {len(recommendations)} recomendaciones basadas en clientes similares")
                        
                        # Mostrar tabla de recomendaciones
                        display_recommendations = recommendations[[
                            'product_id', 'product_name', 'category_name', 
                            'score', 'similar_customers_count'
                        ]].copy()
                        
                        display_recommendations = display_recommendations.rename(columns={
                            'product_id': 'ID Producto',
                            'product_name': 'Nombre',
                            'category_name': 'Categoría',
                            'score': 'Score',
                            'similar_customers_count': 'Clientes que lo compraron'
                        })
                        
                        st.dataframe(
                            display_recommendations.style.format({
                                'Score': '{:.2f}'
                            }).background_gradient(subset=['Score'], cmap='Greens'),
                            width='stretch'
                        )
                        
                        # Gráfico de barras
                        fig = px.bar(
                            recommendations.head(10),
                            x='score',
                            y='product_name',
                            orientation='h',
                            title=f'Top 10 Productos Recomendados para Cliente {customer_id}',
                            labels={'score': 'Score de Recomendación', 'product_name': 'Producto'},
                            color='score',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                        
                        # Botón de descarga
                        csv = recommendations.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Recomendaciones CSV",
                            data=csv,
                            file_name=f"recomendaciones_cliente_{customer_id}.csv",
                            mime="text/csv"
                        )
    
    else:  # Por Producto
        st.subheader("🏷️ Recomendaciones Basadas en Producto")
        st.info(
            "💡 Este sistema utiliza **Market Basket Analysis** con reglas de asociación (Apriori). "
            "Descubre qué productos se compran frecuentemente juntos."
        )
        
        # Inputs de configuración
        col1, col2 = st.columns([2, 2])
        
        with col1:
            product_id = st.number_input(
                "ID del Producto",
                min_value=1,
                value=1,
                help="Ingresa el ID del producto para obtener recomendaciones"
            )
        
        with col2:
            top_n = st.number_input(
                "Top N",
                min_value=1,
                max_value=50,
                value=RecommenderConfig.TOP_N,
                help="Número de productos a recomendar",
                key="top_n_product"
            )
        
        # Parámetros avanzados
        with st.expander("⚙️ Configuración Avanzada de Apriori"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                min_support = st.slider(
                    "Soporte Mínimo (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=RecommenderConfig.MIN_SUPPORT * 100,
                    step=0.1,
                    help="Porcentaje mínimo de transacciones que contienen el itemset"
                ) / 100
            
            with col2:
                min_confidence = st.slider(
                    "Confianza Mínima (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=RecommenderConfig.MIN_CONFIDENCE * 100,
                    step=5.0,
                    help="Probabilidad de que se compre B dado que se compró A"
                ) / 100
            
            with col3:
                min_lift = st.slider(
                    "Lift Mínimo",
                    min_value=1.0,
                    max_value=10.0,
                    value=RecommenderConfig.MIN_LIFT,
                    step=0.1,
                    help="Cuánto más probable es comprar B dado A vs. comprar B solo"
                )
            
            with col4:
                max_transactions_input = st.number_input(
                    "Máx. Transacciones",
                    min_value=10000,
                    max_value=1200000,
                    value=RecommenderConfig.MAX_TRANSACTIONS,
                    step=10000,
                    help="Número máximo de transacciones a analizar. Mayor = más lento pero más preciso"
                )
                # Convertir a None si es muy grande (para procesar todas)
                max_transactions = None if max_transactions_input >= 1000000 else max_transactions_input
            
            st.info(
                "📊 **Métricas explicadas:**\n"
                "- **Support**: Frecuencia de aparición del itemset\n"
                "- **Confidence**: P(B|A) - Probabilidad condicional\n"
                "- **Lift**: Indica qué tan fuerte es la asociación (>1 = asociación positiva)\n"
                "- **Máx. Transacciones**: Usa sample aleatorio para acelerar análisis. Valores recomendados: 50k (rápido), 100k (balance), 200k (preciso)"
            )
        
        # Botón para generar recomendaciones
        if st.button("🔍 Generar Recomendaciones", type="primary", key="rec_product"):
            with st.spinner("Analizando patrones de compra y generando reglas de asociación..."):
                # Obtener estadísticas del producto
                product_stats = recommender.get_product_statistics(product_id, transactions)
                
                if not product_stats:
                    st.error(f"❌ Producto {product_id} no encontrado en los datos.")
                else:
                    # Mostrar información del producto
                    st.markdown("### 📦 Información del Producto")
                    
                    st.markdown(f"**Nombre:** {product_stats['product_name']}")
                    st.markdown(f"**Categoría:** {product_stats['category_name']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Unidades Vendidas", f"{product_stats['total_quantity']:,}")
                    with col2:
                        st.metric("Clientes Únicos", f"{product_stats['unique_customers']:,}")
                    with col3:
                        st.metric("Transacciones", f"{product_stats['unique_transactions']:,}")
                    with col4:
                        st.metric("Cantidad Promedio", f"{product_stats['avg_quantity_per_transaction']:.2f}")
                    
                    st.markdown("---")
                    
                    # Generar recomendaciones usando caché
                    try:
                        # Construir reglas de asociación (cacheadas)
                        association_rules = build_cached_association_rules(
                            _transactions_df=transactions,
                            min_support=min_support,
                            min_confidence=min_confidence,
                            min_lift=min_lift,
                            max_transactions=max_transactions
                        )
                        
                        # Obtener recomendaciones usando las reglas cacheadas
                        recommendations = get_cached_product_recommendations(
                            product_id=product_id,
                            _association_rules=association_rules,
                            _transactions_df=transactions,
                            top_n=top_n
                        )
                        
                        if recommendations.empty:
                            st.warning(
                                "⚠️ No se encontraron productos relacionados con los parámetros actuales. "
                                "Intenta reducir los umbrales en la configuración avanzada."
                            )
                        else:
                            st.markdown(f"### 🛒 Top {len(recommendations)} Productos que se Compran Juntos")
                            st.success(f"✅ Se encontraron {len(recommendations)} productos frecuentemente comprados con este producto")
                            
                            # Mostrar tabla de recomendaciones
                            display_recommendations = recommendations[[
                                'product_id', 'product_name', 'category_name',
                                'score', 'lift', 'confidence', 'support'
                            ]].copy()
                            
                            display_recommendations = display_recommendations.rename(columns={
                                'product_id': 'ID Producto',
                                'product_name': 'Nombre',
                                'category_name': 'Categoría',
                                'score': 'Score',
                                'lift': 'Lift',
                                'confidence': 'Confianza (%)',
                                'support': 'Soporte (%)'
                            })
                            
                            st.dataframe(
                                display_recommendations.style.format({
                                    'Score': '{:.2f}',
                                    'Lift': '{:.2f}',
                                    'Confianza (%)': '{:.2f}',
                                    'Soporte (%)': '{:.2f}'
                                }).background_gradient(subset=['Score'], cmap='Blues'),
                                width='stretch'
                            )
                            
                            # Gráfico de barras
                            fig = px.bar(
                                recommendations.head(10),
                                x='lift',
                                y='product_name',
                                orientation='h',
                                title=f'Top 10 Productos Asociados con {product_stats["product_name"]}',
                                labels={'lift': 'Lift', 'product_name': 'Producto'},
                                color='lift',
                                color_continuous_scale='Blues',
                                hover_data=['confidence', 'support']
                            )
                            fig.update_layout(height=500, showlegend=False)
                            st.plotly_chart(fig, width='stretch')
                            
                            # Botón de descarga
                            csv = recommendations.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Recomendaciones CSV",
                                data=csv,
                                file_name=f"recomendaciones_producto_{product_id}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error al generar recomendaciones: {str(e)}")
                        logger.error(f"Error en recomendaciones: {e}", exc_info=True)


def main():
    """Punto de entrada principal de la aplicación."""
    st.set_page_config(
        page_title="Análisis de Transacciones",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛒 Análisis de Transacciones de Supermercado")
    
    # Verificar si existen datos procesados
    data_exists = (Paths.DATA_PROCESSED / 'transactions_expanded.csv').exists()
    
    # Si no existen datos procesados, generarlos automáticamente
    if not data_exists:
        st.info("🔄 Primera ejecución detectada. Procesando datos iniciales...")
        
        with st.spinner("⏳ Cargando y procesando datos originales... Esto puede tomar unos minutos."):
            try:
                # Crear directorio si no existe
                Paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
                
                # Ejecutar pipeline de procesamiento
                loader = DataLoader()
                processor = DataProcessor()
                
                # Cargar datos raw
                transactions_raw = loader.load_transactions()
                categories = loader.load_categories()
                product_category = loader.load_product_category()
                
                # Procesar todos los datos usando process_all
                processed_data = processor.process_all(transactions_raw, product_category, categories)
                
                # Guardar datos procesados
                processed_data['transactions_expanded'].to_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv', index=False)
                processed_data['customer_metrics'].to_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv', index=False)
                processed_data['product_metrics'].to_csv(Paths.DATA_PROCESSED / 'product_metrics.csv', index=False)
                processed_data['transaction_metrics'].to_csv(Paths.DATA_PROCESSED / 'transaction_metrics.csv', index=False)
                
                st.success("✅ Datos procesados correctamente!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al procesar datos iniciales: {str(e)}")
                logger.error(f"Error en procesamiento inicial: {e}", exc_info=True)
                st.stop()
    
    # Sidebar con navegación
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
    
    # Cargar datos procesados
    with st.spinner("Cargando datos..."):
        transactions, customer_metrics, product_metrics, transaction_metrics = load_processed_data()
    
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
    
    elif page == "Sistema de Recomendación":
        render_recommendations(transactions)
    
    elif page == "Carga de Nuevos Datos":
        render_data_upload()


if __name__ == "__main__":
    main()
