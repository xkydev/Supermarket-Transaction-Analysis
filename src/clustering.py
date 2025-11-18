"""
Módulo para segmentación de clientes mediante clustering.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import ClusteringConfig

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """Clase para segmentación de clientes usando K-Means."""
    
    def __init__(self, random_state: int = None):
        """
        Inicializa el segmentador.
        
        Args:
            random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state or ClusteringConfig.RANDOM_STATE
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.features_used = None
        self.scaled_features = None
        
    def prepare_features(
        self,
        customer_metrics: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepara las features para clustering.
        
        Args:
            customer_metrics: DataFrame con métricas de clientes
            features: Lista de features a usar. Si None, usa las sugeridas.
            
        Returns:
            Tupla de (DataFrame original, features escaladas)
        """
        if features is None:
            # Features sugeridas: frecuencia, productos distintos, volumen, diversidad
            features = [
                'frequency',
                'unique_products',
                'total_quantity',
                'unique_categories'
            ]
        
        # Verificar que las features existen
        available_features = [f for f in features if f in customer_metrics.columns]
        
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"Features no disponibles: {missing}")
        
        self.features_used = available_features
        
        # Extraer features y manejar valores faltantes
        X = customer_metrics[available_features].fillna(0)
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        self.scaled_features = X_scaled
        
        logger.info(f"Features preparadas: {available_features}")
        logger.info(f"Shape de datos escalados: {X_scaled.shape}")
        
        return customer_metrics, X_scaled
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        min_k: int = None,
        max_k: int = None
    ) -> Dict[str, any]:
        """
        Encuentra el número óptimo de clusters usando el método del codo.
        
        Args:
            X: Features escaladas
            min_k: Número mínimo de clusters
            max_k: Número máximo de clusters
            
        Returns:
            Diccionario con métricas por k
        """
        min_k = min_k or ClusteringConfig.MIN_CLUSTERS
        max_k = max_k or ClusteringConfig.MAX_CLUSTERS
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        k_values = range(min_k, max_k + 1)
        
        logger.info(f"Evaluando K desde {min_k} hasta {max_k}...")
        
        for k in k_values:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=ClusteringConfig.N_INIT
            )
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            calinski_scores.append(calinski_harabasz_score(X, labels))
            davies_bouldin_scores.append(davies_bouldin_score(X, labels))
            
            logger.info(
                f"K={k}: Inertia={kmeans.inertia_:.2f}, "
                f"Silhouette={silhouette_scores[-1]:.3f}"
            )
        
        results = {
            'k_values': list(k_values),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
        
        # Recomendar K basado en silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        results['recommended_k'] = k_values[best_k_idx]
        
        logger.info(f"K óptimo recomendado: {results['recommended_k']}")
        
        return results
    
    def fit_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """
        Entrena el modelo K-Means.
        
        Args:
            X: Features escaladas
            n_clusters: Número de clusters
            
        Returns:
            Array de labels de cluster
        """
        logger.info(f"Entrenando K-Means con {n_clusters} clusters...")
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=ClusteringConfig.N_INIT
        )
        
        labels = self.kmeans.fit_predict(X)
        
        # Calcular métricas finales
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        
        logger.info(f"Modelo entrenado exitosamente")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski:.2f}")
        logger.info(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        return labels
    
    def reduce_dimensions(
        self,
        X: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce dimensionalidad usando PCA para visualización.
        
        Args:
            X: Features escaladas
            n_components: Número de componentes
            
        Returns:
            Features reducidas
        """
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_reduced = self.pca.fit_transform(X)
        
        explained_var = sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"PCA aplicado: {n_components} componentes explican "
            f"{explained_var*100:.2f}% de la varianza"
        )
        
        return X_reduced
    
    def profile_clusters(
        self,
        customer_metrics: pd.DataFrame,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Crea perfil de cada cluster.
        
        Args:
            customer_metrics: DataFrame con métricas de clientes
            labels: Array de labels de cluster
            
        Returns:
            DataFrame con perfil de cada cluster
        """
        df = customer_metrics.copy()
        df['cluster'] = labels
        
        # Calcular estadísticas por cluster
        profile = df.groupby('cluster').agg({
            'customer_id': 'count',
            'frequency': ['mean', 'median'],
            'total_quantity': ['mean', 'median', 'sum'],
            'unique_products': ['mean', 'median'],
            'unique_categories': ['mean', 'median'],
            'recency': ['mean', 'median']
        }).round(2)
        
        # Aplanar índice
        profile.columns = ['_'.join(col).strip() for col in profile.columns.values]
        profile = profile.reset_index()
        
        # Renombrar columnas para claridad
        profile = profile.rename(columns={
            'customer_id_count': 'size',
            'frequency_mean': 'avg_frequency',
            'frequency_median': 'median_frequency',
            'total_quantity_mean': 'avg_quantity',
            'total_quantity_median': 'median_quantity',
            'total_quantity_sum': 'total_volume',
            'unique_products_mean': 'avg_products',
            'unique_products_median': 'median_products',
            'unique_categories_mean': 'avg_categories',
            'unique_categories_median': 'median_categories',
            'recency_mean': 'avg_recency',
            'recency_median': 'median_recency'
        })
        
        # Calcular porcentaje
        total_customers = profile['size'].sum()
        profile['percentage'] = (profile['size'] / total_customers * 100).round(2)
        
        return profile
    
    def name_clusters(self, profile: pd.DataFrame) -> Dict[int, Dict[str, str]]:
        """
        Asigna nombres descriptivos a los clusters basado en sus características.
        
        Args:
            profile: DataFrame con perfil de clusters
            
        Returns:
            Diccionario con nombre y descripción por cluster
        """
        cluster_names = {}
        
        for _, row in profile.iterrows():
            cluster_id = int(row['cluster'])
            freq = row['avg_frequency']
            volume = row['total_volume']
            products = row['avg_products']
            
            # Determinar nombre basado en características
            if freq < 3 and volume < profile['total_volume'].quantile(0.25):
                name = "Ocasionales de Bajo Valor"
                desc = "Clientes con baja frecuencia y bajo volumen de compra"
                strategy = "Activar con promociones y descuentos para incentivar recompra"
            
            elif freq >= 3 and freq < 10 and volume < profile['total_volume'].median():
                name = "Regulares Moderados"
                desc = "Clientes con frecuencia media y volumen moderado"
                strategy = "Aumentar ticket promedio con cross-selling y bundling"
            
            elif freq >= 10 and products > profile['avg_products'].quantile(0.75):
                name = "Leales Premium"
                desc = "Clientes frecuentes con alta diversidad de productos"
                strategy = "Retener con programa de fidelización y ofertas exclusivas"
            
            elif volume > profile['total_volume'].quantile(0.75):
                name = "Alto Volumen"
                desc = "Clientes con compras de gran volumen"
                strategy = "Ofrecer descuentos por volumen y servicios premium"
            
            elif freq >= 5 and products < profile['avg_products'].quantile(0.25):
                name = "Frecuentes Focalizados"
                desc = "Clientes frecuentes pero con poca diversidad"
                strategy = "Expandir canasta con recomendaciones personalizadas"
            
            else:
                name = f"Segmento {cluster_id}"
                desc = "Segmento con características mixtas"
                strategy = "Analizar comportamiento específico para estrategia personalizada"
            
            cluster_names[cluster_id] = {
                'name': name,
                'description': desc,
                'strategy': strategy
            }
        
        return cluster_names
    
    def plot_elbow_method(self, optimization_results: Dict) -> go.Figure:
        """
        Crea gráfico del método del codo.
        
        Args:
            optimization_results: Resultados de find_optimal_k
            
        Returns:
            Figura de Plotly
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Método del Codo (Inertia)',
                'Silhouette Score',
                'Calinski-Harabasz Score',
                'Davies-Bouldin Score'
            )
        )
        
        k_values = optimization_results['k_values']
        
        # Inertia (menor es mejor)
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=optimization_results['inertias'],
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Silhouette (mayor es mejor)
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=optimization_results['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Calinski-Harabasz (mayor es mejor)
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=optimization_results['calinski_scores'],
                mode='lines+markers',
                name='Calinski-Harabasz',
                line=dict(color='orange', width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Davies-Bouldin (menor es mejor)
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=optimization_results['davies_bouldin_scores'],
                mode='lines+markers',
                name='Davies-Bouldin',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        # Marcar K óptimo
        optimal_k = optimization_results['recommended_k']
        for i in range(1, 5):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.add_vline(
                x=optimal_k,
                line_dash="dash",
                line_color="gray",
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Número de Clusters (K)", row=2, col=1)
        fig.update_xaxes(title_text="Número de Clusters (K)", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text=f"Optimización de K (K óptimo recomendado: {optimal_k})"
        )
        
        return fig
    
    def plot_clusters_2d(
        self,
        X_reduced: np.ndarray,
        labels: np.ndarray,
        cluster_names: Dict[int, Dict[str, str]],
        title: str = "Visualización de Clusters (PCA 2D)"
    ) -> go.Figure:
        """
        Crea visualización 2D de clusters.
        
        Args:
            X_reduced: Features reducidas con PCA
            labels: Labels de cluster
            cluster_names: Nombres de clusters
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        df_plot = pd.DataFrame({
            'PC1': X_reduced[:, 0],
            'PC2': X_reduced[:, 1],
            'Cluster': labels,
            'Cluster_Name': [cluster_names[l]['name'] for l in labels]
        })
        
        fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster_Name',
            title=title,
            labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'},
            hover_data={'Cluster': True}
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        fig.update_layout(height=600)
        
        return fig
    
    def plot_clusters_3d(
        self,
        X_reduced: np.ndarray,
        labels: np.ndarray,
        cluster_names: Dict[int, Dict[str, str]],
        title: str = "Visualización de Clusters (PCA 3D)"
    ) -> go.Figure:
        """
        Crea visualización 3D de clusters.
        
        Args:
            X_reduced: Features reducidas con PCA (3 componentes)
            labels: Labels de cluster
            cluster_names: Nombres de clusters
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        df_plot = pd.DataFrame({
            'PC1': X_reduced[:, 0],
            'PC2': X_reduced[:, 1],
            'PC3': X_reduced[:, 2],
            'Cluster': labels,
            'Cluster_Name': [cluster_names[l]['name'] for l in labels]
        })
        
        fig = px.scatter_3d(
            df_plot,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster_Name',
            title=title,
            labels={
                'PC1': 'PC1',
                'PC2': 'PC2',
                'PC3': 'PC3'
            },
            hover_data={'Cluster': True}
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.update_layout(height=700)
        
        return fig
