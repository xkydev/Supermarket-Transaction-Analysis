"""
Script de prueba para segmentación de clientes.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import CustomerSegmentation
from config import Paths, ClusteringConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Función principal de prueba."""
    logger.info("=" * 80)
    logger.info("PRUEBA DE SEGMENTACIÓN DE CLIENTES")
    logger.info("=" * 80)
    
    try:
        # Cargar datos de clientes
        logger.info("\n1. CARGANDO DATOS DE CLIENTES...")
        customer_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv')
        logger.info(f"  Clientes cargados: {len(customer_metrics):,}")
        
        # Inicializar segmentador
        logger.info("\n2. PREPARANDO FEATURES...")
        segmenter = CustomerSegmentation()
        df, X_scaled = segmenter.prepare_features(customer_metrics)
        logger.info(f"  Features: {segmenter.features_used}")
        logger.info(f"  Shape: {X_scaled.shape}")
        
        # Encontrar K óptimo
        logger.info("\n3. ENCONTRANDO K ÓPTIMO...")
        optimization = segmenter.find_optimal_k(X_scaled, min_k=ClusteringConfig.MIN_CLUSTERS, max_k=ClusteringConfig.MAX_CLUSTERS)
        logger.info(f"  K óptimo recomendado: {optimization['recommended_k']}")
        logger.info(f"  Silhouette scores: {[f'{s:.3f}' for s in optimization['silhouette_scores']]}")
        
        # Entrenar con K óptimo
        k = optimization['recommended_k']
        logger.info(f"\n4. ENTRENANDO MODELO CON K={k}...")
        labels = segmenter.fit_kmeans(X_scaled, n_clusters=k)
        
        # Perfilar clusters
        logger.info("\n5. PERFILANDO CLUSTERS...")
        profile = segmenter.profile_clusters(customer_metrics, labels)
        cluster_names = segmenter.name_clusters(profile)
        
        logger.info("\n📊 PERFIL DE CLUSTERS:")
        for _, row in profile.iterrows():
            cluster_id = int(row['cluster'])
            info = cluster_names[cluster_id]
            logger.info(f"\n  Cluster {cluster_id}: {info['name']}")
            logger.info(f"    - Tamaño: {int(row['size']):,} clientes ({row['percentage']:.1f}%)")
            logger.info(f"    - Frecuencia promedio: {row['avg_frequency']:.1f}")
            logger.info(f"    - Volumen total: {int(row['total_volume']):,}")
            logger.info(f"    - Productos promedio: {row['avg_products']:.1f}")
            logger.info(f"    - Descripción: {info['description']}")
            logger.info(f"    - Estrategia: {info['strategy']}")
        
        # Reducir dimensionalidad y visualizar
        logger.info("\n6. GENERANDO VISUALIZACIONES...")
        X_reduced_2d = segmenter.reduce_dimensions(X_scaled, n_components=2)
        X_reduced_3d = segmenter.reduce_dimensions(X_scaled, n_components=3)
        
        fig_elbow = segmenter.plot_elbow_method(optimization)
        fig_2d = segmenter.plot_clusters_2d(X_reduced_2d, labels, cluster_names)
        fig_3d = segmenter.plot_clusters_3d(X_reduced_3d, labels, cluster_names)
        
        # Guardar visualizaciones
        output_path = Paths.DATA_PROCESSED / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig_elbow.write_html(output_path / "clustering_elbow.html")
        fig_2d.write_html(output_path / "clustering_2d.html")
        fig_3d.write_html(output_path / "clustering_3d.html")
        
        logger.info(f"  Visualizaciones guardadas en: {output_path}")
        
        # Guardar resultados
        logger.info("\n7. GUARDANDO RESULTADOS...")
        customer_segments = customer_metrics.copy()
        customer_segments['cluster'] = labels
        customer_segments['cluster_name'] = customer_segments['cluster'].map(
            lambda x: cluster_names[x]['name']
        )
        
        customer_segments.to_csv(
            Paths.DATA_PROCESSED / 'customer_segments.csv',
            index=False
        )
        profile.to_csv(
            'data/processed/cluster_profiles.csv',
            index=False
        )
        
        logger.info("  Resultados guardados:")
        logger.info("    - customer_segments.csv")
        logger.info("    - cluster_profiles.csv")
        
        logger.info("\n" + "=" * 80)
        logger.info("PRUEBA COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
