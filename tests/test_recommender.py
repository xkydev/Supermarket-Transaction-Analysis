"""
Script de prueba para el sistema de recomendación.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender import RecommenderSystem
from config import Paths, RecommenderConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Función principal de prueba."""
    logger.info("=" * 80)
    logger.info("PRUEBA DE SISTEMA DE RECOMENDACIÓN")
    logger.info("=" * 80)
    
    try:
        # Cargar datos
        logger.info("\n1. CARGANDO DATOS...")
        transactions = pd.read_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv')
        transactions['date'] = pd.to_datetime(transactions['date'])
        logger.info(f"  Transacciones cargadas: {len(transactions):,}")
        logger.info(f"  Clientes únicos: {transactions['customer_id'].nunique():,}")
        logger.info(f"  Productos únicos: {transactions['product_id'].nunique():,}")
        
        # Inicializar recomendador
        logger.info("\n2. INICIALIZANDO SISTEMA DE RECOMENDACIÓN...")
        recommender = RecommenderSystem()
        
        # === TEST 1: RECOMENDACIONES POR CLIENTE ===
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: RECOMENDACIONES POR CLIENTE (FILTRADO COLABORATIVO)")
        logger.info("=" * 80)
        
        # Seleccionar cliente de prueba (uno con varias compras)
        customer_counts = transactions.groupby('customer_id')['transaction_id'].nunique()
        test_customer = customer_counts[customer_counts >= 5].index[0]
        
        logger.info(f"\nCliente de prueba: {test_customer}")
        
        # Obtener estadísticas del cliente
        customer_stats = recommender.get_customer_statistics(test_customer, transactions)
        logger.info("\nEstadísticas del cliente:")
        for key, value in customer_stats.items():
            logger.info(f"  - {key}: {value}")
        
        # Generar recomendaciones
        logger.info("\nGenerando recomendaciones...")
        customer_recs = recommender.get_customer_recommendations(
            customer_id=test_customer,
            transactions_df=transactions,
            top_n=10,
            min_similarity=0.1
        )
        
        if not customer_recs.empty:
            logger.info(f"\n✓ Se generaron {len(customer_recs)} recomendaciones")
            logger.info("\nTop 5 productos recomendados:")
            for i, row in customer_recs.head(5).iterrows():
                logger.info(
                    f"  {i+1}. {row['product_name']} (ID: {row['product_id']}) - "
                    f"Score: {row['score']:.2f}, "
                    f"Categoría: {row['category_name']}, "
                    f"Compradores similares: {row['similar_customers_count']}"
                )
        else:
            logger.warning("⚠ No se generaron recomendaciones para este cliente")
        
        # === TEST 2: RECOMENDACIONES POR PRODUCTO ===
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: RECOMENDACIONES POR PRODUCTO (MARKET BASKET ANALYSIS)")
        logger.info("=" * 80)
        
        # Seleccionar producto de prueba (uno frecuente)
        product_counts = transactions.groupby('product_id')['transaction_id'].nunique()
        test_product = product_counts.nlargest(20).index[5]  # Tomar el 5to más frecuente
        
        logger.info(f"\nProducto de prueba: {test_product}")
        
        # Obtener estadísticas del producto
        product_stats = recommender.get_product_statistics(test_product, transactions)
        logger.info("\nEstadísticas del producto:")
        for key, value in product_stats.items():
            logger.info(f"  - {key}: {value}")
        
        # Generar recomendaciones
        logger.info("\nGenerando recomendaciones con Market Basket Analysis...")
        logger.info("(Esto puede tardar varios minutos...)")
        
        product_recs = recommender.get_product_recommendations(
            product_id=test_product,
            transactions_df=transactions,
            top_n=10,
            min_support=RecommenderConfig.MIN_SUPPORT,
            min_confidence=RecommenderConfig.MIN_CONFIDENCE,
            min_lift=RecommenderConfig.MIN_LIFT
        )
        
        if not product_recs.empty:
            logger.info(f"\n✓ Se generaron {len(product_recs)} recomendaciones")
            logger.info("\nTop 5 productos que se compran juntos:")
            for i, row in product_recs.head(5).iterrows():
                logger.info(
                    f"  {i+1}. {row['product_name']} (ID: {row['product_id']}) - "
                    f"Lift: {row['lift']:.2f}, "
                    f"Confianza: {row['confidence']:.2f}%, "
                    f"Soporte: {row['support']:.2f}%, "
                    f"Categoría: {row['category_name']}"
                )
        else:
            logger.warning("⚠ No se generaron recomendaciones para este producto")
            logger.info("  Intenta reducir los umbrales (min_support, min_confidence, min_lift)")
        
        # === TEST 3: GUARDAR RESULTADOS ===
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: GUARDANDO RESULTADOS")
        logger.info("=" * 80)
        
        output_path = Paths.DATA_PROCESSED
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not customer_recs.empty:
            customer_recs.to_csv(
                output_path / f"recommendations_customer_{test_customer}.csv",
                index=False
            )
            logger.info(f"\n✓ Recomendaciones por cliente guardadas en {output_path}")
        
        if not product_recs.empty:
            product_recs.to_csv(
                output_path / f"recommendations_product_{test_product}.csv",
                index=False
            )
            logger.info(f"✓ Recomendaciones por producto guardadas en {output_path}")
        
        # === TEST 4: ESTADÍSTICAS DE REGLAS DE ASOCIACIÓN ===
        if recommender.association_rules is not None and len(recommender.association_rules) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("TEST 4: ESTADÍSTICAS DE REGLAS DE ASOCIACIÓN")
            logger.info("=" * 80)
            
            rules = recommender.association_rules
            logger.info(f"\nTotal de reglas generadas: {len(rules):,}")
            logger.info(f"Lift promedio: {rules['lift'].mean():.2f}")
            logger.info(f"Confianza promedio: {rules['confidence'].mean()*100:.2f}%")
            logger.info(f"Soporte promedio: {rules['support'].mean()*100:.2f}%")
            
            logger.info("\nTop 5 reglas con mayor lift:")
            top_rules = rules.nlargest(5, 'lift')
            for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
                logger.info(
                    f"  {i}. {rule['antecedents_list']} => {rule['consequents_list']} - "
                    f"Lift: {rule['lift']:.2f}, "
                    f"Confidence: {rule['confidence']*100:.2f}%, "
                    f"Support: {rule['support']*100:.2f}%"
                )
        
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
