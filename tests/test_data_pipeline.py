"""
Script de prueba para validar carga y procesamiento de datos.
"""

import logging
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from config import Paths

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Función principal de prueba."""
    logger.info("=" * 80)
    logger.info("INICIANDO PRUEBA DE CARGA Y PROCESAMIENTO DE DATOS")
    logger.info("=" * 80)
    
    try:
        # 1. Cargar datos
        logger.info("\n1. CARGANDO DATOS...")
        loader = DataLoader()
        transactions, product_category, categories = loader.load_all_data()
        
        logger.info("\nEstadísticas de carga:")
        logger.info(f"  - Transacciones: {len(transactions):,}")
        logger.info(f"  - Productos-Categorías: {len(product_category):,}")
        logger.info(f"  - Categorías: {len(categories):,}")
        logger.info(f"  - Tiendas: {transactions['store_id'].nunique()}")
        logger.info(f"  - Clientes: {transactions['customer_id'].nunique():,}")
        logger.info(f"  - Rango de fechas: {transactions['date'].min()} a {transactions['date'].max()}")
        
        # 2. Procesar datos
        logger.info("\n2. PROCESANDO DATOS...")
        processor = DataProcessor()
        processed_data = processor.process_all(transactions, product_category, categories)
        
        # 3. Mostrar resumen
        logger.info("\n3. RESUMEN DE DATOS PROCESADOS:")
        logger.info(f"  - Registros expandidos: {len(processed_data['transactions_expanded']):,}")
        logger.info(f"  - Transacciones únicas: {len(processed_data['transaction_metrics']):,}")
        logger.info(f"  - Clientes analizados: {len(processed_data['customer_metrics']):,}")
        logger.info(f"  - Productos analizados: {len(processed_data['product_metrics']):,}")
        
        # 4. Muestras de datos
        logger.info("\n4. MUESTRA DE DATOS PROCESADOS:")
        
        logger.info("\nTop 5 productos más vendidos:")
        top_products = processed_data['product_metrics'].nlargest(5, 'total_quantity')
        for _, row in top_products.iterrows():
            logger.info(
                f"  - Producto {row['product_id']}: "
                f"{row['total_quantity']:,} unidades, "
                f"{row['unique_customers']:,} clientes, "
                f"categoría: {row['category_name']}"
            )
        
        logger.info("\nTop 5 clientes más frecuentes:")
        top_customers = processed_data['customer_metrics'].nlargest(5, 'frequency')
        for _, row in top_customers.iterrows():
            logger.info(
                f"  - Cliente {row['customer_id']}: "
                f"{row['frequency']} transacciones, "
                f"{row['total_quantity']:,} unidades, "
                f"{row['unique_products']} productos únicos, "
                f"recency: {row['recency']} días"
            )
        
        logger.info("\nTop 5 categorías por volumen:")
        category_volume = processed_data['transactions_expanded'].groupby('category_name')['quantity'].sum().sort_values(ascending=False).head(5)
        for cat, vol in category_volume.items():
            logger.info(f"  - {cat}: {vol:,} unidades")
        
        # 5. Guardar datos procesados
        logger.info("\n5. GUARDANDO DATOS PROCESADOS...")
        output_path = Paths.DATA_PROCESSED
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_data['transactions_expanded'].to_csv(
            output_path / "transactions_expanded.csv", index=False
        )
        processed_data['customer_metrics'].to_csv(
            output_path / "customer_metrics.csv", index=False
        )
        processed_data['product_metrics'].to_csv(
            output_path / "product_metrics.csv", index=False
        )
        processed_data['transaction_metrics'].to_csv(
            output_path / "transaction_metrics.csv", index=False
        )
        
        logger.info(f"Datos guardados en: {output_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PRUEBA COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
