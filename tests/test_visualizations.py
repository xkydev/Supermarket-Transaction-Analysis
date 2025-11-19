"""
Script de prueba para métricas y visualizaciones.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import MetricsCalculator
from src.visualizations import Visualizer
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
    logger.info("PRUEBA DE MÉTRICAS Y VISUALIZACIONES")
    logger.info("=" * 80)
    
    try:
        # Cargar datos procesados
        logger.info("\n1. CARGANDO DATOS PROCESADOS...")
        transactions = pd.read_csv(Paths.DATA_PROCESSED / 'transactions_expanded.csv')
        customer_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'customer_metrics.csv')
        product_metrics = pd.read_csv(Paths.DATA_PROCESSED / 'product_metrics.csv')
        
        # Convertir fecha a datetime
        transactions['date'] = pd.to_datetime(transactions['date'])
        
        logger.info(f"  - Transacciones: {len(transactions):,}")
        logger.info(f"  - Clientes: {len(customer_metrics):,}")
        logger.info(f"  - Productos: {len(product_metrics):,}")
        
        # Calcular métricas
        logger.info("\n2. CALCULANDO MÉTRICAS...")
        calc = MetricsCalculator()
        
        kpis = calc.calculate_kpis(transactions)
        logger.info("\n  KPIs Principales:")
        logger.info(f"    - Total unidades: {kpis['total_units']:,}")
        logger.info(f"    - Total transacciones: {kpis['total_transactions']:,}")
        logger.info(f"    - Total clientes: {kpis['total_customers']:,}")
        logger.info(f"    - Total productos: {kpis['total_products']:,}")
        logger.info(f"    - Tamaño promedio canasta: {kpis['avg_basket_size']:.2f}")
        
        category_perf = calc.calculate_category_performance(transactions)
        logger.info(f"\n  Top 5 Categorías:")
        for _, row in category_perf.head(5).iterrows():
            logger.info(f"    - {row['category_name']}: {row['total_units']:,} unidades ({row['percentage']:.1f}%)")
        
        temporal = calc.calculate_temporal_patterns(transactions)
        logger.info(f"\n  Patrones Temporales:")
        logger.info(f"    - Días de semana analizados: {len(temporal['by_day_of_week'])}")
        logger.info(f"    - Meses analizados: {len(temporal['by_month'])}")
        
        # Crear visualizaciones
        logger.info("\n3. GENERANDO VISUALIZACIONES...")
        viz = Visualizer()
        
        # Top productos
        fig1 = viz.plot_top_products(product_metrics, n=10)
        logger.info("  ✓ Gráfico de top productos creado")
        
        # Top clientes
        fig2 = viz.plot_top_customers(customer_metrics, n=10)
        logger.info("  ✓ Gráfico de top clientes creado")
        
        # Categorías
        fig3 = viz.plot_category_distribution(category_perf, n=10, chart_type='bar')
        logger.info("  ✓ Gráfico de categorías creado")
        
        # Serie temporal
        fig4 = viz.plot_time_series(transactions, freq='D', metric='quantity')
        logger.info("  ✓ Serie temporal creada")
        
        # Heatmap día de semana
        fig5 = viz.plot_heatmap_day_hour(transactions)
        logger.info("  ✓ Heatmap de días creado")
        
        # Correlación
        fig6 = viz.plot_correlation_heatmap(customer_metrics)
        logger.info("  ✓ Matriz de correlación creada")
        
        # Boxplot
        fig7 = viz.plot_boxplot(
            customer_metrics,
            column='frequency',
            title='Distribución de Frecuencia de Compra'
        )
        logger.info("  ✓ Boxplot creado")
        
        # Guardar visualizaciones como HTML
        logger.info("\n4. GUARDANDO VISUALIZACIONES...")
        output_path = Paths.DATA_PROCESSED / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig1.write_html(output_path / "top_products.html")
        fig2.write_html(output_path / "top_customers.html")
        fig3.write_html(output_path / "categories.html")
        fig4.write_html(output_path / "time_series.html")
        fig5.write_html(output_path / "heatmap_days.html")
        fig6.write_html(output_path / "correlation.html")
        fig7.write_html(output_path / "boxplot_frequency.html")
        
        logger.info(f"  Visualizaciones guardadas en: {output_path}")
        
        # Análisis de canasta
        logger.info("\n5. ANÁLISIS DE CANASTA...")
        basket = calc.calculate_basket_analysis(transactions)
        logger.info(f"  - Productos promedio por canasta: {basket['avg_products_per_basket']:.2f}")
        logger.info(f"  - Cantidad promedio por canasta: {basket['avg_quantity_per_basket']:.2f}")
        logger.info(f"  - Categorías promedio por canasta: {basket['avg_categories_per_basket']:.2f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PRUEBA COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"\nPuedes abrir las visualizaciones en: {output_path.absolute()}")
        
        return True
        
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
