"""
Script de prueba para la funcionalidad de Carga de Nuevos Datos.
Genera un archivo CSV de prueba válido y verifica el procesamiento.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.metrics import MetricsCalculator
from config import Paths

def generate_test_transaction_file(output_path: Path, n_transactions: int = 100, with_header: bool = True):
    """
    Genera un archivo de transacciones de prueba.
    
    Args:
        output_path: Ruta donde guardar el archivo
        n_transactions: Número de transacciones a generar
        with_header: Si False, genera archivo sin nombres de columnas
    """
    print(f"\n{'='*80}")
    print(f"GENERANDO ARCHIVO DE PRUEBA ({'con' if with_header else 'sin'} encabezados)")
    print(f"{'='*80}\n")
    
    np.random.seed(42)
    
    # Generar datos aleatorios
    dates = [
        (datetime(2013, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
        for _ in range(n_transactions)
    ]
    
    store_ids = np.random.choice([102, 103, 107, 110], n_transactions)
    customer_ids = np.random.randint(200000, 210000, n_transactions)
    
    # Generar listas de productos (IDs entre 1 y 50)
    products = []
    for _ in range(n_transactions):
        n_products = np.random.randint(1, 10)
        product_list = np.random.randint(1, 51, n_products)
        products.append(' '.join(map(str, product_list)))
    
    # Crear DataFrame
    test_df = pd.DataFrame({
        'date': dates,
        'store_id': store_ids,
        'customer_id': customer_ids,
        'products': products
    })
    
    # Guardar archivo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False, header=with_header)
    
    print(f"✅ Archivo generado: {output_path}")
    print(f"   Con encabezados: {'Sí' if with_header else 'No'}")
    print(f"   Transacciones: {len(test_df):,}")
    print(f"   Clientes únicos: {test_df['customer_id'].nunique():,}")
    print(f"   Tiendas: {test_df['store_id'].nunique()}")
    
    # Contar productos únicos
    all_products = set()
    for products_str in test_df['products']:
        all_products.update(products_str.split())
    print(f"   Productos únicos: {len(all_products):,}")
    
    return test_df


def test_data_validation(file_path: Path):
    """
    Prueba las validaciones del archivo.
    
    Args:
        file_path: Ruta al archivo a validar
    """
    print(f"\n{'='*80}")
    print("VALIDANDO ESTRUCTURA DEL ARCHIVO")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(file_path)
    
    # Validar columnas requeridas
    required_columns = ['date', 'store_id', 'customer_id', 'products']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Columnas faltantes: {', '.join(missing_columns)}")
        return False
    else:
        print("✅ Todas las columnas requeridas están presentes")
    
    # Validar fechas
    try:
        pd.to_datetime(df['date'])
        print("✅ Formato de fechas válido")
    except:
        print("❌ Error en formato de fechas")
        return False
    
    # Validar valores nulos
    null_counts = df[required_columns].isnull().sum()
    has_nulls = null_counts.sum() > 0
    
    if not has_nulls:
        print("✅ Sin valores nulos")
    else:
        print(f"❌ Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
        return False
    
    # Validar IDs positivos
    positive_ids = (df['customer_id'] > 0).all() and (df['store_id'] > 0).all()
    
    if positive_ids:
        print("✅ Todos los IDs son positivos")
    else:
        print("❌ Algunos IDs son negativos o cero")
        return False
    
    print("\n✅ TODAS LAS VALIDACIONES PASADAS")
    return True


def test_data_processing(file_path: Path):
    """
    Prueba el procesamiento del archivo.
    
    Args:
        file_path: Ruta al archivo a procesar
    """
    print(f"\n{'='*80}")
    print("PROCESANDO ARCHIVO")
    print(f"{'='*80}\n")
    
    processor = DataProcessor()
    
    # Leer archivo
    print("Leyendo archivo...")
    raw_df = pd.read_csv(file_path)
    
    # Expandir transacciones
    print("Expandiendo transacciones...")
    expanded_df = processor.expand_transactions(raw_df)
    
    # Agregar features temporales
    print("Agregando features temporales...")
    expanded_df = processor.add_temporal_features(expanded_df)
    
    print(f"\n✅ Procesamiento completado:")
    print(f"   Filas originales: {len(raw_df):,}")
    print(f"   Filas expandidas: {len(expanded_df):,}")
    print(f"   Clientes: {expanded_df['customer_id'].nunique():,}")
    print(f"   Productos: {expanded_df['product_id'].nunique():,}")
    print(f"   Cantidad total: {expanded_df['quantity'].sum():,}")
    
    # Mostrar muestra
    print("\n📋 Muestra de datos procesados:")
    print(expanded_df.head(10).to_string())
    
    return expanded_df


def test_metrics_calculation(transactions_df: pd.DataFrame):
    """
    Prueba el cálculo de métricas básicas.
    
    Args:
        transactions_df: DataFrame de transacciones expandidas
    """
    print(f"\n{'='*80}")
    print("CALCULANDO MÉTRICAS BÁSICAS")
    print(f"{'='*80}\n")
    
    # Calcular métricas básicas sin usar el processor
    print("Calculando estadísticas básicas...")
    
    # Métricas por cliente
    customer_stats = transactions_df.groupby('customer_id').agg({
        'transaction_id': 'nunique',
        'product_id': 'nunique',
        'quantity': 'sum'
    }).reset_index()
    customer_stats.columns = ['customer_id', 'transactions', 'unique_products', 'total_quantity']
    print(f"✅ {len(customer_stats):,} clientes procesados")
    
    # Métricas por producto
    product_stats = transactions_df.groupby('product_id').agg({
        'customer_id': 'nunique',
        'quantity': ['sum', 'mean']
    }).reset_index()
    product_stats.columns = ['product_id', 'unique_customers', 'total_quantity', 'avg_quantity']
    print(f"✅ {len(product_stats):,} productos procesados")
    
    # Métricas por día
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    daily_stats = transactions_df.groupby('date').agg({
        'transaction_id': 'nunique',
        'quantity': 'sum'
    }).reset_index()
    daily_stats.columns = ['date', 'transactions', 'total_quantity']
    print(f"✅ {len(daily_stats):,} días con transacciones")
    
    print("\n📊 Resumen de métricas de clientes:")
    print(customer_stats.describe().to_string())
    
    return customer_stats, product_stats, daily_stats


def test_append_mode():
    """
    Prueba el modo de agregar datos a existentes.
    """
    print(f"\n{'='*80}")
    print("PROBANDO MODO 'AGREGAR A DATOS EXISTENTES'")
    print(f"{'='*80}\n")
    
    # Cargar datos existentes
    existing_path = Paths.DATA_PROCESSED / 'transactions_expanded.csv'
    
    if not existing_path.exists():
        print("⚠️ No hay datos existentes. Saltando prueba de agregación.")
        return
    
    existing_df = pd.read_csv(existing_path)
    print(f"Datos existentes: {len(existing_df):,} filas")
    
    # Generar archivo de prueba pequeño
    test_file = Paths.DATA_RAW / 'test_append_999_Tran.csv'
    test_df = generate_test_transaction_file(test_file, n_transactions=50)
    
    # Procesar nuevo archivo
    processor = DataProcessor()
    test_raw = pd.read_csv(test_file)
    new_expanded = processor.expand_transactions(test_raw)
    new_expanded = processor.add_temporal_features(new_expanded)
    print(f"Nuevos datos: {len(new_expanded):,} filas")
    
    # Combinar
    combined_df = pd.concat([existing_df, new_expanded], ignore_index=True)
    print(f"Datos combinados: {len(combined_df):,} filas")
    
    # Verificar
    expected = len(existing_df) + len(new_expanded)
    if len(combined_df) == expected:
        print("✅ Agregación exitosa")
    else:
        print(f"❌ Error en agregación. Esperado: {expected}, Obtenido: {len(combined_df)}")
    
    # Limpiar archivo de prueba
    test_file.unlink()
    print("\n🗑️ Archivo de prueba eliminado")


def test_file_without_headers():
    """
    Prueba el procesamiento de archivos sin encabezados.
    """
    print(f"\n{'='*80}")
    print("PROBANDO ARCHIVO SIN ENCABEZADOS")
    print(f"{'='*80}\n")
    
    # Generar archivo sin encabezados
    test_file = Paths.DATA_RAW / 'test_no_header_999_Tran.csv'
    generate_test_transaction_file(test_file, n_transactions=100, with_header=False)
    
    # Leer archivo
    df_no_header = pd.read_csv(test_file, header=None)
    print(f"\n📋 Columnas detectadas sin header: {df_no_header.columns.tolist()}")
    
    # Simular la lógica de detección y asignación de nombres
    if df_no_header.columns.tolist() == [0, 1, 2, 3]:
        print("✅ Detectado archivo sin encabezados")
        df_no_header.columns = ['date', 'store_id', 'customer_id', 'products']
        print(f"✅ Nombres asignados: {df_no_header.columns.tolist()}")
    
    # Validar que ahora tiene las columnas correctas
    required_columns = ['date', 'store_id', 'customer_id', 'products']
    if all(col in df_no_header.columns for col in required_columns):
        print("✅ Todas las columnas requeridas presentes después de asignación")
    else:
        print("❌ Error en asignación de columnas")
    
    # Limpiar
    test_file.unlink()
    print("\n🗑️ Archivo de prueba eliminado")


def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "="*80)
    print("TEST: FUNCIONALIDAD DE CARGA DE NUEVOS DATOS")
    print("="*80)
    
    # Generar archivo de prueba CON encabezados
    test_file = Paths.DATA_RAW / 'test_999_Tran.csv'
    generate_test_transaction_file(test_file, n_transactions=200, with_header=True)
    
    # Validar archivo
    if not test_data_validation(test_file):
        print("\n❌ Validación fallida. Abortando pruebas.")
        return
    
    # Procesar archivo
    expanded_df = test_data_processing(test_file)
    
    # Calcular métricas
    customer_metrics, product_metrics, transaction_metrics = test_metrics_calculation(expanded_df)
    
    # Probar modo agregar
    test_append_mode()
    
    # Probar archivo sin encabezados
    test_file_without_headers()
    
    # Guardar resultados de prueba (opcional)
    print(f"\n{'='*80}")
    print("GUARDANDO RESULTADOS DE PRUEBA")
    print(f"{'='*80}\n")
    
    test_output_dir = Paths.DATA_PROCESSED / 'test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    expanded_df.to_csv(test_output_dir / 'test_transactions_expanded.csv', index=False)
    customer_metrics.to_csv(test_output_dir / 'test_customer_metrics.csv', index=False)
    product_metrics.to_csv(test_output_dir / 'test_product_metrics.csv', index=False)
    transaction_metrics.to_csv(test_output_dir / 'test_transaction_metrics.csv', index=False)
    
    print(f"✅ Resultados guardados en: {test_output_dir}")
    
    # Limpiar archivo de prueba
    print("\n🗑️ Limpiando archivos temporales...")
    test_file.unlink()
    
    print("\n" + "="*80)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("="*80)
    
    print("\n📝 INSTRUCCIONES:")
    print("1. Abre la aplicación Streamlit")
    print("2. Ve a la página 'Carga de Nuevos Datos'")
    print("3. Carga el archivo generado en: data/processed/test/")
    print("4. Verifica que la validación y procesamiento funcionen correctamente")


if __name__ == "__main__":
    main()
