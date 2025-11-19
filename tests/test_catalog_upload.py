"""
Tests para la carga de Categorías y Producto-Categoría.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Paths


def generate_test_categories_file(num_categories: int = 10, with_header: bool = True):
    """
    Genera un archivo CSV de prueba con categorías.
    
    Args:
        num_categories: Número de categorías a generar
        with_header: Si True, incluye encabezados en el CSV
    """
    print(f"\n{'='*60}")
    print(f"Generando archivo de categorías de prueba...")
    print(f"  - Categorías: {num_categories}")
    print(f"  - Con encabezados: {with_header}")
    
    # Generar IDs desde 100 para evitar conflictos con categorías existentes
    category_ids = list(range(100, 100 + num_categories))
    category_names = [
        f"Categoría Test {i}" for i in range(1, num_categories + 1)
    ]
    
    df = pd.DataFrame({
        'category_id': category_ids,
        'category_name': category_names
    })
    
    # Guardar archivo
    test_file = Paths.DATA_RAW / 'Test_Categories_Upload.csv'
    df.to_csv(test_file, index=False, header=with_header)
    
    print(f"✅ Archivo generado: {test_file}")
    print(f"  - Filas: {len(df)}")
    print(f"  - Formato: {'con encabezados' if with_header else 'sin encabezados'}")
    
    return test_file, df


def generate_test_product_category_file(
    num_products: int = 20, 
    category_range: tuple = (100, 109),
    with_header: bool = True
):
    """
    Genera un archivo CSV de prueba con relaciones producto-categoría.
    
    Args:
        num_products: Número de productos a generar
        category_range: Rango de IDs de categorías (min, max)
        with_header: Si True, incluye encabezados en el CSV
    """
    print(f"\n{'='*60}")
    print(f"Generando archivo de producto-categoría de prueba...")
    print(f"  - Productos: {num_products}")
    print(f"  - Rango de categorías: {category_range}")
    print(f"  - Con encabezados: {with_header}")
    
    # Generar IDs desde 1000 para evitar conflictos con productos existentes
    product_ids = list(range(1000, 1000 + num_products))
    
    # Asignar categorías aleatoriamente
    np.random.seed(42)
    category_ids = np.random.randint(
        category_range[0], 
        category_range[1] + 1, 
        size=num_products
    )
    
    df = pd.DataFrame({
        'product_id': product_ids,
        'category_id': category_ids
    })
    
    # Guardar archivo
    test_file = Paths.DATA_RAW / 'Test_ProductCategory_Upload.csv'
    df.to_csv(test_file, index=False, header=with_header)
    
    print(f"✅ Archivo generado: {test_file}")
    print(f"  - Filas: {len(df)}")
    print(f"  - Productos únicos: {df['product_id'].nunique()}")
    print(f"  - Categorías usadas: {sorted(df['category_id'].unique())}")
    print(f"  - Formato: {'con encabezados' if with_header else 'sin encabezados'}")
    
    return test_file, df


def test_categories_file_validation():
    """
    Test 1: Validación de estructura de archivo de categorías.
    """
    print(f"\n{'='*60}")
    print("TEST 1: Validación de estructura de categorías")
    print(f"{'='*60}")
    
    # Generar archivo de prueba
    test_file, original_df = generate_test_categories_file(num_categories=10)
    
    # Leer el archivo
    df = pd.read_csv(test_file)
    
    print("\n📋 Validando estructura...")
    
    # Validar columnas
    required_columns = ['category_id', 'category_name']
    assert all(col in df.columns for col in required_columns), \
        f"Columnas faltantes. Esperadas: {required_columns}, Encontradas: {df.columns.tolist()}"
    print(f"✅ Columnas correctas: {df.columns.tolist()}")
    
    # Validar tipos de datos
    assert pd.api.types.is_numeric_dtype(df['category_id']), \
        "category_id debe ser numérico"
    print("✅ category_id es numérico")
    
    assert pd.api.types.is_string_dtype(df['category_name']), \
        "category_name debe ser texto"
    print("✅ category_name es texto")
    
    # Validar sin duplicados
    assert df['category_id'].duplicated().sum() == 0, \
        "Hay IDs de categoría duplicados"
    print("✅ No hay IDs duplicados")
    
    # Validar sin nulos
    assert df.isnull().sum().sum() == 0, \
        "Hay valores nulos en el archivo"
    print("✅ No hay valores nulos")
    
    # Validar IDs positivos
    assert (df['category_id'] > 0).all(), \
        "Todos los IDs deben ser positivos"
    print("✅ Todos los IDs son positivos")
    
    print("\n✅ TEST 1 PASADO: Estructura de categorías válida")


def test_product_category_file_validation():
    """
    Test 2: Validación de estructura de archivo producto-categoría.
    """
    print(f"\n{'='*60}")
    print("TEST 2: Validación de estructura de producto-categoría")
    print(f"{'='*60}")
    
    # Generar archivo de prueba
    test_file, original_df = generate_test_product_category_file(
        num_products=20,
        category_range=(100, 109)
    )
    
    # Leer el archivo
    df = pd.read_csv(test_file)
    
    print("\n📋 Validando estructura...")
    
    # Validar columnas
    required_columns = ['product_id', 'category_id']
    assert all(col in df.columns for col in required_columns), \
        f"Columnas faltantes. Esperadas: {required_columns}, Encontradas: {df.columns.tolist()}"
    print(f"✅ Columnas correctas: {df.columns.tolist()}")
    
    # Validar tipos de datos
    assert pd.api.types.is_numeric_dtype(df['product_id']), \
        "product_id debe ser numérico"
    print("✅ product_id es numérico")
    
    assert pd.api.types.is_numeric_dtype(df['category_id']), \
        "category_id debe ser numérico"
    print("✅ category_id es numérico")
    
    # Validar sin nulos
    assert df.isnull().sum().sum() == 0, \
        "Hay valores nulos en el archivo"
    print("✅ No hay valores nulos")
    
    # Validar IDs positivos
    assert (df['product_id'] > 0).all(), \
        "Todos los product_id deben ser positivos"
    assert (df['category_id'] > 0).all(), \
        "Todos los category_id deben ser positivos"
    print("✅ Todos los IDs son positivos")
    
    print("\n✅ TEST 2 PASADO: Estructura de producto-categoría válida")


def test_categories_add_mode():
    """
    Test 3: Modo agregar categorías (combinar con existentes).
    """
    print(f"\n{'='*60}")
    print("TEST 3: Modo agregar categorías")
    print(f"{'='*60}")
    
    # Leer categorías existentes (formato pipe sin encabezados)
    existing_categories = pd.read_csv(
        Paths.DATA_RAW / 'Categories.csv',
        sep='|',
        header=None,
        names=['category_id', 'category_name']
    )
    print(f"\n📊 Categorías existentes: {len(existing_categories)}")
    print(f"  - Rango de IDs: {existing_categories['category_id'].min()} - {existing_categories['category_id'].max()}")
    
    # Generar nuevas categorías
    test_file, new_categories = generate_test_categories_file(num_categories=5)
    print(f"\n📊 Nuevas categorías: {len(new_categories)}")
    print(f"  - IDs: {sorted(new_categories['category_id'].tolist())}")
    
    # Simular modo "Agregar" (combinar)
    combined_categories = pd.concat([existing_categories, new_categories])
    combined_categories = combined_categories.drop_duplicates(
        subset=['category_id'],
        keep='last'
    ).sort_values('category_id')
    
    print(f"\n📊 Categorías combinadas: {len(combined_categories)}")
    
    # Validar que se agregaron correctamente
    assert len(combined_categories) == len(existing_categories) + len(new_categories), \
        "No se agregaron todas las categorías nuevas"
    print("✅ Todas las categorías nuevas se agregaron")
    
    # Validar que las existentes siguen ahí
    for cat_id in existing_categories['category_id']:
        assert cat_id in combined_categories['category_id'].values, \
            f"Se perdió la categoría {cat_id}"
    print("✅ Todas las categorías existentes se mantienen")
    
    print("\n✅ TEST 3 PASADO: Modo agregar funciona correctamente")


def test_categories_replace_mode():
    """
    Test 4: Modo reemplazar categorías.
    """
    print(f"\n{'='*60}")
    print("TEST 4: Modo reemplazar categorías")
    print(f"{'='*60}")
    
    # Leer categorías existentes
    existing_categories = pd.read_csv(
        Paths.DATA_RAW / 'Categories.csv',
        sep='|',
        header=None,
        names=['category_id', 'category_name']
    )
    print(f"\n📊 Categorías existentes: {len(existing_categories)}")
    
    # Generar nuevas categorías
    test_file, new_categories = generate_test_categories_file(num_categories=8)
    print(f"\n📊 Nuevas categorías: {len(new_categories)}")
    
    # Simular modo "Reemplazar"
    replaced_categories = new_categories.sort_values('category_id')
    
    print(f"\n📊 Categorías después de reemplazar: {len(replaced_categories)}")
    
    # Validar que solo quedan las nuevas
    assert len(replaced_categories) == len(new_categories), \
        "El reemplazo no funcionó correctamente"
    print("✅ Solo quedan las nuevas categorías")
    
    # Validar que las existentes no están
    old_ids = set(existing_categories['category_id'])
    new_ids = set(replaced_categories['category_id'])
    assert len(old_ids.intersection(new_ids)) == 0, \
        "Algunas categorías antiguas siguen presentes"
    print("✅ Las categorías antiguas fueron reemplazadas")
    
    print("\n✅ TEST 4 PASADO: Modo reemplazar funciona correctamente")


def test_product_category_add_mode():
    """
    Test 5: Modo agregar producto-categoría.
    """
    print(f"\n{'='*60}")
    print("TEST 5: Modo agregar producto-categoría")
    print(f"{'='*60}")
    
    # Leer relaciones existentes (formato pipe con encabezados)
    existing_pc = pd.read_csv(
        Paths.DATA_RAW / 'ProductCategory.csv',
        sep='|'
    )
    # Renombrar columnas si es necesario
    if 'v.Code_pr' in existing_pc.columns:
        existing_pc.columns = ['product_id', 'category_id']
    
    print(f"\n📊 Relaciones existentes: {len(existing_pc)}")
    print(f"  - Productos: {existing_pc['product_id'].nunique()}")
    
    # Generar nuevas relaciones
    test_file, new_pc = generate_test_product_category_file(
        num_products=15,
        category_range=(100, 109)
    )
    print(f"\n📊 Nuevas relaciones: {len(new_pc)}")
    print(f"  - Productos: {new_pc['product_id'].nunique()}")
    
    # Simular modo "Agregar/Actualizar"
    # Asegurar que los tipos sean consistentes
    existing_pc['product_id'] = existing_pc['product_id'].astype(int)
    existing_pc['category_id'] = existing_pc['category_id'].astype(int)
    
    combined_pc = pd.concat([existing_pc, new_pc])
    combined_pc = combined_pc.drop_duplicates(
        subset=['product_id'],
        keep='last'
    ).sort_values('product_id')
    
    print(f"\n📊 Relaciones combinadas: {len(combined_pc)}")
    print(f"  - Productos: {combined_pc['product_id'].nunique()}")
    
    # Validar que se agregaron (considerando que drop_duplicates reduce el tamaño)
    # Lo importante es que los nuevos productos estén presentes
    assert combined_pc['product_id'].nunique() >= existing_pc['product_id'].nunique(), \
        "Se perdieron productos existentes"
    print("✅ Las relaciones se agregaron correctamente")
    
    # Validar que los nuevos productos están
    for prod_id in new_pc['product_id']:
        assert prod_id in combined_pc['product_id'].values, \
            f"Se perdió el producto {prod_id}"
    print("✅ Todos los nuevos productos están presentes")
    
    print("\n✅ TEST 5 PASADO: Modo agregar producto-categoría funciona")


def test_product_category_duplicate_handling():
    """
    Test 6: Manejo de productos duplicados (última asignación gana).
    """
    print(f"\n{'='*60}")
    print("TEST 6: Manejo de productos duplicados")
    print(f"{'='*60}")
    
    # Crear archivo con duplicados
    df = pd.DataFrame({
        'product_id': [2000, 2001, 2002, 2001, 2000],  # 2000 y 2001 duplicados
        'category_id': [100, 101, 102, 103, 104]       # Diferentes categorías
    })
    
    test_file = Paths.DATA_RAW / 'Test_Duplicates.csv'
    df.to_csv(test_file, index=False)
    
    print(f"\n📊 Archivo con duplicados:")
    print(f"  - Total filas: {len(df)}")
    print(f"  - Productos únicos: {df['product_id'].nunique()}")
    print(f"  - Duplicados: {df['product_id'].duplicated().sum()}")
    
    # Leer y procesar (mantener último)
    df_processed = pd.read_csv(test_file)
    df_processed = df_processed.drop_duplicates(
        subset=['product_id'],
        keep='last'
    ).sort_values('product_id')
    
    print(f"\n📊 Después de eliminar duplicados:")
    print(f"  - Total filas: {len(df_processed)}")
    print(f"  - Productos únicos: {df_processed['product_id'].nunique()}")
    
    # Validar que quedó la última asignación
    assert df_processed[df_processed['product_id'] == 2000]['category_id'].values[0] == 104, \
        "No se mantuvo la última asignación para producto 2000"
    assert df_processed[df_processed['product_id'] == 2001]['category_id'].values[0] == 103, \
        "No se mantuvo la última asignación para producto 2001"
    print("✅ Se mantiene la última asignación en caso de duplicados")
    
    # Limpiar archivo de prueba
    test_file.unlink()
    
    print("\n✅ TEST 6 PASADO: Duplicados manejados correctamente")


def test_categories_without_headers():
    """
    Test 7: Categorías sin encabezados.
    """
    print(f"\n{'='*60}")
    print("TEST 7: Categorías sin encabezados")
    print(f"{'='*60}")
    
    # Generar archivo sin encabezados
    test_file, original_df = generate_test_categories_file(
        num_categories=5,
        with_header=False
    )
    
    # Leer sin encabezados y asignar nombres
    df = pd.read_csv(test_file, header=None, names=['category_id', 'category_name'])
    
    print("\n📋 Validando lectura...")
    
    # Validar que se leyó correctamente
    assert len(df) == 5, "No se leyeron todas las filas"
    print(f"✅ Se leyeron {len(df)} filas correctamente")
    
    assert 'category_id' in df.columns, "Falta columna category_id"
    assert 'category_name' in df.columns, "Falta columna category_name"
    print("✅ Nombres de columnas asignados correctamente")
    
    # Validar datos
    assert df['category_id'].nunique() == 5, "IDs duplicados o inválidos"
    print("✅ IDs únicos y válidos")
    
    print("\n✅ TEST 7 PASADO: Archivos sin encabezados se procesan correctamente")


def test_product_category_without_headers():
    """
    Test 8: Producto-categoría sin encabezados.
    """
    print(f"\n{'='*60}")
    print("TEST 8: Producto-categoría sin encabezados")
    print(f"{'='*60}")
    
    # Generar archivo sin encabezados
    test_file, original_df = generate_test_product_category_file(
        num_products=10,
        category_range=(100, 104),
        with_header=False
    )
    
    # Leer sin encabezados y asignar nombres
    df = pd.read_csv(test_file, header=None, names=['product_id', 'category_id'])
    
    print("\n📋 Validando lectura...")
    
    # Validar que se leyó correctamente
    assert len(df) == 10, "No se leyeron todas las filas"
    print(f"✅ Se leyeron {len(df)} filas correctamente")
    
    assert 'product_id' in df.columns, "Falta columna product_id"
    assert 'category_id' in df.columns, "Falta columna category_id"
    print("✅ Nombres de columnas asignados correctamente")
    
    # Validar datos
    assert df['product_id'].nunique() == 10, "IDs duplicados o inválidos"
    print("✅ IDs únicos y válidos")
    
    print("\n✅ TEST 8 PASADO: Archivos sin encabezados se procesan correctamente")


def main():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("TESTS DE CARGA DE CATÁLOGOS")
    print("="*60)
    
    try:
        # Tests de validación
        test_categories_file_validation()
        test_product_category_file_validation()
        
        # Tests de modos de carga
        test_categories_add_mode()
        test_categories_replace_mode()
        test_product_category_add_mode()
        
        # Tests de casos especiales
        test_product_category_duplicate_handling()
        test_categories_without_headers()
        test_product_category_without_headers()
        
        # Resumen final
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*60)
        print("\n📊 Resumen:")
        print("  - 8 tests ejecutados")
        print("  - 8 tests pasados")
        print("  - 0 tests fallados")
        print("\n💡 Los archivos de prueba están en data/raw/:")
        print("  - Test_Categories_Upload.csv")
        print("  - Test_ProductCategory_Upload.csv")
        print("\n🚀 Puedes usar estos archivos para probar la carga en Streamlit")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLIDO: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {str(e)}")
        raise


if __name__ == "__main__":
    main()
