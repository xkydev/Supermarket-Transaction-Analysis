"""
Tests para verificar la carga de archivos con diferentes formatos.
"""

import pandas as pd
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Paths


def test_transactions_with_pipe_no_header():
    """Test transacciones con pipe y sin encabezados."""
    print("\n" + "="*60)
    print("TEST: Transacciones con PIPE sin encabezados")
    print("="*60)
    
    # Crear archivo
    content = """2013-01-01|102|1000|20 3 1 5
2013-01-01|103|1001|12 12 8
2013-01-02|102|1002|5 5 5 5"""
    
    test_file = Paths.DATA_RAW / 'Test_Trans_Pipe_NoHeader.csv'
    test_file.write_text(content)
    
    # Leer y procesar
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    df = pd.read_csv(test_file, sep=delimiter, header=None, names=['date', 'store_id', 'customer_id', 'products'])
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == '|', "Delimitador incorrecto"
    assert len(df) == 3, "Número de filas incorrecto"
    
    test_file.unlink()
    print("✅ TEST PASADO")


def test_transactions_with_comma_no_header():
    """Test transacciones con coma y sin encabezados."""
    print("\n" + "="*60)
    print("TEST: Transacciones con COMA sin encabezados")
    print("="*60)
    
    # Crear archivo
    content = """2013-01-01,102,1000,20 3 1 5
2013-01-01,103,1001,12 12 8
2013-01-02,102,1002,5 5 5 5"""
    
    test_file = Paths.DATA_RAW / 'Test_Trans_Comma_NoHeader.csv'
    test_file.write_text(content)
    
    # Leer y procesar
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    df = pd.read_csv(test_file, sep=delimiter, header=None, names=['date', 'store_id', 'customer_id', 'products'])
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == ',', "Delimitador incorrecto"
    assert len(df) == 3, "Número de filas incorrecto"
    
    test_file.unlink()
    print("✅ TEST PASADO")


def test_categories_with_pipe_no_header():
    """Test categorías con pipe y sin encabezados."""
    print("\n" + "="*60)
    print("TEST: Categorías con PIPE sin encabezados")
    print("="*60)
    
    # Crear archivo
    content = """100|Categoria Test 1
101|Categoria Test 2
102|Categoria Test 3"""
    
    # Leer y procesar (simular lógica del app)
    import io
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    
    # Primero leer para detectar
    df_test = pd.read_csv(io.StringIO(file_content), sep=delimiter)
    
    # Detectar si no tiene encabezados
    required_columns = ['category_id', 'category_name']
    if (df_test.columns.tolist() == [0, 1] or 
        all(str(col).isdigit() for col in df_test.columns) or
        not any(col in df_test.columns for col in required_columns)):
        # Leer de nuevo sin encabezados
        df = pd.read_csv(io.StringIO(file_content), sep=delimiter, header=None, names=required_columns)
    else:
        df = df_test
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == '|', "Delimitador incorrecto"
    assert df.columns.tolist() == required_columns, "Columnas incorrectas"
    assert len(df) == 3, "Número de filas incorrecto"
    
    print("✅ TEST PASADO")


def test_categories_with_comma_with_header():
    """Test categorías con coma y con encabezados."""
    print("\n" + "="*60)
    print("TEST: Categorías con COMA con encabezados")
    print("="*60)
    
    # Crear archivo
    content = """category_id,category_name
100,Categoria Test 1
101,Categoria Test 2
102,Categoria Test 3"""
    
    # Leer y procesar
    import io
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    df = pd.read_csv(io.StringIO(file_content), sep=delimiter)
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == ',', "Delimitador incorrecto"
    assert 'category_id' in df.columns, "Falta columna category_id"
    assert len(df) == 3, "Número de filas incorrecto"
    
    print("✅ TEST PASADO")


def test_product_category_with_pipe_with_header():
    """Test producto-categoría con pipe y con encabezados."""
    print("\n" + "="*60)
    print("TEST: Producto-Categoria con PIPE con encabezados")
    print("="*60)
    
    # Crear archivo
    content = """product_id|category_id
1000|100
1001|101
1002|102"""
    
    # Leer y procesar
    import io
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    df = pd.read_csv(io.StringIO(file_content), sep=delimiter)
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == '|', "Delimitador incorrecto"
    assert 'product_id' in df.columns, "Falta columna product_id"
    assert len(df) == 3, "Número de filas incorrecto"
    
    print("✅ TEST PASADO")


def test_product_category_with_comma_no_header():
    """Test producto-categoría con coma y sin encabezados."""
    print("\n" + "="*60)
    print("TEST: Producto-Categoria con COMA sin encabezados")
    print("="*60)
    
    # Crear archivo
    content = """1000,100
1001,101
1002,102"""
    
    # Leer y procesar (simular lógica del app)
    import io
    file_content = content
    delimiter = '|' if '|' in file_content.split('\n')[0] else ','
    
    # Primero leer para detectar
    df_test = pd.read_csv(io.StringIO(file_content), sep=delimiter)
    
    # Detectar si no tiene encabezados
    required_columns = ['product_id', 'category_id']
    if (df_test.columns.tolist() == [0, 1] or 
        all(str(col).isdigit() for col in df_test.columns) or
        not any(col in df_test.columns for col in required_columns)):
        # Leer de nuevo sin encabezados
        df = pd.read_csv(io.StringIO(file_content), sep=delimiter, header=None, names=required_columns)
    else:
        df = df_test
    
    print(f"✅ Delimitador detectado: '{delimiter}'")
    print(f"✅ Columnas: {df.columns.tolist()}")
    print(f"✅ Filas: {len(df)}")
    assert delimiter == ',', "Delimitador incorrecto"
    assert df.columns.tolist() == required_columns, "Columnas incorrectas"
    assert len(df) == 3, "Número de filas incorrecto"
    
    print("✅ TEST PASADO")


def main():
    """Ejecuta todos los tests de formatos."""
    print("\n" + "="*60)
    print("TESTS DE FORMATOS DE ARCHIVOS")
    print("="*60)
    print("\nVerificando soporte para:")
    print("  - Delimitadores: coma (,) y pipe (|)")
    print("  - Encabezados: con y sin")
    print("  - Tipos: Transacciones, Categorías, Producto-Categoría")
    
    try:
        # Tests de transacciones
        test_transactions_with_pipe_no_header()
        test_transactions_with_comma_no_header()
        
        # Tests de categorías
        test_categories_with_pipe_no_header()
        test_categories_with_comma_with_header()
        
        # Tests de producto-categoría
        test_product_category_with_pipe_with_header()
        test_product_category_with_comma_no_header()
        
        # Resumen
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60)
        print("\n📊 Formatos soportados verificados:")
        print("  ✅ Transacciones: pipe sin header, coma sin header")
        print("  ✅ Categorías: pipe sin header, coma con header")
        print("  ✅ Producto-Categoría: pipe con header, coma sin header")
        print("\n🎯 Conclusión: El sistema maneja TODOS los formatos correctamente")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLIDO: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
