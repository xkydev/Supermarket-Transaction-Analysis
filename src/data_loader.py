"""
Módulo para cargar y validar los datos de transacciones.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Paths

logger = logging.getLogger(__name__)


class DataLoader:
    """Clase para cargar y validar datos de transacciones de supermercado."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_path: Ruta a la carpeta con los datos. Por defecto usa config.Paths.DATA_RAW
        """
        self.data_path = data_path or Paths.DATA_RAW
        
    def load_categories(self) -> pd.DataFrame:
        """
        Carga el catálogo de categorías.
        
        Returns:
            DataFrame con columnas [category_id, category_name]
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo Categories.csv
            ValueError: Si el archivo tiene formato inválido
        """
        file_path = self.data_path / "Categories.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        try:
            df = pd.read_csv(
                file_path,
                delimiter='|',
                header=None,
                names=['category_id', 'category_name'],
                dtype={'category_id': int, 'category_name': str}
            )
            
            # Limpiar espacios en blanco
            df['category_name'] = df['category_name'].str.strip()
            
            logger.info(f"Cargadas {len(df)} categorías")
            return df
            
        except Exception as e:
            raise ValueError(f"Error al cargar categorías: {str(e)}")
    
    def load_product_category(self) -> pd.DataFrame:
        """
        Carga la relación producto-categoría.
        
        Returns:
            DataFrame con columnas [product_id, category_id]
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo ProductCategory.csv
            ValueError: Si el archivo tiene formato inválido
        """
        file_path = self.data_path / "ProductCategory.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        try:
            df = pd.read_csv(
                file_path,
                delimiter='|',
                header=None,
                names=['product_id', 'category_id'],
                dtype=str  # Cargar como string primero para verificar headers
            )
            
            # Eliminar fila de cabecera si existe (buscar por texto común en headers)
            if df.iloc[0]['product_id'].lower().startswith('v.code'):
                df = df.iloc[1:].reset_index(drop=True)
                logger.info("Header detectado y eliminado en ProductCategory.csv")
            
            # Convertir a tipos numéricos
            df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce')
            df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce')
            
            # Eliminar filas con valores nulos
            df = df.dropna()
            
            # Convertir a enteros
            df['product_id'] = df['product_id'].astype(int)
            df['category_id'] = df['category_id'].astype(int)
            
            logger.info(f"Cargadas {len(df)} relaciones producto-categoría")
            return df
            
        except Exception as e:
            raise ValueError(f"Error al cargar producto-categoría: {str(e)}")
    
    def load_transactions(self, store_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Carga las transacciones de una o más tiendas.
        
        Args:
            store_ids: Lista de IDs de tiendas a cargar. Si es None, carga todas.
            
        Returns:
            DataFrame con columnas [date, store_id, customer_id, products]
            
        Raises:
            FileNotFoundError: Si no se encuentra algún archivo de transacciones
            ValueError: Si hay error al cargar los datos
        """
        if store_ids is None:
            # Buscar todos los archivos de transacciones
            transaction_files = list(self.data_path.glob("*_Tran.csv"))
            if not transaction_files:
                raise FileNotFoundError(
                    f"No se encontraron archivos de transacciones en {self.data_path}"
                )
            store_ids = [int(f.stem.split('_')[0]) for f in transaction_files]
        
        dfs = []
        for store_id in store_ids:
            file_path = self.data_path / f"{store_id}_Tran.csv"
            
            if not file_path.exists():
                logger.warning(f"No se encontró archivo para tienda {store_id}: {file_path}")
                continue
            
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter='|',
                    header=None,
                    names=['date', 'store_id', 'customer_id', 'products'],
                    dtype={'store_id': int, 'customer_id': int, 'products': str}
                )
                
                # Validar que store_id coincida
                if df['store_id'].nunique() == 1 and df['store_id'].iloc[0] == store_id:
                    dfs.append(df)
                    logger.info(
                        f"Cargadas {len(df):,} transacciones de tienda {store_id}"
                    )
                else:
                    logger.warning(
                        f"El store_id en el archivo no coincide con {store_id}"
                    )
                    
            except Exception as e:
                logger.error(f"Error al cargar transacciones de tienda {store_id}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No se pudo cargar ningún archivo de transacciones")
        
        # Combinar todos los DataFrames
        result = pd.concat(dfs, ignore_index=True)
        
        # Convertir fechas a datetime
        result['date'] = pd.to_datetime(result['date'], format='%Y-%m-%d')
        
        logger.info(
            f"Total cargado: {len(result):,} transacciones de {len(store_ids)} tiendas"
        )
        
        return result
    
    def validate_data_integrity(
        self,
        transactions: pd.DataFrame,
        product_category: pd.DataFrame,
        categories: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Valida la integridad de los datos cargados.
        
        Args:
            transactions: DataFrame de transacciones
            product_category: DataFrame de producto-categoría
            categories: DataFrame de categorías
            
        Returns:
            Diccionario con resultados de validación
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validar que todas las categorías en product_category existen
        invalid_categories = set(product_category['category_id']) - set(categories['category_id'])
        if invalid_categories:
            validation['warnings'].append(
                f"Categorías referenciadas pero no definidas: {invalid_categories}"
            )
        
        # Validar rangos de fechas
        if transactions['date'].isnull().any():
            validation['errors'].append("Existen fechas nulas en las transacciones")
            validation['valid'] = False
        
        # Validar IDs de clientes
        if transactions['customer_id'].isnull().any():
            validation['errors'].append("Existen customer_id nulos")
            validation['valid'] = False
        
        # Validar productos
        if transactions['products'].isnull().any():
            validation['errors'].append("Existen listas de productos nulas")
            validation['valid'] = False
        
        return validation
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carga todos los datos necesarios para el análisis.
        
        Returns:
            Tupla con (transactions, product_category, categories)
            
        Raises:
            FileNotFoundError: Si falta algún archivo
            ValueError: Si hay errores de validación
        """
        logger.info("Iniciando carga de datos...")
        
        categories = self.load_categories()
        product_category = self.load_product_category()
        transactions = self.load_transactions()
        
        # Validar integridad
        validation = self.validate_data_integrity(transactions, product_category, categories)
        
        if not validation['valid']:
            raise ValueError(f"Validación falló: {validation['errors']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(warning)
        
        logger.info("Carga de datos completada exitosamente")
        
        return transactions, product_category, categories
