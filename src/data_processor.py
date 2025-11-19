"""
Módulo para procesar y transformar datos de transacciones.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import Paths

logger = logging.getLogger(__name__)


class DataProcessor:
    """Clase para procesar y transformar datos de transacciones."""
    
    def __init__(self):
        """Inicializa el procesador de datos."""
        pass
    
    def expand_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Expande las transacciones para crear registros individuales por producto.
        
        Cada transacción con múltiples productos se convierte en múltiples registros,
        uno por cada producto, calculando la cantidad como la frecuencia del producto
        en la lista.
        
        Args:
            transactions: DataFrame con columnas [date, store_id, customer_id, products]
            
        Returns:
            DataFrame expandido con columnas [transaction_id, date, store_id, customer_id, 
                                              product_id, quantity]
        """
        logger.info(f"Expandiendo {len(transactions):,} transacciones...")
        
        expanded_rows = []
        
        for idx, row in transactions.iterrows():
            transaction_id = f"{row['store_id']}_{row['customer_id']}_{idx}"
            products_str = str(row['products']).strip()
            
            if not products_str or products_str == 'nan':
                continue
            
            # Parsear la lista de productos
            product_ids = products_str.split()
            
            # Contar frecuencia de cada producto
            product_counts = {}
            for pid in product_ids:
                try:
                    pid_int = int(pid)
                    product_counts[pid_int] = product_counts.get(pid_int, 0) + 1
                except ValueError:
                    logger.warning(f"ID de producto inválido: {pid} en transacción {idx}")
                    continue
            
            # Crear un registro por cada producto único
            for product_id, quantity in product_counts.items():
                expanded_rows.append({
                    'transaction_id': transaction_id,
                    'date': row['date'],
                    'store_id': row['store_id'],
                    'customer_id': row['customer_id'],
                    'product_id': product_id,
                    'quantity': quantity
                })
        
        result = pd.DataFrame(expanded_rows)
        
        logger.info(
            f"Expansión completada: {len(result):,} registros de producto "
            f"desde {len(transactions):,} transacciones"
        )
        
        return result
    
    def enrich_with_categories(
        self,
        transactions: pd.DataFrame,
        product_category: pd.DataFrame,
        categories: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enriquece las transacciones con información de categorías.
        
        Args:
            transactions: DataFrame expandido de transacciones
            product_category: DataFrame de producto-categoría
            categories: DataFrame de categorías
            
        Returns:
            DataFrame enriquecido con columnas adicionales [category_id, category_name]
        """
        logger.info("Enriqueciendo transacciones con categorías...")
        
        # Join con product_category
        result = transactions.merge(
            product_category,
            on='product_id',
            how='left'
        )
        
        # Join con categories
        result = result.merge(
            categories,
            on='category_id',
            how='left'
        )
        
        # Contar productos sin categoría
        missing_category = result['category_id'].isnull().sum()
        if missing_category > 0:
            logger.warning(
                f"{missing_category:,} registros ({missing_category/len(result)*100:.2f}%) "
                f"sin categoría asignada"
            )
        
        return result
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega features temporales derivadas de la fecha.
        
        Args:
            df: DataFrame con columna 'date'
            
        Returns:
            DataFrame con features temporales adicionales
        """
        logger.info("Agregando features temporales...")
        
        df = df.copy()
        
        # Asegurar que 'date' es datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Extraer componentes temporales
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        return df
    
    def calculate_transaction_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas a nivel de transacción.
        
        Args:
            df: DataFrame de transacciones expandidas y enriquecidas
            
        Returns:
            DataFrame con métricas agregadas por transaction_id
        """
        logger.info("Calculando métricas de transacción...")
        
        transaction_metrics = df.groupby('transaction_id').agg({
            'date': 'first',
            'store_id': 'first',
            'customer_id': 'first',
            'product_id': 'nunique',  # Número de productos únicos
            'quantity': 'sum',  # Total de unidades
            'category_id': 'nunique'  # Número de categorías únicas
        }).reset_index()
        
        transaction_metrics.columns = [
            'transaction_id', 'date', 'store_id', 'customer_id',
            'basket_size', 'total_quantity', 'category_diversity'
        ]
        
        return transaction_metrics
    
    def calculate_customer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas agregadas por cliente (RFM adaptado y diversidad).
        
        Args:
            df: DataFrame de transacciones expandidas y enriquecidas
            
        Returns:
            DataFrame con métricas por customer_id
        """
        logger.info("Calculando métricas de cliente...")
        
        # Fecha de referencia (última fecha en los datos)
        reference_date = df['date'].max()
        
        # Agrupar por cliente
        customer_metrics = df.groupby('customer_id').agg({
            'transaction_id': 'nunique',  # Frecuencia (número de transacciones)
            'date': 'max',  # Última fecha de compra
            'quantity': 'sum',  # Total de unidades compradas
            'product_id': 'nunique',  # Productos únicos
            'category_id': 'nunique',  # Categorías únicas
            'store_id': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Tienda preferida
        }).reset_index()
        
        customer_metrics.columns = [
            'customer_id', 'frequency', 'last_purchase_date',
            'total_quantity', 'unique_products', 'unique_categories',
            'preferred_store'
        ]
        
        # Calcular recency (días desde última compra)
        customer_metrics['recency'] = (
            reference_date - customer_metrics['last_purchase_date']
        ).dt.days
        
        # Calcular tamaño promedio de canasta
        customer_metrics['avg_basket_size'] = (
            customer_metrics['total_quantity'] / customer_metrics['frequency']
        )
        
        # Calcular diversidad de categorías (normalizada)
        customer_metrics['category_diversity_score'] = (
            customer_metrics['unique_categories'] / customer_metrics['frequency']
        )
        
        return customer_metrics
    
    def calculate_product_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas agregadas por producto.
        
        Args:
            df: DataFrame de transacciones expandidas y enriquecidas
            
        Returns:
            DataFrame con métricas por product_id
        """
        logger.info("Calculando métricas de producto...")
        
        product_metrics = df.groupby('product_id').agg({
            'transaction_id': 'nunique',  # Número de transacciones
            'customer_id': 'nunique',  # Clientes únicos
            'quantity': ['sum', 'mean'],  # Total y promedio de cantidad
            'category_id': 'first',  # Categoría del producto
            'category_name': 'first'  # Nombre de categoría
        }).reset_index()
        
        # Aplanar columnas multi-índice
        product_metrics.columns = [
            'product_id', 'product_frequency', 'unique_customers',
            'total_quantity', 'avg_quantity', 'category_id', 'category_name'
        ]
        
        return product_metrics
    
    def process_all(
        self,
        transactions: pd.DataFrame,
        product_category: pd.DataFrame,
        categories: pd.DataFrame
    ) -> dict:
        """
        Procesa todos los datos y genera todas las vistas necesarias.
        
        Args:
            transactions: DataFrame crudo de transacciones
            product_category: DataFrame de producto-categoría
            categories: DataFrame de categorías
            
        Returns:
            Diccionario con todos los DataFrames procesados:
                - transactions_expanded: transacciones expandidas y enriquecidas
                - transaction_metrics: métricas por transacción
                - customer_metrics: métricas por cliente
                - product_metrics: métricas por producto
                - categories: catálogo de categorías
        """
        logger.info("Iniciando procesamiento completo de datos...")
        
        # 1. Expandir transacciones
        expanded = self.expand_transactions(transactions)
        
        # 2. Enriquecer con categorías
        enriched = self.enrich_with_categories(expanded, product_category, categories)
        
        # 3. Agregar features temporales
        enriched = self.add_temporal_features(enriched)
        
        # 4. Calcular métricas
        transaction_metrics = self.calculate_transaction_metrics(enriched)
        customer_metrics = self.calculate_customer_metrics(enriched)
        product_metrics = self.calculate_product_metrics(enriched)
        
        result = {
            'transactions_expanded': enriched,
            'transaction_metrics': transaction_metrics,
            'customer_metrics': customer_metrics,
            'product_metrics': product_metrics,
            'categories': categories
        }
        
        logger.info("Procesamiento completado exitosamente")
        
        # Log de resumen
        logger.info(f"Resumen del procesamiento:")
        logger.info(f"  - Transacciones expandidas: {len(enriched):,} registros")
        logger.info(f"  - Transacciones únicas: {len(transaction_metrics):,}")
        logger.info(f"  - Clientes únicos: {len(customer_metrics):,}")
        logger.info(f"  - Productos únicos: {len(product_metrics):,}")
        logger.info(f"  - Categorías: {len(categories):,}")
        
        return result
