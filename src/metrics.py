"""
Módulo para calcular métricas y KPIs del negocio.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Clase para calcular métricas de negocio."""
    
    def __init__(self):
        """Inicializa el calculador de métricas."""
        pass
    
    def calculate_kpis(self, transactions_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calcula los KPIs principales del negocio.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            Diccionario con KPIs principales
        """
        kpis = {
            'total_units': int(transactions_df['quantity'].sum()),
            'total_transactions': int(transactions_df['transaction_id'].nunique()),
            'total_customers': int(transactions_df['customer_id'].nunique()),
            'total_products': int(transactions_df['product_id'].nunique()),
            'total_categories': int(transactions_df['category_id'].nunique()),
            'total_stores': int(transactions_df['store_id'].nunique()),
            'avg_basket_size': float(transactions_df.groupby('transaction_id')['product_id'].nunique().mean()),
            'avg_units_per_transaction': float(transactions_df.groupby('transaction_id')['quantity'].sum().mean()),
            'date_range': {
                'start': transactions_df['date'].min(),
                'end': transactions_df['date'].max(),
                'days': (transactions_df['date'].max() - transactions_df['date'].min()).days
            }
        }
        
        return kpis
    
    def calculate_growth_metrics(
        self,
        transactions_df: pd.DataFrame,
        period: str = 'M'
    ) -> pd.DataFrame:
        """
        Calcula métricas de crecimiento por período.
        
        Args:
            transactions_df: DataFrame de transacciones
            period: Período de agregación ('D'=día, 'W'=semana, 'M'=mes)
            
        Returns:
            DataFrame con métricas por período y tasa de crecimiento
        """
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Agrupar por período
        growth = df.groupby(pd.Grouper(key='date', freq=period)).agg({
            'quantity': 'sum',
            'transaction_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        
        growth.columns = ['period', 'total_units', 'transactions', 'customers']
        
        # Calcular tasas de crecimiento
        growth['units_growth'] = growth['total_units'].pct_change() * 100
        growth['transactions_growth'] = growth['transactions'].pct_change() * 100
        growth['customers_growth'] = growth['customers'].pct_change() * 100
        
        return growth
    
    def calculate_category_performance(
        self,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula métricas de desempeño por categoría.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            DataFrame con métricas por categoría
        """
        category_metrics = transactions_df.groupby('category_name').agg({
            'quantity': 'sum',
            'transaction_id': 'nunique',
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        
        category_metrics.columns = [
            'category_name', 'total_units', 'transactions',
            'customers', 'products'
        ]
        
        # Calcular porcentajes
        total_units = category_metrics['total_units'].sum()
        category_metrics['percentage'] = (category_metrics['total_units'] / total_units * 100)
        
        # Ordenar por volumen
        category_metrics = category_metrics.sort_values('total_units', ascending=False)
        
        return category_metrics
    
    def calculate_store_performance(
        self,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula métricas de desempeño por tienda.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            DataFrame con métricas por tienda
        """
        store_metrics = transactions_df.groupby('store_id').agg({
            'quantity': 'sum',
            'transaction_id': 'nunique',
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        
        store_metrics.columns = [
            'store_id', 'total_units', 'transactions',
            'customers', 'products'
        ]
        
        # Calcular métricas derivadas
        store_metrics['avg_basket_size'] = (
            store_metrics['total_units'] / store_metrics['transactions']
        )
        store_metrics['avg_transactions_per_customer'] = (
            store_metrics['transactions'] / store_metrics['customers']
        )
        
        return store_metrics
    
    def calculate_customer_segmentation_summary(
        self,
        customer_metrics: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calcula resumen de segmentación de clientes.
        
        Args:
            customer_metrics: DataFrame con métricas por cliente
            
        Returns:
            Diccionario con estadísticas de segmentación
        """
        summary = {
            'total_customers': len(customer_metrics),
            'avg_frequency': float(customer_metrics['frequency'].mean()),
            'avg_recency': float(customer_metrics['recency'].mean()),
            'avg_total_quantity': float(customer_metrics['total_quantity'].mean()),
            'customers_by_frequency': {
                'low (1-2 trans)': int((customer_metrics['frequency'] <= 2).sum()),
                'medium (3-10 trans)': int(((customer_metrics['frequency'] > 2) & 
                                           (customer_metrics['frequency'] <= 10)).sum()),
                'high (11-50 trans)': int(((customer_metrics['frequency'] > 10) & 
                                          (customer_metrics['frequency'] <= 50)).sum()),
                'very_high (>50 trans)': int((customer_metrics['frequency'] > 50).sum())
            },
            'customers_by_recency': {
                'recent (0-30 days)': int((customer_metrics['recency'] <= 30).sum()),
                'moderate (31-90 days)': int(((customer_metrics['recency'] > 30) & 
                                             (customer_metrics['recency'] <= 90)).sum()),
                'old (>90 days)': int((customer_metrics['recency'] > 90).sum())
            }
        }
        
        return summary
    
    def calculate_product_penetration(
        self,
        product_metrics: pd.DataFrame,
        total_customers: int
    ) -> pd.DataFrame:
        """
        Calcula penetración de productos (% de clientes que lo compraron).
        
        Args:
            product_metrics: DataFrame con métricas por producto
            total_customers: Número total de clientes
            
        Returns:
            DataFrame con métricas de penetración
        """
        df = product_metrics.copy()
        df['penetration'] = (df['unique_customers'] / total_customers * 100)
        df['avg_frequency'] = df['product_frequency'] / df['unique_customers']
        
        return df.sort_values('penetration', ascending=False)
    
    def calculate_temporal_patterns(
        self,
        transactions_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calcula patrones temporales de compra.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            Diccionario con DataFrames de patrones temporales
        """
        df = transactions_df.copy()
        
        patterns = {
            'by_day_of_week': df.groupby('day_of_week').agg({
                'quantity': 'sum',
                'transaction_id': 'nunique'
            }).reset_index(),
            
            'by_month': df.groupby('month').agg({
                'quantity': 'sum',
                'transaction_id': 'nunique'
            }).reset_index(),
            
            'by_weekday_name': df.groupby('day_name').agg({
                'quantity': 'sum',
                'transaction_id': 'nunique'
            }).reset_index(),
            
            'weekend_vs_weekday': df.groupby('is_weekend').agg({
                'quantity': 'sum',
                'transaction_id': 'nunique'
            }).reset_index()
        }
        
        # Ordenar día de semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['by_weekday_name']['day_name'] = pd.Categorical(
            patterns['by_weekday_name']['day_name'],
            categories=day_order,
            ordered=True
        )
        patterns['by_weekday_name'] = patterns['by_weekday_name'].sort_values('day_name')
        
        return patterns
    
    def calculate_basket_analysis(
        self,
        transactions_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calcula análisis de canasta de compra.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            Diccionario con análisis de canasta
        """
        # Agrupar por transacción
        basket_sizes = transactions_df.groupby('transaction_id').agg({
            'product_id': 'nunique',
            'quantity': 'sum',
            'category_id': 'nunique'
        }).reset_index()
        
        basket_sizes.columns = ['transaction_id', 'unique_products', 'total_quantity', 'unique_categories']
        
        analysis = {
            'avg_products_per_basket': float(basket_sizes['unique_products'].mean()),
            'median_products_per_basket': float(basket_sizes['unique_products'].median()),
            'avg_quantity_per_basket': float(basket_sizes['total_quantity'].mean()),
            'median_quantity_per_basket': float(basket_sizes['total_quantity'].median()),
            'avg_categories_per_basket': float(basket_sizes['unique_categories'].mean()),
            'distribution': basket_sizes['unique_products'].value_counts().sort_index().to_dict()
        }
        
        return analysis
