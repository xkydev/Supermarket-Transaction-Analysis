"""
Módulo para sistema de recomendación de productos.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import RecommenderConfig

logger = logging.getLogger(__name__)


class RecommenderSystem:
    """Sistema de recomendación de productos usando filtrado colaborativo y market basket analysis."""
    
    def __init__(self):
        """Inicializa el sistema de recomendación."""
        self.customer_item_matrix = None
        self.item_similarity_matrix = None
        self.association_rules = None
        self.product_names = None
        
    def build_customer_item_matrix(
        self,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construye matriz de interacción cliente x producto.
        
        Args:
            transactions_df: DataFrame de transacciones expandidas
            
        Returns:
            Matriz pivotada con clientes como filas y productos como columnas
        """
        logger.info("Construyendo matriz cliente-producto...")
        
        # Agrupar por cliente y producto, sumar cantidades
        customer_product = transactions_df.groupby(
            ['customer_id', 'product_id']
        )['quantity'].sum().reset_index()
        
        # Crear matriz pivotada
        matrix = customer_product.pivot_table(
            index='customer_id',
            columns='product_id',
            values='quantity',
            fill_value=0
        )
        
        self.customer_item_matrix = matrix
        
        logger.info(f"Matriz construida: {matrix.shape[0]:,} clientes x {matrix.shape[1]:,} productos")
        
        return matrix
    
    def calculate_customer_similarity_efficient(
        self,
        target_customer_id: int,
        top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula similaridad entre un cliente y los K clientes más similares.
        Versión eficiente en memoria que no requiere matriz completa.
        
        Args:
            target_customer_id: ID del cliente objetivo
            top_k: Número de clientes similares a considerar
            
        Returns:
            Tupla de (índices de clientes similares, scores de similaridad)
        """
        if self.customer_item_matrix is None:
            raise ValueError("Debe construir la matriz cliente-producto primero")
        
        logger.info(f"Calculando top {top_k} clientes similares...")
        
        # Obtener vector del cliente objetivo
        customer_vector = np.array(self.customer_item_matrix.loc[target_customer_id].values).reshape(1, -1)
        customer_vector_normalized = normalize(customer_vector, norm='l2')
        
        # Normalizar toda la matriz
        matrix_normalized = normalize(self.customer_item_matrix.values, norm='l2', axis=1)
        
        # Calcular similaridad solo con el cliente objetivo (vector vs matriz)
        similarities = cosine_similarity(customer_vector_normalized, matrix_normalized)[0]
        
        # Obtener top K más similares (excluyendo el mismo cliente)
        customer_idx = self.customer_item_matrix.index.get_loc(target_customer_id)
        similarities[customer_idx] = -1  # Excluir el mismo cliente
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_k_scores = similarities[top_k_indices]
        
        logger.info(f"Top {top_k} clientes similares calculados")
        
        return top_k_indices, top_k_scores
    
    def get_customer_recommendations(
        self,
        customer_id: int,
        transactions_df: pd.DataFrame,
        top_n: Optional[int] = None,
        min_similarity: float = 0.1,
        top_k_similar: int = 1000
    ) -> pd.DataFrame:
        """
        Recomienda productos para un cliente basado en clientes similares.
        Versión optimizada que calcula similaridad solo con top K clientes.
        
        Args:
            customer_id: ID del cliente
            transactions_df: DataFrame de transacciones
            top_n: Número de productos a recomendar
            min_similarity: Umbral mínimo de similaridad
            top_k_similar: Número de clientes similares a considerar
            
        Returns:
            DataFrame con productos recomendados y scores
        """
        top_n = top_n or RecommenderConfig.TOP_N
        
        # Construir matriz si no existe
        if self.customer_item_matrix is None:
            self.build_customer_item_matrix(transactions_df)
        
        # Verificar que el cliente existe
        if customer_id not in self.customer_item_matrix.index:
            logger.warning(f"Cliente {customer_id} no encontrado")
            return pd.DataFrame()
        
        logger.info(f"Generando recomendaciones para cliente {customer_id}...")
        
        # Calcular similaridad eficientemente (solo top K)
        similar_indices, similar_scores = self.calculate_customer_similarity_efficient(
            customer_id, 
            top_k=top_k_similar
        )
        
        # Productos que el cliente ya compró
        customer_products = set(
            self.customer_item_matrix.loc[customer_id][
                self.customer_item_matrix.loc[customer_id] > 0
            ].index
        )
        
        # Scores de productos basados en clientes similares
        product_scores = {}
        
        for idx, similarity in zip(similar_indices, similar_scores):
            # Filtrar por umbral de similaridad
            if similarity < min_similarity:
                continue
            
            similar_customer_id = self.customer_item_matrix.index[idx]
            
            # Productos comprados por el cliente similar
            similar_customer_products = self.customer_item_matrix.loc[similar_customer_id]
            similar_customer_products = similar_customer_products[similar_customer_products > 0]
            
            # Acumular scores para productos no comprados aún
            for product_id, quantity in similar_customer_products.items():
                if product_id not in customer_products:
                    if product_id not in product_scores:
                        product_scores[product_id] = 0
                    product_scores[product_id] += similarity * quantity
        
        if not product_scores:
            logger.info("No se encontraron productos para recomendar")
            return pd.DataFrame()
        
        # Crear DataFrame de recomendaciones
        recommendations = pd.DataFrame([
            {'product_id': pid, 'score': score}
            for pid, score in product_scores.items()
        ]).sort_values('score', ascending=False).head(top_n)
        
        # Enriquecer con información del producto
        if 'product_name' in transactions_df.columns:
            product_info = transactions_df[['product_id', 'product_name', 'category_name']].drop_duplicates(subset=['product_id'])
            recommendations = recommendations.merge(product_info, on='product_id', how='left')
        else:
            # Si no hay product_name, solo agregar category_name (tomar primera categoría si hay duplicados)
            product_info = transactions_df[['product_id', 'category_name']].drop_duplicates(subset=['product_id'])
            recommendations = recommendations.merge(product_info, on='product_id', how='left')
            recommendations['product_name'] = 'Producto ' + recommendations['product_id'].astype(str)
        
        # Calcular número de clientes similares que compraron cada producto
        recommendations['similar_customers_count'] = recommendations['product_id'].apply(
            lambda pid: int((self.customer_item_matrix[pid] > 0).sum())
        )
        
        # Normalizar score a 0-100
        recommendations['score'] = (
            recommendations['score'] / recommendations['score'].max() * 100
        ).round(2)
        
        logger.info(f"Se generaron {len(recommendations)} recomendaciones")
        
        return recommendations
    
    def build_association_rules(
        self,
        transactions_df: pd.DataFrame,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None,
        max_transactions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Genera reglas de asociación usando Apriori.
        Optimizado para grandes datasets usando sampling.
        
        Args:
            transactions_df: DataFrame de transacciones
            min_support: Soporte mínimo
            min_confidence: Confianza mínima
            min_lift: Lift mínimo
            max_transactions: Máximo número de transacciones a procesar
            
        Returns:
            DataFrame con reglas de asociación
        """
        min_support = min_support or RecommenderConfig.MIN_SUPPORT
        min_confidence = min_confidence or RecommenderConfig.MIN_CONFIDENCE
        min_lift = min_lift or RecommenderConfig.MIN_LIFT
        max_transactions = max_transactions if max_transactions is not None else RecommenderConfig.MAX_TRANSACTIONS
        
        logger.info("Generando reglas de asociación con Market Basket Analysis...")
        logger.info(f"Parámetros: support={min_support}, confidence={min_confidence}, lift={min_lift}")
        
        # Agrupar productos por transacción
        transactions_grouped = transactions_df.groupby('transaction_id')['product_id'].apply(list)
        
        # Usar sample si hay demasiadas transacciones y max_transactions está definido
        if max_transactions and len(transactions_grouped) > max_transactions:
            logger.info(f"Dataset grande detectado ({len(transactions_grouped):,} transacciones)")
            logger.info(f"Usando sample aleatorio de {max_transactions:,} transacciones para eficiencia...")
            transactions_grouped = transactions_grouped.sample(n=max_transactions, random_state=42)
        
        transactions_list = transactions_grouped.values
        logger.info(f"Analizando {len(transactions_list):,} transacciones...")
        
        # Encodear transacciones para mlxtend (más eficiente sin DataFrame intermedio)
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Aplicar Apriori
        logger.info("Ejecutando algoritmo Apriori...")
        frequent_itemsets = apriori(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            low_memory=True
        )
        
        if len(frequent_itemsets) == 0:
            logger.warning("No se encontraron itemsets frecuentes. Reduce min_support.")
            return pd.DataFrame()
        
        logger.info(f"Itemsets frecuentes encontrados: {len(frequent_itemsets):,}")
        
        # Generar reglas de asociación
        logger.info("Generando reglas de asociación...")
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets)
        )
        
        if len(rules) == 0:
            logger.warning("No se encontraron reglas. Ajusta los parámetros.")
            return pd.DataFrame()
        
        # Filtrar por lift
        rules = rules[rules['lift'] >= min_lift]
        
        logger.info(f"Reglas generadas: {len(rules):,}")
        
        # Convertir frozensets a listas para mejor manejo
        rules['antecedents_list'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents_list'] = rules['consequents'].apply(lambda x: list(x))
        
        self.association_rules = rules
        
        return rules
    
    def get_product_recommendations(
        self,
        product_id: int,
        transactions_df: pd.DataFrame,
        top_n: Optional[int] = None,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None,
        max_transactions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Recomienda productos que se compran junto al producto dado.
        
        Args:
            product_id: ID del producto
            transactions_df: DataFrame de transacciones
            top_n: Número de productos a recomendar
            min_support: Soporte mínimo
            min_confidence: Confianza mínima
            min_lift: Lift mínimo
            
        Returns:
            DataFrame con productos recomendados y métricas
        """
        top_n = top_n or RecommenderConfig.TOP_N
        
        # Generar reglas si no existen
        if self.association_rules is None or len(self.association_rules) == 0:
            self.build_association_rules(
                transactions_df,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                max_transactions=max_transactions
            )
        
        if self.association_rules is None or len(self.association_rules) == 0:
            logger.warning("No hay reglas de asociación disponibles")
            return pd.DataFrame()
        
        logger.info(f"Generando recomendaciones para producto {product_id}...")
        
        # Filtrar reglas donde el producto está en antecedents
        relevant_rules = self.association_rules[
            self.association_rules['antecedents_list'].apply(lambda x: product_id in x)
        ].copy()
        
        if len(relevant_rules) == 0:
            logger.info(f"No se encontraron reglas para producto {product_id}")
            return pd.DataFrame()
        
        # Extraer productos recomendados (consequents)
        recommendations = []
        
        for _, rule in relevant_rules.iterrows():
            for consequent_product in rule['consequents_list']:
                if consequent_product != product_id:  # Excluir el mismo producto
                    recommendations.append({
                        'product_id': consequent_product,
                        'support': rule['support'],
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
        
        if not recommendations:
            logger.info("No se encontraron productos para recomendar")
            return pd.DataFrame()
        
        # Crear DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Agrupar por producto (puede aparecer en múltiples reglas)
        recommendations_df = recommendations_df.groupby('product_id').agg({
            'support': 'max',
            'confidence': 'max',
            'lift': 'max'
        }).reset_index()
        
        # Ordenar por lift (mejor indicador de asociación)
        recommendations_df = recommendations_df.sort_values('lift', ascending=False).head(top_n)
        
        # Enriquecer con información del producto
        if 'product_name' in transactions_df.columns:
            product_info = transactions_df[['product_id', 'product_name', 'category_name']].drop_duplicates(subset=['product_id'])
            recommendations_df = recommendations_df.merge(product_info, on='product_id', how='left')
        else:
            # Si no hay product_name, solo agregar category_name (tomar primera categoría si hay duplicados)
            product_info = transactions_df[['product_id', 'category_name']].drop_duplicates(subset=['product_id'])
            recommendations_df = recommendations_df.merge(product_info, on='product_id', how='left')
            recommendations_df['product_name'] = 'Producto ' + recommendations_df['product_id'].astype(str)
        
        # Calcular score normalizado (0-100)
        recommendations_df['score'] = (
            recommendations_df['lift'] / recommendations_df['lift'].max() * 100
        ).round(2)
        
        # Redondear métricas
        recommendations_df['support'] = (recommendations_df['support'] * 100).round(2)
        recommendations_df['confidence'] = (recommendations_df['confidence'] * 100).round(2)
        recommendations_df['lift'] = recommendations_df['lift'].round(2)
        
        logger.info(f"Se generaron {len(recommendations_df)} recomendaciones")
        
        return recommendations_df
    
    def get_product_statistics(
        self,
        product_id: int,
        transactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un producto.
        
        Args:
            product_id: ID del producto
            transactions_df: DataFrame de transacciones
            
        Returns:
            Diccionario con estadísticas
        """
        product_data = transactions_df[transactions_df['product_id'] == product_id]
        
        if len(product_data) == 0:
            return {}
        
        stats = {
            'product_name': product_data['product_name'].iloc[0] if 'product_name' in product_data.columns else f'Producto {product_id}',
            'category_name': product_data['category_name'].iloc[0] if pd.notna(product_data['category_name'].iloc[0]) else 'Sin categoría',
            'total_quantity': int(product_data['quantity'].sum()),
            'unique_customers': int(product_data['customer_id'].nunique()),
            'unique_transactions': int(product_data['transaction_id'].nunique()),
            'avg_quantity_per_transaction': float(product_data['quantity'].mean())
        }
        
        return stats
    
    def get_customer_statistics(
        self,
        customer_id: int,
        transactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un cliente.
        
        Args:
            customer_id: ID del cliente
            transactions_df: DataFrame de transacciones
            
        Returns:
            Diccionario con estadísticas
        """
        customer_data = transactions_df[transactions_df['customer_id'] == customer_id]
        
        if len(customer_data) == 0:
            return {}
        
        stats = {
            'total_transactions': int(customer_data['transaction_id'].nunique()),
            'total_quantity': int(customer_data['quantity'].sum()),
            'unique_products': int(customer_data['product_id'].nunique()),
            'unique_categories': int(customer_data['category_name'].nunique()),
            'favorite_category': customer_data.groupby('category_name')['quantity'].sum().idxmax(),
            'avg_products_per_transaction': float(
                customer_data.groupby('transaction_id')['product_id'].count().mean()
            )
        }
        
        return stats
