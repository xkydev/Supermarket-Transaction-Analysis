"""
Módulo para crear visualizaciones interactivas con Plotly.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import VisualizationConfig

logger = logging.getLogger(__name__)


class Visualizer:
    """Clase para crear visualizaciones interactivas."""
    
    def __init__(self, template: str = None):
        """
        Inicializa el visualizador.
        
        Args:
            template: Template de Plotly a usar. Por defecto usa config.
        """
        self.template = template or VisualizationConfig.PLOT_TEMPLATE
        self.height = VisualizationConfig.HEIGHT
        self.width = VisualizationConfig.WIDTH
        
        # Configuración común de gráficos
        self.config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    
    def plot_top_products(
        self,
        product_metrics: pd.DataFrame,
        n: int = 10,
        metric: str = 'total_quantity'
    ) -> go.Figure:
        """
        Crea gráfico de barras horizontales de top productos.
        
        Args:
            product_metrics: DataFrame con métricas de productos
            n: Número de productos a mostrar
            metric: Métrica a visualizar ('total_quantity', 'unique_customers', etc.)
            
        Returns:
            Figura de Plotly
        """
        top_n = product_metrics.nlargest(n, metric).sort_values(metric)
        
        # Crear etiquetas con ID de producto y categoría
        top_n['label'] = top_n.apply(
            lambda x: f"Producto {x['product_id']} - {x['category_name']}" 
            if pd.notna(x['category_name']) else f"Producto {x['product_id']}",
            axis=1
        )
        
        metric_labels = {
            'total_quantity': 'Unidades Vendidas',
            'unique_customers': 'Clientes Únicos',
            'product_frequency': 'Frecuencia de Compra'
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_n['label'],
            x=top_n[metric],
            orientation='h',
            marker=dict(
                color=top_n[metric],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_n[metric].apply(lambda x: f'{x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         f'{metric_labels.get(metric, metric)}: %{{x:,.0f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top {n} Productos por {metric_labels.get(metric, metric)}',
            xaxis_title=metric_labels.get(metric, metric),
            yaxis_title='',
            template=self.template,
            height=max(400, n * 40),
            showlegend=False
        )
        
        return fig
    
    def plot_top_customers(
        self,
        customer_metrics: pd.DataFrame,
        n: int = 10,
        metric: str = 'frequency'
    ) -> go.Figure:
        """
        Crea gráfico de barras horizontales de top clientes.
        
        Args:
            customer_metrics: DataFrame con métricas de clientes
            n: Número de clientes a mostrar
            metric: Métrica a visualizar ('frequency', 'total_quantity', etc.)
            
        Returns:
            Figura de Plotly
        """
        top_n = customer_metrics.nlargest(n, metric).sort_values(metric)
        
        metric_labels = {
            'frequency': 'Número de Transacciones',
            'total_quantity': 'Unidades Compradas',
            'unique_products': 'Productos Únicos'
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_n['customer_id'].astype(str),
            x=top_n[metric],
            orientation='h',
            marker=dict(
                color=top_n[metric],
                colorscale='Blues',
                showscale=True
            ),
            text=top_n[metric].apply(lambda x: f'{x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>Cliente %{y}</b><br>' +
                         f'{metric_labels.get(metric, metric)}: %{{x:,.0f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top {n} Clientes por {metric_labels.get(metric, metric)}',
            xaxis_title=metric_labels.get(metric, metric),
            yaxis_title='ID Cliente',
            template=self.template,
            height=max(400, n * 40),
            showlegend=False
        )
        
        return fig
    
    def plot_category_distribution(
        self,
        category_metrics: pd.DataFrame,
        n: int = 10,
        chart_type: str = 'bar'
    ) -> go.Figure:
        """
        Crea gráfico de distribución por categoría.
        
        Args:
            category_metrics: DataFrame con métricas por categoría
            n: Número de categorías a mostrar
            chart_type: Tipo de gráfico ('bar' o 'pie')
            
        Returns:
            Figura de Plotly
        """
        top_n = category_metrics.nlargest(n, 'total_units')
        
        if chart_type == 'pie':
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=top_n['category_name'],
                values=top_n['total_units'],
                hovertemplate='<b>%{label}</b><br>' +
                             'Unidades: %{value:,.0f}<br>' +
                             'Porcentaje: %{percent}<br>' +
                             '<extra></extra>',
                textposition='auto',
                textinfo='label+percent'
            ))
            
            fig.update_layout(
                title=f'Top {n} Categorías por Volumen de Ventas',
                template=self.template,
                height=self.height
            )
        else:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_n['category_name'],
                y=top_n['total_units'],
                marker=dict(
                    color=top_n['total_units'],
                    colorscale='Plasma',
                    showscale=True
                ),
                text=top_n['total_units'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             'Unidades: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Top {n} Categorías por Volumen de Ventas',
                xaxis_title='',
                yaxis_title='Unidades Vendidas',
                template=self.template,
                height=self.height,
                showlegend=False
            )
            
            # Rotar etiquetas si son muchas
            fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def plot_time_series(
        self,
        transactions_df: pd.DataFrame,
        freq: str = 'D',
        metric: str = 'quantity'
    ) -> go.Figure:
        """
        Crea gráfico de serie temporal.
        
        Args:
            transactions_df: DataFrame de transacciones
            freq: Frecuencia de agregación ('D', 'W', 'M')
            metric: Métrica a graficar
            
        Returns:
            Figura de Plotly
        """
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Agrupar por fecha
        if metric == 'quantity':
            ts = df.groupby(pd.Grouper(key='date', freq=freq))['quantity'].sum().reset_index()
            y_label = 'Unidades Vendidas'
        elif metric == 'transactions':
            ts = df.groupby(pd.Grouper(key='date', freq=freq))['transaction_id'].nunique().reset_index()
            ts.columns = ['date', 'quantity']
            y_label = 'Número de Transacciones'
        else:
            ts = df.groupby(pd.Grouper(key='date', freq=freq))['customer_id'].nunique().reset_index()
            ts.columns = ['date', 'quantity']
            y_label = 'Clientes Únicos'
        
        freq_labels = {'D': 'Diaria', 'W': 'Semanal', 'M': 'Mensual'}
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ts['date'],
            y=ts['quantity'],
            mode='lines+markers',
            name=y_label,
            line=dict(color='#636EFA', width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                         f'{y_label}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Tendencia {freq_labels.get(freq, "")} de {y_label}',
            xaxis_title='Fecha',
            yaxis_title=y_label,
            template=self.template,
            height=self.height,
            hovermode='x unified'
        )
        
        # Agregar rangeslider
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def plot_heatmap_day_hour(self, transactions_df: pd.DataFrame) -> go.Figure:
        """
        Crea heatmap de transacciones por día de semana.
        
        Args:
            transactions_df: DataFrame de transacciones
            
        Returns:
            Figura de Plotly
        """
        df = transactions_df.copy()
        
        # Crear pivot table
        pivot = df.groupby(['day_name', 'day_of_week']).size().reset_index(name='count')
        
        # Ordenar días de semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot['day_name'] = pd.Categorical(pivot['day_name'], categories=day_order, ordered=True)
        pivot = pivot.sort_values('day_name')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pivot['day_name'],
            y=pivot['count'],
            marker=dict(
                color=pivot['count'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Transacciones")
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Transacciones: %{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Distribución de Transacciones por Día de la Semana',
            xaxis_title='Día de la Semana',
            yaxis_title='Número de Transacciones',
            template=self.template,
            height=self.height
        )
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        customer_metrics: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Crea heatmap de correlación entre variables.
        
        Args:
            customer_metrics: DataFrame con métricas de clientes
            features: Lista de features a incluir. Si None, usa las principales.
            
        Returns:
            Figura de Plotly
        """
        if features is None:
            features = [
                'frequency', 'recency', 'total_quantity',
                'unique_products', 'unique_categories', 'avg_basket_size'
            ]
        
        # Filtrar features disponibles
        available_features = [f for f in features if f in customer_metrics.columns]
        
        # Calcular correlación
        corr = customer_metrics[available_features].corr()
        
        # Crear labels más legibles
        labels_map = {
            'frequency': 'Frecuencia',
            'recency': 'Recencia',
            'total_quantity': 'Cantidad Total',
            'unique_products': 'Productos Únicos',
            'unique_categories': 'Categorías Únicas',
            'avg_basket_size': 'Tamaño Promedio Canasta'
        }
        
        labels = [labels_map.get(f, f) for f in available_features]
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Matriz de Correlación - Métricas de Cliente',
            template=self.template,
            height=600,
            width=700
        )
        
        return fig
    
    def plot_boxplot(
        self,
        data: pd.DataFrame,
        column: str,
        group_by: Optional[str] = None,
        title: str = None
    ) -> go.Figure:
        """
        Crea boxplot para análisis de distribuciones.
        
        Args:
            data: DataFrame con los datos
            column: Columna a graficar
            group_by: Columna para agrupar (opcional)
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        if group_by:
            groups = data[group_by].unique()
            for group in groups:
                subset = data[data[group_by] == group]
                fig.add_trace(go.Box(
                    y=subset[column],
                    name=str(group),
                    boxmean='sd',
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '%{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        else:
            fig.add_trace(go.Box(
                y=data[column],
                name=column,
                boxmean='sd',
                hovertemplate='%{y:.2f}<br><extra></extra>'
            ))
        
        fig.update_layout(
            title=title or f'Distribución de {column}',
            yaxis_title=column,
            template=self.template,
            height=self.height,
            showlegend=bool(group_by)
        )
        
        return fig
    
    def plot_histogram(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: str = None
    ) -> go.Figure:
        """
        Crea histograma.
        
        Args:
            data: DataFrame con los datos
            column: Columna a graficar
            bins: Número de bins
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data[column],
            nbinsx=bins,
            marker=dict(
                color='#636EFA',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Rango: %{x}<br>Frecuencia: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or f'Distribución de {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            template=self.template,
            height=self.height
        )
        
        return fig
    
    def plot_scatter_2d(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = None
    ) -> go.Figure:
        """
        Crea scatter plot 2D.
        
        Args:
            data: DataFrame con los datos
            x: Columna para eje X
            y: Columna para eje Y
            color: Columna para colorear puntos (opcional)
            size: Columna para tamaño de puntos (opcional)
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            template=self.template,
            height=self.height,
            title=title or f'{y} vs {x}'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x:.2f}, %{y:.2f}</b><extra></extra>'
        )
        
        return fig
