# 🛒 Análisis de Transacciones de Supermercado

Sistema completo de análisis de transacciones de supermercado con dashboard interactivo, segmentación de clientes, sistema de recomendación y carga dinámica de datos.

## 🚀 Características Principales

### 📊 Dashboard Ejecutivo

- Métricas clave (unidades vendidas, transacciones, clientes, productos)
- Top 10 productos y clientes
- Análisis por categorías
- Filtros dinámicos (fecha, tienda, categoría)

### ⏰ Análisis Temporal

- Series de tiempo (diario/semanal/mensual)
- Heatmaps de patrones de compra
- Análisis de estacionalidad
- Identificación de tendencias

### 📈 Análisis de Distribuciones

- Boxplots y histogramas
- Detección de outliers
- Análisis de correlaciones

### 👥 Segmentación de Clientes (K-Means)

- Clustering automático con K óptimo
- Visualización 2D/3D de segmentos
- Perfiles detallados por cluster
- Estrategias de negocio recomendadas

### 🎯 Sistema de Recomendación

- **Por Cliente**: Filtrado colaborativo basado en clientes similares
- **Por Producto**: Market Basket Analysis con reglas de asociación (Apriori)
- Caché inteligente para performance (50-200x más rápido)
- Configuración avanzada de parámetros

### 📤 Carga de Nuevos Datos (3 Tipos de Archivos)

#### 📊 Transacciones

- Upload de archivos CSV con transacciones
- Expansión automática a registros individuales por producto
- Enriquecimiento con categorías
- Features temporales automáticos

#### 🏷️ Categorías

- Gestión del catálogo de categorías
- Modo agregar o reemplazar
- Validación de IDs y nombres

#### 🔗 Producto-Categoría

- Relaciones producto-categoría
- Verificación de integridad referencial
- Detección de categorías faltantes

#### ✨ Detección Automática de Formato

- **Delimitadores**: Coma (`,`) y Pipe (`|`) detectados automáticamente
- **Encabezados**: Con y sin encabezados soportados
- **Preview**: Visualización antes de procesar
- **Validación exhaustiva**: Estructura, tipos, nulos, duplicados
- **Progress bar**: Feedback en tiempo real
- **Recálculo automático**: Métricas actualizadas tras carga

## 📦 Instalación

### Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes y entornos)
- 2GB RAM mínimo (recomendado 8GB para datasets grandes)

### Setup

```bash
# Clonar repositorio
git clone https://github.com/xkydev/Supermarket-Transaction-Analysis.git
cd Supermarket-Transaction-Analysis

# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Crear entorno virtual e instalar dependencias
uv venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias desde pyproject.toml
uv pip install -e .
```

> **💡 Nota sobre dependencias**: Este proyecto usa **uv** como gestor de paquetes y todas las dependencias están definidas en `pyproject.toml`. No se usa `requirements.txt` ni `pip` directamente.

## 🎯 Uso

### 1. Iniciar aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### 2. Primera ejecución automática

En la primera ejecución, la aplicación:

- ✅ Detecta automáticamente que no hay datos procesados
- ✅ Carga los archivos originales desde `data/raw/`
- ✅ Procesa y expande las transacciones (1.1M → 15.2M registros)
- ✅ Enriquece con categorías y features temporales
- ✅ Calcula métricas de clientes, productos y transacciones
- ✅ Guarda todo en `data/processed/`
- ✅ Recarga automáticamente la aplicación

**Tiempo estimado**: 2-5 minutos (dependiendo del hardware)

### 3. Navegar por las páginas

1. **Dashboard Ejecutivo**: Vista general de métricas
2. **Análisis Temporal**: Patrones en el tiempo
3. **Análisis de Distribuciones**: Estadísticas y outliers
4. **Análisis de Correlaciones**: Relaciones entre variables
5. **Segmentación de Clientes**: Clustering con K-Means
6. **Sistema de Recomendación**: Sugerencias personalizadas
7. **Carga de Nuevos Datos**: Upload de archivos CSV (4 tabs)
   - 📊 Cargar Transacciones
   - 🏷️ Cargar Categorías
   - 🔗 Cargar Producto-Categoría
   - 📈 Ver Estado Actual

### 4. Resetear a datos originales (opcional)

Si deseas volver a los datos iniciales después de hacer cambios:

1. Ve a la página **"Carga de Nuevos Datos"**
2. Tab **"📈 Ver Estado Actual"**
3. Clic en **"🔄 Resetear a Datos Originales"**
4. El sistema automáticamente:
   - Elimina todos los datos procesados
   - Recarga los archivos originales
   - Procesa todo desde cero
   - Recarga la aplicación
5. Presiona F5 para ver los cambios

## 📁 Estructura del Proyecto

```text
project/
├── app.py                          # Aplicación principal Streamlit
├── config.py                       # Configuraciones centralizadas
├── pyproject.toml                  # Dependencias y configuración del proyecto
├── data/
│   ├── raw/                       # Datos originales
│   │   ├── Categories.csv         # 50 categorías
│   │   ├── ProductCategory.csv    # 112k relaciones
│   │   ├── 102_Tran.csv          # 14 MB - Tienda 102
│   │   ├── 103_Tran.csv          # 21 MB - Tienda 103
│   │   ├── 107_Tran.csv          # 13 MB - Tienda 107
│   │   ├── 110_Tran.csv          # 6.9 MB - Tienda 110
│   └── processed/                 # Datos procesados
│       ├── transactions_expanded.csv    # 15.2M registros expandidos
│       ├── customer_metrics.csv         # 131k clientes con métricas
│       ├── product_metrics.csv          # 449 productos con métricas
│       └── transaction_metrics.csv      # Métricas temporales
├── src/                           # Módulos principales
│   ├── data_loader.py            # Carga y validación de datos
│   ├── data_processor.py         # Transformación y feature engineering
│   ├── metrics.py                # Cálculo de KPIs y métricas
│   ├── visualizations.py         # Gráficos con Plotly
│   ├── clustering.py             # Segmentación K-Means
│   └── recommender.py            # Sistema de recomendación
├── tests/                         # Suite de tests
│   ├── test_data_upload.py       # Tests de carga de transacciones
│   ├── test_catalog_upload.py    # Tests de carga de catálogos (8 tests)
│   ├── test_file_formats.py      # Tests de formatos (6 tests)
│   ├── test_recommender.py       # Tests del sistema de recomendación
│   ├── test_cache.py             # Tests del sistema de caché
│   ├── test_clustering.py        # Tests de segmentación
│   ├── test_visualizations.py    # Tests de visualizaciones
│   └── test_data_pipeline.py     # Tests del pipeline completo
├── utils/                         # Utilidades adicionales
├── main.py                        # Script principal alternativo
├── prompt.md                      # Especificaciones del proyecto
├── pyproject.toml                 # Dependencias y configuración
├── uv.lock                        # Lock file de uv
├── .python-version                # Versión de Python
└── .gitignore                     # Archivos ignorados por Git
```

## 🔧 Formato de Datos

El sistema **detecta automáticamente** el delimitador (`,` o `|`) y si el archivo tiene encabezados.

### Archivos de Transacciones (4 columnas)

#### Formato 1: Coma con encabezados

```csv
date,store_id,customer_id,products
2013-01-01,102,1000,20 3 1 5
2013-01-01,103,1001,12 12 8
```

#### Formato 2: Coma sin encabezados

```csv
2013-01-01,102,1000,20 3 1 5
2013-01-01,103,1001,12 12 8
```

#### Formato 3: Pipe con encabezados

```csv
date|store_id|customer_id|products
2013-01-01|102|1000|20 3 1 5
2013-01-01|103|1001|12 12 8
```

#### Formato 4: Pipe sin encabezados

```csv
2013-01-01|102|1000|20 3 1 5
2013-01-01|103|1001|12 12 8
```

**Columnas (en orden):**

1. `date`: YYYY-MM-DD
2. `store_id`: ID numérico de tienda
3. `customer_id`: ID numérico de cliente
4. `products`: IDs de productos separados por espacios

**Ejemplo:** `"20 20 3"` = 2 unidades del producto 20, 1 del producto 3

### Categories.csv (2 columnas)

Soporta todos los formatos: `,` o `|`, con o sin encabezados

```csv
category_id,category_name
1,Bebidas
2,Lácteos
```

```csv
1|Bebidas
2|Lácteos
```

### ProductCategory.csv (2 columnas)

Soporta todos los formatos: `,` o `|`, con o sin encabezados

```csv
product_id,category_id
1,1
2,2
```

```csv
1|1
2|2
```

## 🧪 Testing

Suite completa de **20+ tests** que validan todas las funcionalidades:

```bash
# Test de carga de transacciones (validación, procesamiento, métricas)
python tests/test_data_upload.py

# Test de carga de catálogos (8 tests: categorías y producto-categoría)
python tests/test_catalog_upload.py

# Test de formatos de archivos (6 tests: pipe/coma, con/sin headers)
python tests/test_file_formats.py

# Test del sistema de recomendación (filtrado colaborativo + Apriori)
python tests/test_recommender.py

# Test del sistema de caché (performance y validación)
python tests/test_cache.py

# Test de segmentación (K-Means clustering)
python tests/test_clustering.py

# Test de visualizaciones (gráficos Plotly)
python tests/test_visualizations.py

# Test del pipeline completo (end-to-end)
python tests/test_data_pipeline.py
```

### 📊 Cobertura de Tests

- ✅ **Carga de datos**: 12 formatos diferentes (coma/pipe, con/sin headers)
- ✅ **Validación**: Estructura, tipos, nulos, duplicados, integridad referencial
- ✅ **Procesamiento**: Expansión, enriquecimiento, features temporales
- ✅ **Métricas**: Customer, product, transaction metrics
- ✅ **Recomendaciones**: Collaborative filtering + Market Basket Analysis
- ✅ **Clustering**: K óptimo, perfiles, visualizaciones
- ✅ **Caché**: Performance, invalidación, TTL

## 📊 Datos de Ejemplo

### Dataset Principal (4 Tiendas)

El proyecto incluye datos reales de transacciones:

| Métrica | Valor |
|---------|-------|
| **Período** | Año 2013 completo |
| **Tiendas** | 4 (102, 103, 107, 110) |
| **Transacciones únicas** | 1.1M |
| **Registros expandidos** | 15.2M |
| **Clientes únicos** | 131,186 |
| **Productos únicos** | 449 |
| **Categorías** | 50 |
| **Tamaño archivos raw** | 55+ MB |
| **Tamaño procesado** | ~1.2 GB |
