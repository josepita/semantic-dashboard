"""
Sistema de ayuda contextual para todas las funciones de la aplicación.

Cada función tiene 3 componentes de ayuda:
1. Fundamento teórico SEO
2. Cómo interpretar resultados
3. Mejores prácticas de uso
"""

from __future__ import annotations

from typing import Dict, TypedDict


class HelpContent(TypedDict):
    """Estructura de contenido de ayuda para cada función."""
    title: str
    seo_theory: str
    interpretation: str
    best_practices: str


# Diccionario centralizado con toda la ayuda
HELP_SECTIONS: Dict[str, HelpContent] = {
    "clustering": {
        "title": "Clustering de Contenido",
        "seo_theory": (
            "El clustering semántico agrupa URLs con contenido similar basándose en la proximidad "
            "vectorial de sus embeddings. En SEO, esto permite identificar canibalización de keywords "
            "(múltiples páginas compitiendo por la misma intención de búsqueda), detectar gaps de "
            "contenido (temas relacionados sin cobertura) y optimizar la arquitectura de información. "
            "Los motores de búsqueda modernos como Google usan embeddings similares (BERT, MUM) para "
            "entender la similitud semántica entre documentos, por lo que agrupar tu contenido de forma "
            "coherente mejora la relevancia temática y la autoridad topical."
        ),
        "interpretation": (
            "Cada cluster representa un tema o intención de búsqueda. URLs en el mismo cluster tienen "
            "contenido muy similar y probablemente compiten entre sí. Un **cluster grande** (muchas URLs) "
            "puede indicar canibalización de keywords. **Clusters pequeños** (2-3 URLs) pueden ser "
            "oportunidades para consolidar contenido. El **Silhouette Score** (0-1) mide la calidad: "
            "valores >0.5 indican clusters bien definidos, <0.3 sugieren solapamiento. URLs con etiqueta "
            "'noise' (-1) son outliers sin tema claro. El **número óptimo de clusters** (k) se detecta "
            "automáticamente buscando el máximo Silhouette Score."
        ),
        "best_practices": (
            "**Paso 1**: Carga un CSV con columna de URL y embeddings pre-calculados. **Paso 2**: Usa "
            "detección automática de k para encontrar el número óptimo de grupos. **Paso 3**: Revisa "
            "clusters grandes (>10 URLs) para identificar canibalización - considera consolidar contenido "
            "similar en una página autoritativa. **Paso 4**: Examina 'noise' URLs - pueden necesitar "
            "reescritura para enfocarse en un tema claro. **Paso 5**: Exporta resultados y úsalos para "
            "planificar internal linking estratégico (enlaza dentro de cada cluster y entre clusters "
            "relacionados). Ideal para auditorías de contenido y planificación de arquitectura de información."
        ),
    },
    "knowledge_graph": {
        "title": "Análisis de Grafo de Conocimiento",
        "seo_theory": (
            "Los Knowledge Graphs son fundamentales en SEO moderno. Google utiliza su Knowledge Graph para "
            "conectar entidades (personas, lugares, conceptos) y entender las relaciones entre ellas. Cuando "
            "tu contenido menciona entidades de forma coherente y frecuente, se beneficia del 'Quality Smear' "
            "(Patente US9183499B1): la calidad de entidades relacionadas se propaga entre sí. Un contenido que "
            "co-ocurre sistemáticamente con entidades de alta autoridad señala profundidad temática. El análisis "
            "NER (Named Entity Recognition) + Entity Linking a Wikidata te permite mapear tu contenido al mismo "
            "Knowledge Graph que usa Google, identificando gaps de entidades y optimizando la cobertura semántica."
        ),
        "interpretation": (
            "El grafo visualiza entidades (nodos) y sus relaciones (aristas). El **tamaño del nodo** representa "
            "la autoridad tópica (prominence + centralidad). **Nodos centrales** con muchas conexiones son "
            "conceptos clave de tu contenido. **Relaciones SPO** (Sujeto-Predicado-Objeto) muestran cómo se "
            "conectan las entidades. La **tabla de entidades** muestra: (1) Frecuencia consolidada = menciones "
            "totales, (2) Prominence sintáctica = importancia gramatical (sujetos/objetos pesan más), "
            "(3) Closeness/Betweenness = centralidad en el grafo. **QID de Wikidata** vincula entidades a la "
            "base canónica. Entidades con alta frecuencia pero baja centralidad pueden ser menciones superficiales "
            "sin desarrollo profundo."
        ),
        "best_practices": (
            "**Paso 1**: Usa texto plano (no HTML) en la columna de contenido para análisis limpio. **Paso 2**: "
            "Activa Entity Linking a Wikidata para vincular con el Knowledge Graph oficial. **Paso 3**: Configura "
            "tipos de entidad según tu nicho (ORG, PRODUCT para eCommerce; PERSON, EVENT para noticias). **Paso 4**: "
            "Usa la whitelist manual para forzar la inclusión de entidades clave de tu marca/nicho (recibirán boost "
            "3x en prominence). **Paso 5**: Revisa entidades de alta frecuencia pero baja centralidad - necesitan "
            "más contexto y relaciones. **Paso 6**: Identifica entidades ausentes comparando con competidores - son "
            "oportunidades de expansión. Ideal para content briefs, optimización on-page y estrategia de E-E-A-T."
        ),
    },
    "semantic_depth_score": {
        "title": "Semantic Depth Score (SDS)",
        "seo_theory": (
            "El SDS cuantifica la calidad del contenido más allá de la longitud o densidad de keywords. Combina "
            "dos dimensiones críticas: (1) **Entity Relevance (ER)**: mide la completitud del conocimiento mediante "
            "densidad y co-ocurrencia de entidades relevantes, capitalizando el efecto Quality Smear de la patente "
            "de Google; (2) **Vector Cohesion (CV)**: mide la coherencia narrativa calculando la distancia promedio "
            "de frases al centroide semántico del documento. Google MUM y BERT evalúan contenido de forma similar, "
            "premiando cobertura exhaustiva de entidades (E-E-A-T) y discurso cohesivo (user experience). Un SDS "
            "alto indica contenido que satisface completamente la intención de búsqueda con profundidad y enfoque."
        ),
        "interpretation": (
            "El SDS va de 0-100 con 3 rangos: **🔴 0-33 (Thin)**: baja densidad de entidades, dispersión alta - "
            "contenido superficial que necesita expansión profunda. **🟡 34-66 (Decente)**: fuerte en un componente "
            "pero débil en otro - por ejemplo, muchas entidades pero sin cohesión narrativa, o muy enfocado pero "
            "con pocas entidades. **🟢 67-100 (Óptimo)**: alta densidad de entidades relevantes + discurso cohesivo "
            "= autoridad temática. **Score ER** alto con **Score CV** bajo indica contenido disperso (menciona muchos "
            "temas sin profundizar). ER bajo con CV alto indica contenido enfocado pero superficial (falta cobertura "
            "de entidades). La **densidad de entidades** óptima varía por nicho: técnico/educativo >0.05, comercial "
            "0.02-0.04."
        ),
        "best_practices": (
            "**Paso 1**: Analiza tu top 10 páginas con mejor rendimiento para establecer tu baseline de SDS óptimo. "
            "**Paso 2**: Identifica páginas con SDS <50 - son candidatas prioritarias para reescritura. **Paso 3**: "
            "Para mejorar Score ER, añade entidades relacionadas del Knowledge Graph (usa análisis de competidores "
            "top-ranking). **Paso 4**: Para mejorar Score CV, reestructura el contenido agrupando ideas relacionadas, "
            "elimina secciones off-topic y refuerza transiciones entre párrafos. **Paso 5**: Monitorea SDS después "
            "de optimizaciones - aumentos de 10+ puntos correlacionan con mejoras de ranking. **Paso 6**: Usa SDS "
            "en content briefs para writers: especifica target SDS >70 y lista entidades obligatorias. Ideal para "
            "content quality audits y optimización on-page estratégica."
        ),
    },
    "linking_lab": {
        "title": "Laboratorio de Enlazado Interno",
        "seo_theory": (
            "El internal linking es uno de los factores de ranking más controlables. Distribuye PageRank, señala "
            "relevancia temática y mejora crawlability. El enlazado semántico (basado en similitud de embeddings) "
            "supera al keyword matching tradicional porque refleja cómo Google entiende relaciones entre páginas. "
            "El algoritmo PageRank flow muestra cómo se distribuye la autoridad: páginas con alto PR pero pocas "
            "outbound links son 'hoarders' (acumulan autoridad sin distribuir), páginas con bajo PR pero muchos "
            "inbound son 'candidates' (receptores potenciales). El linking strategy óptimo balancea: (1) enlazar "
            "contenido similar (clusters semánticos), (2) distribuir autoridad de hubs a contenido long-tail, "
            "(3) crear topic clusters con pillar pages."
        ),
        "interpretation": (
            "La tabla de **sugerencias de enlaces** muestra pares Source→Target con: **Similitud semántica** (0-1): "
            ">0.7 = muy relacionado, 0.5-0.7 = relacionado, <0.5 = débilmente relacionado. **PageRank Source/Target**: "
            "enlazar desde páginas con PR alto hacia PR bajo distribuye autoridad. **Ganancia de PR**: proyección de "
            "cuánto PR ganaría el target si recibe el enlace. **Prioridad**: métrica combinada que balancea similitud, "
            "ganancia de PR y estrategia. El **análisis de PageRank** muestra: páginas con alto PR son tus assets "
            "(distribuye su autoridad), páginas con bajo PR pero alto potencial necesitan enlaces internos, páginas "
            "huérfanas (sin inbound) son invisibles para bots."
        ),
        "best_practices": (
            "**Paso 1**: Carga CSV con URL, embeddings y datos de PageRank (o usa solo embeddings para análisis básico). "
            "**Paso 2**: Configura similitud mínima ~0.6 para enlaces muy relacionados, o ~0.4 para descubrimiento más "
            "amplio. **Paso 3**: Filtra por tipo de página (product→product, blog→blog) para consistencia. **Paso 4**: "
            "Prioriza enlaces desde páginas con PR >0.01 hacia páginas con PR <0.001 (flujo estratégico de autoridad). "
            "**Paso 5**: Implementa 5-10 enlaces internos por página para evitar dilución. **Paso 6**: Exporta "
            "sugerencias y valida manualmente que el contexto sea relevante antes de implementar. **Paso 7**: Re-analiza "
            "después de 30 días para medir cambios en distribución de PageRank. Ideal para rescatar contenido huérfano, "
            "potenciar product pages y construir topic authority."
        ),
    },
    "positions_report": {
        "title": "Informe de Posiciones SEO",
        "seo_theory": (
            "El tracking de posiciones es fundamental para medir el impacto de optimizaciones SEO. El análisis por "
            "familias de keywords permite detectar: (1) **Topic authority**: grupos de keywords rankean alto = "
            "Google reconoce tu expertise, (2) **Keyword gaps**: familias con posiciones bajas = oportunidades "
            "de optimización, (3) **SERP volatility**: cambios frecuentes indican competencia alta o contenido "
            "que no satisface intent. La evolución temporal de posiciones (líneas de tendencia) correlaciona con "
            "updates de algoritmo, acciones de competidores y tus optimizaciones. Keywords en posiciones 11-20 "
            "(página 2) son low-hanging fruit con máximo ROI de optimización - pequeñas mejoras generan tráfico "
            "significativo."
        ),
        "interpretation": (
            "La **tabla de posiciones** muestra keyword, posición actual y dominio ranking. **Familias de keywords** "
            "agrupan términos relacionados - observa: (1) Familias con posición promedio <10 son tus fortalezas, "
            "(2) Familias con posición 11-30 son quick wins, (3) Familias >30 necesitan estrategia profunda. "
            "**Featured snippets** (posición 0) generan CTR alto con bajo traffic - optimiza para question queries. "
            "**Dominios competidores** frecuentes en top 10 son benchmarks - analiza su contenido para identificar "
            "gaps. El **gráfico de evolución** muestra tendencias: subidas consistentes = optimizaciones efectivas, "
            "caídas súbitas = penalización o update de algoritmo, volatilidad = SERP inestable."
        ),
        "best_practices": (
            "**Paso 1**: Carga CSV de rank tracker (SEMrush, Ahrefs, SERPWatcher) con formato: Keyword | Position | URL. "
            "Soporta formato SERP (columnas Position 1, Position 2, etc.) o simple. **Paso 2**: Define familias de "
            "keywords usando patterns (ej: 'Familia Product: *producto*, *precio*, *comprar*'). **Paso 3**: Filtra "
            "keywords en posiciones 11-20 para identificar quick wins. **Paso 4**: Para cada familia débil, cruza con "
            "análisis de SDS - páginas con SDS bajo necesitan content rewrite, SDS alto necesitan más backlinks. "
            "**Paso 5**: Monitorea evolución semanal - cambios >5 posiciones requieren investigación (competidor nuevo, "
            "update, etc.). **Paso 6**: Exporta reporte con familias prioritarias para planning. Ideal para tracking "
            "de campañas, reporting a clientes y priorización de optimizaciones."
        ),
    },
    "embeddings_analysis": {
        "title": "Análisis de Embeddings Semánticos",
        "seo_theory": (
            "Los embeddings (representaciones vectoriales de texto) son la base de cómo Google entiende el lenguaje "
            "desde BERT (2019). Cada documento se convierte en un vector de alta dimensión (384-4096 dims) donde "
            "la distancia coseno refleja similitud semántica. Esto permite: (1) **Content similarity**: detectar "
            "contenido duplicado semántico (no solo textual), (2) **Topic modeling**: descubrir temas latentes, "
            "(3) **Search intent matching**: alinear contenido con queries de usuarios. La visualización t-SNE/UMAP "
            "reduce dimensiones para mostrar clusters naturales - documentos cercanos tienen intención de búsqueda "
            "similar. Modelos SOTA (BGE-M3, NV-Embed-v2) capturan matices que TF-IDF no puede, como sinónimos, "
            "contexto y relaciones semánticas complejas."
        ),
        "interpretation": (
            "El **gráfico t-SNE** muestra tus URLs como puntos en 2D - la proximidad espacial indica similitud "
            "semántica. **Clusters densos** = grupo de URLs sobre el mismo tema (verifica si es canibalización o "
            "topic cluster intencional). **URLs aisladas** = contenido único sin páginas relacionadas (oportunidad "
            "de expansión o contenido off-brand). **Colores** representan clusters automáticos o tipos de página. "
            "La **matriz de similitud** muestra pares de URLs con similitud >umbral - valores >0.8 indican "
            "canibalización probable, 0.6-0.8 = contenido relacionado (buen candidato para internal linking), "
            "<0.4 = sin relación semántica."
        ),
        "best_practices": (
            "**Paso 1**: Genera embeddings con modelos multilingües (all-MiniLM-L6-v2 para velocidad, BGE-M3 para "
            "precisión, NV-Embed-v2 para máxima calidad). **Paso 2**: Usa texto completo (no snippets) para embeddings "
            "precisos - mínimo 100 palabras por documento. **Paso 3**: Revisa clusters en t-SNE: densos requieren "
            "validación de canibalización, dispersos sugieren falta de topic authority. **Paso 4**: Exporta matriz "
            "de similitud y úsala para: (a) internal linking strategy, (b) identificar contenido a consolidar, "
            "(c) detectar gaps temáticos (áreas sin cobertura). **Paso 5**: Combina con análisis de performance: "
            "clusters con bajo tráfico necesitan optimización. Ideal para auditorías de arquitectura de información "
            "y detección de canibalización semántica."
        ),
    },
    "topic_modeling": {
        "title": "Modelado de Tópicos (LDA/NMF)",
        "seo_theory": (
            "El topic modeling descubre automáticamente temas latentes en tu contenido usando algoritmos LDA "
            "(Latent Dirichlet Allocation) o NMF (Non-negative Matrix Factorization). En SEO, esto permite: "
            "(1) **Topic coverage audit**: identificar qué temas cubres vs. competidores, (2) **Content gaps**: "
            "detectar subtemas ausentes en tu sitio, (3) **Topical authority**: medir profundidad de cobertura "
            "por tema. Google evalúa sites por topical authority - sitios que cubren un tema exhaustivamente "
            "(múltiples subtemas relacionados) rankean mejor que sitios dispersos. Cada tópico se representa por "
            "keywords clave - la distribución de tópicos por URL muestra si tu contenido está enfocado (1-2 tópicos "
            "dominantes) o disperso (muchos tópicos con peso similar)."
        ),
        "interpretation": (
            "La **tabla de tópicos** muestra cada tema descubierto con sus top keywords. Interpreta: (1) Keywords "
            "coherentes = tópico bien definido, (2) Keywords inconexas = ruido o tópico muy genérico. La **distribución "
            "de tópicos por URL** muestra qué porcentaje de cada documento pertenece a cada tema. **Documentos "
            "enfocados** tienen 1 tópico dominante (>60%), **documentos dispersos** tienen múltiples tópicos con "
            "peso similar (<30% cada uno) - señal de falta de enfoque. El **gráfico de tópicos** visualiza la "
            "distancia entre temas - tópicos cercanos son relacionados (buenos para topic clusters), tópicos "
            "lejanos son independientes."
        ),
        "best_practices": (
            "**Paso 1**: Usa corpus >50 documentos para topic modeling robusto (mínimo 30). **Paso 2**: Prueba "
            "diferentes números de tópicos (k): empieza con k=10, ajusta según coherencia de keywords. **Paso 3**: "
            "LDA para tópicos más coherentes y interpretables, NMF para tópicos más específicos y separables. "
            "**Paso 4**: Revisa tópicos descubiertos: (a) tópicos con muchos documentos son tus pilares de contenido, "
            "(b) tópicos con pocos documentos son gaps de oportunidad, (c) tópicos inesperados revelan contenido "
            "off-brand o secundario. **Paso 5**: Crea topic clusters: agrupa URLs del mismo tópico con pillar page "
            "central enlazando a subtemas. **Paso 6**: Compara tus tópicos vs. competidores para estrategia. Ideal "
            "para auditorías de contenido, planificación de content hubs y estrategia de topical authority."
        ),
    },
    "csv_upload": {
        "title": "Carga y Procesamiento de CSV",
        "seo_theory": (
            "La preparación correcta de datos es fundamental para análisis SEO precisos. Un CSV bien estructurado "
            "permite análisis a escala que serían imposibles manualmente. Columnas esenciales: (1) **URL**: "
            "identificador único de cada página, (2) **Texto/Contenido**: contenido completo para NER y embeddings, "
            "(3) **Embeddings**: representación vectorial pre-calculada (opcional), (4) **Metadata**: tipo de página, "
            "fecha de publicación, autor, etc. para segmentación. La calidad del input determina la calidad del "
            "análisis - garbage in, garbage out. Datos limpios y completos permiten identificar patterns que "
            "correlacionan con performance de ranking."
        ),
        "interpretation": (
            "Después de cargar el CSV, la **vista previa** muestra las primeras filas - verifica: (1) columnas "
            "detectadas correctamente, (2) tipos de datos apropiados (URL como texto, embeddings como vectores), "
            "(3) valores faltantes (NaN). El **resumen de columnas** indica: URLs únicas, valores nulos, longitud "
            "promedio de contenido. **Warnings** sobre columnas faltantes o formato incorrecto deben resolverse "
            "antes de análisis. La **detección automática** identifica columnas de URL, texto y embeddings - "
            "verifica que la asignación sea correcta."
        ),
        "best_practices": (
            "**Paso 1**: Exporta datos desde tu herramienta SEO (Screaming Frog, SEMrush, Ahrefs) o CMS. **Paso 2**: "
            "Asegura formato UTF-8 para caracteres especiales (tildes, ñ). **Paso 3**: Columna de texto: usa texto "
            "plano completo, no HTML - mínimo 100 palabras para análisis robusto. **Paso 4**: Si incluyes embeddings "
            "pre-calculados, usa formato array numpy o lista. **Paso 5**: Añade columnas de metadata útiles: "
            "'tipo_pagina', 'categoria', 'autor' para análisis segmentado. **Paso 6**: Elimina filas con URL "
            "duplicadas antes de cargar. **Paso 7**: Para sitios grandes (>10k URLs), segmenta el análisis por "
            "sección/categoria para performance. Ideal como primer paso para cualquier análisis SEO a escala."
        ),
    },
}


def get_help_content(section_key: str) -> HelpContent | None:
    """
    Obtiene el contenido de ayuda para una sección específica.

    Args:
        section_key: Clave de la sección (ej: "clustering", "knowledge_graph")

    Returns:
        Diccionario con contenido de ayuda o None si no existe
    """
    return HELP_SECTIONS.get(section_key)


def list_available_help_sections() -> list[str]:
    """
    Lista todas las secciones con ayuda disponible.

    Returns:
        Lista de claves de secciones
    """
    return list(HELP_SECTIONS.keys())


__all__ = [
    "HELP_SECTIONS",
    "HelpContent",
    "get_help_content",
    "list_available_help_sections",
]
