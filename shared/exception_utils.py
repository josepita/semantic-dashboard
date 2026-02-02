"""
Utilidades para manejo mejorado de excepciones.

Este módulo proporciona clases de excepción personalizadas y helpers
para mejorar el manejo de errores en las aplicaciones.
"""

import functools
from typing import Optional, Type
import streamlit as st


# Excepciones personalizadas
class ProjectError(Exception):
    """Error relacionado con operaciones de proyecto."""
    pass


class ProjectNotFoundError(ProjectError):
    """Proyecto no encontrado."""
    pass


class ProjectLoadError(ProjectError):
    """Error al cargar proyecto."""
    pass


class ProjectCreateError(ProjectError):
    """Error al crear proyecto."""
    pass


class DataValidationError(Exception):
    """Error de validación de datos."""
    pass


class FileProcessingError(Exception):
    """Error al procesar archivo."""
    pass


class APIError(Exception):
    """Error relacionado con APIs externas."""
    pass


class ConfigurationError(Exception):
    """Error de configuración."""
    pass


# Helpers para manejo de excepciones
def handle_exception(
    exception: Exception,
    user_message: str,
    show_details: bool = True,
    log_error: bool = True
) -> None:
    """
    Maneja una excepción mostrando mensaje al usuario.
    
    Args:
        exception: Excepción capturada
        user_message: Mensaje amigable para el usuario
        show_details: Si mostrar detalles técnicos del error
        log_error: Si registrar el error en logs
    """
    if log_error:
        # TODO: Implementar logging real
        print(f"ERROR: {exception.__class__.__name__}: {str(exception)}")
    
    if show_details:
        st.error(f"{user_message}\n\n**Detalles**: {str(exception)}")
    else:
        st.error(user_message)


def safe_execute(
    func,
    *args,
    error_message: str = "Error al ejecutar operación",
    default_return=None,
    show_error: bool = True,
    **kwargs
):
    """
    Ejecuta una función de forma segura capturando excepciones.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales para la función
        error_message: Mensaje de error a mostrar
        default_return: Valor a retornar en caso de error
        show_error: Si mostrar el error al usuario
        **kwargs: Argumentos nombrados para la función
        
    Returns:
        Resultado de la función o default_return en caso de error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if show_error:
            handle_exception(e, error_message)
        return default_return


def validate_file_upload(
    uploaded_file,
    allowed_extensions: Optional[list] = None,
    max_size_mb: Optional[float] = None
) -> None:
    """
    Valida un archivo subido.
    
    Args:
        uploaded_file: Archivo subido por Streamlit
        allowed_extensions: Lista de extensiones permitidas (ej: ['csv', 'xlsx'])
        max_size_mb: Tamaño máximo en MB
        
    Raises:
        FileProcessingError: Si el archivo no es válido
    """
    if uploaded_file is None:
        raise FileProcessingError("No se ha subido ningún archivo")
    
    # Validar extensión
    if allowed_extensions:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            raise FileProcessingError(
                f"Extensión de archivo no permitida. "
                f"Permitidas: {', '.join(allowed_extensions)}"
            )
    
    # Validar tamaño
    if max_size_mb:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise FileProcessingError(
                f"Archivo demasiado grande ({file_size_mb:.1f} MB). "
                f"Máximo permitido: {max_size_mb} MB"
            )


def validate_dataframe(
    df,
    required_columns: Optional[list] = None,
    min_rows: Optional[int] = None
) -> None:
    """
    Valida un DataFrame.
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas
        min_rows: Número mínimo de filas
        
    Raises:
        DataValidationError: Si el DataFrame no es válido
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("El objeto no es un DataFrame válido")
    
    if df.empty:
        raise DataValidationError("El DataFrame está vacío")
    
    # Validar columnas requeridas
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Faltan columnas requeridas: {', '.join(missing_cols)}"
            )
    
    # Validar número mínimo de filas
    if min_rows and len(df) < min_rows:
        raise DataValidationError(
            f"El DataFrame tiene {len(df)} filas. "
            f"Se requieren al menos {min_rows} filas"
        )


def get_exception_message(exception: Exception) -> str:
    """
    Obtiene un mensaje amigable para una excepción.
    
    Args:
        exception: Excepción a procesar
        
    Returns:
        Mensaje amigable para el usuario
    """
    exception_messages = {
        FileNotFoundError: "Archivo no encontrado",
        PermissionError: "Sin permisos para acceder al archivo",
        ValueError: "Valor inválido",
        KeyError: "Clave no encontrada",
        ImportError: "Módulo no instalado",
        ConnectionError: "Error de conexión",
        TimeoutError: "Tiempo de espera agotado",
    }
    
    exception_type = type(exception)
    base_message = exception_messages.get(exception_type, "Error desconocido")
    
    return f"{base_message}: {str(exception)}"


class ExceptionContext:
    """
    Contexto para manejo de excepciones.
    
    Ejemplo:
        with ExceptionContext("Error al cargar datos"):
            # código que puede fallar
            df = pd.read_csv(file)
    """
    
    def __init__(
        self,
        user_message: str,
        show_details: bool = True,
        reraise: bool = False
    ):
        """
        Args:
            user_message: Mensaje para el usuario
            show_details: Si mostrar detalles del error
            reraise: Si re-lanzar la excepción después de manejarla
        """
        self.user_message = user_message
        self.show_details = show_details
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            handle_exception(exc_val, self.user_message, self.show_details)
            return not self.reraise  # Suprimir excepción si reraise=False
        return True


# Decorador para manejo de excepciones
def handle_errors(
    user_message: str = "Error al ejecutar la función",
    show_details: bool = True,
    default_return=None
):
    """
    Decorador para manejar excepciones en funciones.
    
    Ejemplo:
        @handle_errors("Error al procesar datos")
        def process_data(df):
            # código que puede fallar
            return df.groupby('col').sum()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_exception(e, user_message, show_details)
                return default_return
        return wrapper
    return decorator
