# 🚀 Cómo Crear Accesos Directos en el Escritorio

## 📁 Archivos Launcher Creados

Se han creado 4 launchers (archivos `.vbs`) que inician las aplicaciones **sin mostrar la ventana de consola**:

1. **Launcher - Dashboard Principal.vbs** → Inicia el dashboard principal completo
2. **Launcher - Content Analyzer.vbs** → Inicia solo Content Analyzer
3. **Launcher - Linking Optimizer.vbs** → Inicia solo Linking Optimizer
4. **Launcher - GSC Insights.vbs** → Inicia solo GSC Insights

### Versión alternativa (Minimizado)
- **Launcher - Dashboard Principal (Minimizado).vbs** → Muestra consola minimizada en barra de tareas

---

## 🖱️ Opción 1: Uso Directo (Más Simple)

**Simplemente haz doble clic en cualquier archivo `.vbs`** y la aplicación se iniciará automáticamente sin mostrar consola.

✅ **Ventajas:**
- Sin pasos adicionales
- Funciona inmediatamente

❌ **Desventajas:**
- Los archivos están en la carpeta del proyecto
- No tienen icono personalizado

---

## 🎨 Opción 2: Crear Accesos Directos en el Escritorio (Recomendado)

### Paso 1: Crear el acceso directo

1. **Clic derecho** en el archivo `.vbs` que quieras (ej: `Launcher - Dashboard Principal.vbs`)
2. Selecciona **"Enviar a" → "Escritorio (crear acceso directo)"**
3. Se creará un acceso directo en tu escritorio

### Paso 2: Renombrar (opcional)

1. Clic derecho en el acceso directo → **"Cambiar nombre"**
2. Ponle un nombre corto, ej: `SEO Dashboard`, `Content Analyzer`, etc.

### Paso 3: Cambiar icono (opcional)

1. **Clic derecho** en el acceso directo → **"Propiedades"**
2. En la pestaña **"Acceso directo"**, clic en **"Cambiar icono..."**
3. Opciones:
   - **Iconos del sistema:** Clic en "Examinar" y navega a `C:\Windows\System32\shell32.dll`
   - **Iconos personalizados:** Descarga un `.ico` de internet y selecciónalo
4. Selecciona el icono que prefieras
5. Clic en **"Aceptar"** y **"Aplicar"**

### Sugerencias de iconos del sistema (shell32.dll)

- **Icono 14:** Globo terráqueo (ideal para dashboard web)
- **Icono 21:** Monitor con carpeta (ideal para análisis)
- **Icono 44:** Gráfico con flecha (ideal para insights)
- **Icono 165:** Carpeta con lupa (ideal para analyzer)
- **Icono 220:** Estrella dorada (ideal para optimizador)

---

## 🎯 Opción 3: Anclar a la Barra de Tareas

1. Crea el acceso directo en el escritorio (Opción 2)
2. **Clic derecho** en el acceso directo
3. Selecciona **"Anclar a la barra de tareas"**
4. Ahora puedes iniciar la app con un solo clic desde la barra de tareas

---

## 🎯 Opción 4: Anclar al Menú Inicio

1. Crea el acceso directo en el escritorio (Opción 2)
2. **Clic derecho** en el acceso directo
3. Selecciona **"Anclar a Inicio"**
4. Aparecerá en el menú inicio de Windows

---

## ⚙️ Diferencia entre versiones

| Versión | Ventana de Consola | Cuándo Usar |
|---------|-------------------|-------------|
| **Normal** (Launcher - XXX.vbs) | Completamente oculta | Uso cotidiano, interfaz limpia |
| **Minimizado** (Launcher - XXX (Minimizado).vbs) | Minimizada en barra de tareas | Debug, ver mensajes de error |

---

## 🔧 Personalización Avanzada

### Cambiar comportamiento del launcher

Abre el archivo `.vbs` con un editor de texto (clic derecho → Editar) y modifica el número en `WshShell.Run`:

```vbscript
WshShell.Run "start_streamlit.bat", 0, False
'                                    ^
'                                    |
'         0 = Oculto completamente
'         1 = Normal (con ventana)
'         7 = Minimizado
```

### Crear launcher con icono embebido (Avanzado)

Si quieres un ejecutable `.exe` con icono embebido:

1. Usa herramientas como **Bat To Exe Converter** o **Advanced BAT to EXE Converter**
2. Convierte el archivo `.bat` a `.exe`
3. Asigna un icono `.ico` durante la conversión

---

## 🌐 Acceso desde otros equipos

Si quieres acceder desde otros equipos en la red local:

1. Ejecuta el launcher en el equipo servidor
2. La aplicación mostrará una URL tipo: `http://localhost:8501`
3. Encuentra tu IP local: `ipconfig` en cmd → busca "IPv4"
4. Accede desde otro equipo: `http://[TU_IP]:8501`

---

## 📌 Resumen Rápido

1. ✅ **Más simple:** Doble clic en archivos `.vbs`
2. ✨ **Recomendado:** Crear acceso directo en escritorio con icono
3. 🚀 **Más rápido:** Anclar a barra de tareas
4. 📱 **Más accesible:** Anclar a menú inicio

---

## ❓ Solución de Problemas

**Problema:** Al hacer doble clic no pasa nada

**Solución 1:** Verifica que el entorno virtual `.venv` existe en la carpeta raíz del proyecto

**Solución 2:** Abre el `.bat` directamente para ver mensajes de error

**Solución 3:** Verifica que Python está instalado: `py --version` en cmd

---

**Problema:** Windows bloquea la ejecución del .vbs

**Solución:**
1. Clic derecho en el archivo .vbs → Propiedades
2. Marca "Desbloquear" en la parte inferior
3. Clic en Aplicar

---

**Problema:** Quiero cerrar la aplicación

**Solución:**
- Si usas versión oculta: Busca `streamlit` en el Administrador de Tareas y terminar proceso
- Si usas versión minimizada: Restaura la ventana y presiona `Ctrl+C`
- O simplemente cierra la pestaña del navegador y el proceso se cerrará automáticamente tras inactividad
