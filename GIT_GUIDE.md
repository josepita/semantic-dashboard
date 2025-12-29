# Control de Versiones con Git - Guía Rápida

## ✅ Configuración Completada

Tu repositorio Git está listo y funcionando. Se han realizado dos commits iniciales:

```bash
d35a015 Update .gitignore: exclude virtual environment folders
766bd2a Initial commit: Embedding Dashboard with knowledge graph improvements
```

## 📝 Comandos Git Básicos

### Ver el estado actual
```bash
git status
```

### Añadir cambios al staging area
```bash
# Añadir todos los archivos modificados
git add .

# Añadir un archivo específico
git add nombre_archivo.py
```

### Hacer un commit
```bash
git commit -m "Descripción del cambio"
```

### Ver el historial
```bash
# Ver las últimas 10 commits
git log --oneline -10

# Ver cambios detallados
git log -p
```

### Crear una rama nueva
```bash
# Crear y cambiar a una nueva rama
git checkout -b nombre-rama

# O en versiones recientes
git switch -c nombre-rama
```

### Ver diferencias
```bash
# Ver cambios no añadidos
git diff

# Ver cambios en staging
git diff --staged
```

### Deshacer cambios
```bash
# Descartar cambios en un archivo
git checkout -- nombre_archivo.py

# Deshacer el último commit (manteniendo cambios)
git reset --soft HEAD~1

# Deshacer el último commit (descartando cambios)
git reset --hard HEAD~1
```

## 📦 Archivos Excluidos (.gitignore)

El `.gitignore` está configurado para excluir:

- ✅ Entornos virtuales (`.venv/`, `.venv311/`, `.venv312/`)
- ✅ Archivos Python compilados (`__pycache__/`, `*.pyc`)
- ✅ Archivos de configuración IDE (`.vscode/`, `.idea/`)
- ✅ Secretos de Streamlit (`.streamlit/secrets.toml`)
- ✅ Variables de entorno (`.env`)

## 🚀 Workflow Recomendado

### 1. Antes de empezar a trabajar
```bash
git status  # Verificar que no hay cambios pendientes
```

### 2. Trabajar en una característica nueva
```bash
git checkout -b feature/nueva-funcionalidad
# ... hacer cambios ...
git add .
git commit -m "Añadir nueva funcionalidad X"
```

### 3. Volver a la rama principal
```bash
git checkout master
git  merge feature/nueva-funcionalidad
```

### 4. Hacer commit regularmente
- Commit después de cada funcionalidad completa
- Mensajes descriptivos: "Añadir filtro de entidades" mejor que "Cambios"

## 📋 Ejemplos de Mensajes de Commit

**Buenos:**
- `feat: añadir whitelist manual para entidades`
- `fix: corregir error en cálculo de PageRank`
- `refactor: mejorar estructura de función generate_knowledge_graph`
- `docs: actualizar README con instrucciones de instalación`

**Malos:**
- `cambios`
- `fix`
- `update file`

## 🔧 Configuración Actual

```bash
Usuario: jdiaz
Email: jdiaz@local
Rama principal: master
```

## ⚠️ Importante

- Los entornos virtuales (.venv311, .venv312) ya están excluidos del control de versiones
- Las dependencias están en `requirements.txt`
- Para recrear el entorno: `pip install -r requirements.txt`

## 🌐 Conectar a GitHub (Opcional)

Si quieres subir esto a GitHub:

```bash
# 1. Crear repositorio en GitHub (sin README)
# 2. Añadir remote
git remote add origin https://github.com/tu-usuario/EmbeddingDashboard.git

# 3. Push inicial
git push -u origin master
```

## 📞 Ayuda

Si necesitas ayuda con Git:
- `git help <comando>` - Ayuda para un comando específico
- `git --help` - Ayuda general
