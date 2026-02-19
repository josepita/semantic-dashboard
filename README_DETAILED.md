# Embedding Insights Dashboard Suite

Suite de herramientas SEO construida con **Streamlit** para análisis semántico, embeddings, clustering, relaciones entre URLs, insights de GSC y generación de briefs.

## Qué incluye el proyecto

- **Hub principal**: `streamlit_app.py` (pantalla central con secciones en `app_sections/`).
- **Apps independientes** en `apps/`:
  - `content_analyzer`
  - `gsc_insights`
  - `linking_optimizer`
  - `fanout_analyzer`
  - `brief_generator`
- **Módulos compartidos**: `shared/` (gestión de proyectos, OAuth, clientes de APIs, utilidades).
- **API REST**: `api/app.py` con FastAPI.
- **Datos de trabajo**: `workspace/` (proyectos, cachés y artefactos locales).

## Requisitos (Linux / WSL)

- Python **3.10+** (recomendado 3.11/3.12)
- `pip`
- Git
- Conexión a internet para instalar dependencias/modelos
- API keys según funcionalidades usadas (OpenAI, Gemini, Serprobot)

## Setup rápido (copy/paste)

### Linux nativo

```bash
git clone <tu-repo>
cd Embeding-Dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tus API keys antes de continuar
python -m spacy download es_core_news_sm
python verify_setup.py
streamlit run streamlit_app.py
```

### WSL2 (Ubuntu)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
cd ~
git clone <tu-repo>
cd Embeding-Dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tus API keys antes de continuar
python -m spacy download es_core_news_sm
python verify_setup.py
streamlit run streamlit_app.py
```

## Configuración en Linux (nativo)

### 1) Clonar y entrar al repositorio

```bash
git clone <tu-repo>
cd Embeding-Dashboard
```

### 2) Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Opcional (si ejecutas apps por separado):
```bash
pip install -r apps/content_analyzer/requirements.txt
pip install -r apps/gsc_insights/requirements.txt
pip install -r apps/linking_optimizer/requirements.txt
```

### 4) Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y completa al menos:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `SERPROBOT_API_KEY` (si usas Brief Generator)

### 5) Instalar modelo spaCy

```bash
python -m spacy download es_core_news_sm
# o
python -m spacy download en_core_web_sm
```

### 6) Verificar instalación

```bash
python verify_setup.py
```

## Configuración en Linux con WSL

> Recomendado: usar **WSL2 + Ubuntu** y trabajar dentro del filesystem Linux (`~/...`), no en `/mnt/c/...` para mejor rendimiento.

### 1) Preparar WSL (una sola vez)

En Windows PowerShell (Administrador):
```powershell
wsl --install -d Ubuntu
```

Dentro de Ubuntu (WSL):
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### 2) Clonar y configurar proyecto dentro de WSL

```bash
cd ~
git clone <tu-repo>
cd Embeding-Dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 3) Instalar spaCy y verificar

```bash
python -m spacy download es_core_news_sm
python verify_setup.py
```

## Cómo lanzar el proyecto

### Opción A: Hub principal (recomendado)

```bash
streamlit run streamlit_app.py
```

URL habitual: `http://localhost:8501`

### Opción B: Apps individuales

```bash
streamlit run apps/content_analyzer/app.py
streamlit run apps/gsc_insights/app.py
streamlit run apps/linking_optimizer/app.py
streamlit run apps/fanout_analyzer/app.py
streamlit run apps/brief_generator/app.py
```

### Opción C: API REST (FastAPI)

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
```

Documentación interactiva: `http://localhost:8001/docs`

## Notas específicas para WSL

- Si no abre navegador automáticamente, copia la URL (`http://localhost:8501`) en tu navegador de Windows.
- Si necesitas exponerlo en red local: `streamlit run streamlit_app.py --server.address 0.0.0.0`.
- Evita ejecutar los `.bat`/`.vbs` en WSL; usa comandos `bash`.

## Estructura recomendada para desarrollo

```text
.
├── streamlit_app.py
├── app_sections/
├── apps/
├── shared/
├── api/
├── scripts/
└── workspace/
```

## Solución de problemas rápida

- **`python: command not found`**: usa `python3`.
- **Error de imports/paquetes**: confirma que `.venv` está activado.
- **`No module named spacy`**: instala spaCy + modelo.
- **Puerto ocupado**: `streamlit run streamlit_app.py --server.port 8502`.
- **Features con APIs no funcionan**: revisa claves en `.env` o configuración en UI.

## Seguridad y buenas prácticas

- No commits de secrets (`.env`, tokens OAuth, credenciales).
- Trata `workspace/` como datos sensibles de proyecto.
- Antes de subir cambios, valida manualmente los flujos tocados.

## Despliegue (resumen)

Existe workflow en `.github/workflows/deploy.yml` para test/build/deploy en ramas `main`/`master`. Para local (Linux/WSL), prioriza ejecución con Streamlit y validación de endpoints API.
