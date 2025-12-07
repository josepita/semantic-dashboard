# Semantic Keyword Builder

Semantic Keyword Builder is an open-source Streamlit app that generates a semantically enriched keyword universe for SEO and content strategy using an Entity–Attribute–Variable (EAV) modeling approach.

The app takes basic project context (country, language, website, niche, and optional seed keywords) and returns a structured keyword dataset with intent, topic-level conceptual entities, attributes, variables, topics, and clusters. Results can be explored in a table and downloaded as CSV.

This project focuses on practical, scalable semantic SEO workflows, not on formal knowledge-graph engineering.

---
## Important Clarification on “Entities”
Although the internal prompt uses the term entity (for model compatibility), the app renames this field in the UI and CSV output to:

topic_entity_label

This field represents:

-A conceptual topic-level label for clustering
-Examples: service types, room types, problem categories, product groups, use cases

It does not represent:

-A resolved Google Knowledge Graph entity
-A Wikidata / DBpedia / ontology-based entity
-A formally disambiguated named entity

This distinction is intentional to avoid false semantic precision.

The tool currently performs:
✅ Conceptual semantic structuring
❌ Formal knowledge-graph entity resolution

The data model is designed to allow future KG enrichment as a separate layer.
---
## Features

- Interactive Streamlit interface
- Project configuration via sidebar:
  - Country
  - Language for keywords
  - Website / brand context
  - Niche / service
  - Approximate number of keywords
  - Optional seed keywords
- EAV-based keyword generation:
  - Entity (Topic-level conceptual entity labels), attribute, variable fields per keyword
- Search intent classification:
  - informational
  - commercial
  - transactional
  - navigational
  - mixed
- Qualitative search volume estimation:
  - high, medium, low, unknown
- Topic clustering with cluster IDs (C1, C2, etc.)
- Downloadable CSV export of the keyword builder
- Raw JSON output inspector for debugging

---

## How it works

1. You provide:
   - Country
   - Language for keywords
   - Website / brand context
   - Niche / service
   - Optional seed keywords
   - Minimum number of keywords to generate

2. The app builds a strict JSON-only prompt describing:
   - The niche and search context
   - Requirements for generating an EAV-based keyword builder
   - Required fields for each keyword (intent, volume_level, entity, attribute, variable, topic, cluster_id)

3. A large language model (accessed via the `google.generativeai` Python client) generates a JSON object with:
   - Project metadata
   - A list of keyword objects

4. The app:
   - Parses the JSON
   - Displays the keyword builder in a table
   - Lets you download the data as CSV
   - Shows the raw JSON for inspection

---

## Requirements

- Python 3.9 or higher (3.10+ recommended)
- A Google Generative AI API key (for example, for a Gemini-compatible model)
- The following Python packages:
  - streamlit
  - google-generativeai
  - pandas

You can install dependencies using a `requirements.txt` like:

```txt
streamlit
google-generativeai
pandas
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Getting started

1. Clone this repository:

```bash
git clone https://github.com/VitoKon/semantic-keyword-builder.git
cd semantic-keyword-builder
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Save the app code as `semantic_kb.py` (or another filename of your choice).

4. Run the Streamlit app:

```bash
streamlit run semantic_kb.py
```

5. Open the local URL that Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## Configuration in the app

In the Streamlit sidebar you will see the following fields:

- API key
  - Your Google Generative AI API key
- Model name
  - Default: `gemini-2.5-flash`
  - Can be changed to any compatible model name
- Country
  - Default: `United States`
- Language (for keywords)
  - Default: `English`
- Website / brand context
  - Example: `https://www.example.com/home-renovation`
- Niche / service
  - Example: `Service for home renovation`
- Approximate number of keywords to generate
  - Slider: 30 to 200 (default 60)
- Optional seed keywords (one per line)
  - Example:
    - `home renovation`
    - `apartment renovation`
    - `kitchen remodel`
    - `bathroom renovation`

Click the button

- Generate Keyword Builder

to start the generation.

---

## Output

The app returns:

1. Keyword builder table (in the UI)
   - Columns (when available):
     - keyword
     - volume_level
     - intent
     - topic_entity_label (conceptual, not knowledge-graph)
     - attribute
     - variable
     - source
     - topic
     - cluster_id
     - any extra fields included by the model

2. CSV export
   - Download button: `semantic_keyword_builder.csv`

3. Project metadata
   - Shown as a JSON block under “Project metadata”

4. Raw JSON output
   - Available inside an expandable section
   - Useful for debugging parsing issues or refining the prompt

---

## JSON structure expected from the model

The app prompts the model to return only a JSON object with the following structure:

```json
{
  "project": {
    "country": "United States",
    "language": "English",
    "website": "https://www.example.com/home-renovation",
    "niche": "Service for home renovation",
    "approx_keywords_requested": 60
  },
  "keywords": [
    {
      "keyword": "example keyword in English",
      "volume_level": "high | medium | low | unknown",
      "intent": "informational | commercial | transactional | navigational | mixed",
      "entity": "main entity for this keyword in English",
      "attribute": "attribute in English",
      "variable": "variable value in English",
      "source": "ai_synthetic | seed | competitor | site_search | other",
      "topic": "human-friendly cluster name in English",
      "cluster_id": "C1"
    }
  ]
}
```

Constraints enforced by the prompt:

- The `keywords` array must contain at least the requested number of keyword objects.
- `intent` must be one of:
  - informational
  - commercial
  - transactional
  - navigational
  - mixed
- `volume_level` must be one of:
  - high
  - medium
  - low
  - unknown
- `cluster_id` is a short code (for example `C1`, `C2`) reused across keywords in the same topic.

---

## Error handling

The app includes basic error handling:

- If the API key is missing, the app stops and shows an error message.
- If the model fails to initialize, an error is shown.
- If the response cannot be parsed as valid JSON, the app:
  - Shows a JSON parsing error
  - Displays the raw response for debugging
- If no keywords are returned in the `keywords` array, the app:
  - Shows a warning
  - Lets you inspect the raw JSON response

---

## Customization ideas

You can easily extend or customize the app:

- Change defaults for country, language, niche, and seed keywords
- Add additional fields for:
  - Persona
  - Funnel stage
  - Content format (blog, video, landing page)
- Adjust the prompt to:
  - Enforce custom naming conventions for topics or clusters
  - Include additional metadata per keyword
- Integrate external keyword metrics (for example, from SEO tools) after generation

---

## Contributing

Contributions are welcome. Ideas for improvement:

- Support for multiple model providers
- Persistent storage (for example, saving runs to a database)
- Additional visualizations (intent distribution, topic maps)
- Integration with search console or analytics exports

Feel free to open issues or pull requests.


