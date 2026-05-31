# Documentation ThermoPath

Cette documentation est construite avec Sphinx et le thème ReadTheDocs.

## Installation locale

```bash
python -m venv .venv-docs
.venv-docs\Scripts\activate
pip install -r docs/requirements.txt
```

Sous Linux ou macOS, l'activation devient :

```bash
source .venv-docs/bin/activate
```

## Build HTML

Depuis la racine du projet :

```bash
sphinx-build -b html docs docs/_build/html
```

La sortie HTML est générée dans `docs/_build/html`.
