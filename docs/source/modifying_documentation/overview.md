# Overview

## Background

- HTML Documentation is generated using <a href="https://www.sphinx-doc.org/en/master/" target="_blank">Sphinx</a>
- Theme: [Furo](https://pradyunsg.me/furo/reference/)
  - Here you find information about the reST or Markdown syntax processed by the theme
- Extensions:
  - Support for <a href="https://www.markdownguide.org/" target="_blank">Markdown (.md)</a> and <a href="https://rstdoc.readthedocs.io/en/latest/readme.html" target="_blank">ReStructured Text (.rst)</a> using [Markedly Structured Text (MyST)](https://myst-parser.readthedocs.io/en/latest/intro.html)
  - Support for <a href="https://mermaid.js.org/" target="_blank">Mermaid Diagrams</a>

## Installing Sphinx Dependencies

If using the .devcontainer, this should already be done. However, you can do it manually by installing the `docs` extra:

```bash
# From the root folder of the repo
pip install -e ".[docs]"
```

## Building the Documentation

```bash
# Watch /docs/source folder and rebuild the documentation on file change
make watch
```

This will output to the `/docs/build` folder

## Viewing the Documentation

```bash
# Start server to view the documentation
make serve
```

### Open the URL in command output

http://127.0.0.1:7878/index.html

