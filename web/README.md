# SCRIPTS Website

Static GitHub Pages site for the SCRIPTS paper.

**Live URL (after setup):** https://rladmstn1714.github.io/SCRIPTS/

## Local preview

```bash
python scripts/build_web_data.py
cd web
python -m http.server 8080
```

Open http://localhost:8080/

## GitHub Pages setup

In **Settings → Pages** of the GitHub repo, set **Source** to **GitHub Actions**.

The workflow `.github/workflows/pages.yml` builds `web/data/examples.json` from the dataset CSVs and deploys the `web/` directory.

## Dataset examples on the page

The **Dataset examples** section shows 3 English and 3 Korean dialogues inline. Each card is clickable to expand the dialogue and displays only **Highly likely** and **Unlikely** relation labels.

To change which scenes appear, edit `EXAMPLE_SCENE_IDS` in `scripts/build_web_data.py`.
