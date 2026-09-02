# Interactive figure publishing

Build and upload every standalone figure page with:

```bash
./scripts/publish_figures.sh --push
```

To preview without committing or pushing, omit `--push`, then open
`docs/figures/index.html`. Each notebook figure has its own page, and its
dropdowns redraw the Plotly chart entirely in the browser.

The GitHub Actions workflow deploys `docs/figures` after it is pushed. In the
repository's **Settings → Pages**, set **Source** to **GitHub Actions** once.
The gallery will then be available at:

```text
https://jcbliao.github.io/segclr_classifier/
```

Individual figure URLs shown in the gallery can be pasted into Notion with
`/embed`. The pages load Plotly from its CDN, so viewers need network access.
