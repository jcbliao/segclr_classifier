# Interactive figure publishing

Build and upload every standalone figure page with:

```bash
./scripts/publish_figures.sh --push
```

Both commands submit `scripts/sbatch/export_interactive_figures.sh`; figure
generation never runs on the login node. Without `--push`, wait for the Slurm
job to finish and then open `docs/figures/index.html`. With `--push`, the job
commits and pushes `docs/figures` only after a successful export.

Each notebook figure has its own page. Every dropdown state is rendered by the
notebook's actual Matplotlib plotting function and serialized with mpld3, so
figure dimensions, subplot geometry, colors, legends, labels, and annotations
come from the Matplotlib `Figure` rather than a separate Plotly recreation.

The GitHub Actions workflow deploys `docs/figures` after it is pushed. In the
repository's **Settings → Pages**, set **Source** to **GitHub Actions** once.
The gallery will then be available at:

```text
https://jcbliao.github.io/segclr_classifier/
```

Individual figure URLs shown in the gallery can be pasted into Notion with
`/embed`. The pages load mpld3 and D3 from their CDNs, so viewers need network
access.
