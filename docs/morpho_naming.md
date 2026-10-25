# Morphological cell-type naming

Use **morpho** for the morphological cell-type taxonomy and experiments using it.
This replaces the former contributor-based experiment name. Attribution to
Casey Schneider-Mizell remains in source provenance.

Local analysis, result, model, data and script paths now use `morpho`. Relative
symlinks preserve the previous paths, including saved checkpoint/report links.
The path mapping is in [morpho_naming_migration.json](morpho_naming_migration.json).
Notebook source, plot titles, table captions and model-card text use Morpho.
Existing raster plot outputs are historical renderings; rerunning their notebook
updates titles from the revised plotting code.

Frozen JSON/CSV/parquet/checkpoint contents are byte-preserved. Historical column
names, registry identifiers, hashes and external storage locations can therefore
still contain the old spelling. They are compatibility/provenance fields, not
the name to use for new analyses. No classifications, splits, training examples,
predictions or model weights change.

Use `MORPHO_FOLDS` for fold-selecting submission/summary scripts. The former
environment variable is accepted as a fallback. The root-matching command now
accepts `--morpho`; its former input flag remains an alias.
