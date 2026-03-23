# Raw Split-Aware CSV Thumbnail Explorer With Shared Movement Calculations

## Summary
- Load thumbnail source data directly from `/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example/cleaned_outliers/`.
- Keep split-aware browsing from each CSV's `set` column.
- Preserve the existing black-and-white thumbnail rendering and add optional per-fix variable coloring for markers and path segments.
- Reuse the repo's shared movement-feature pipeline so thumbnail movement variables stay aligned with anomaly/iSSF code.

## Key Changes
- App data source:
  - replace `Study` selection with direct CSV-file selection from the fixed cleaned-outliers folder
  - keep a split selector driven by unique `set` values, with `all` always available
- Shared thumbnail data prep:
  - load the CSV directly
  - parse timestamps and drop rows missing timestamp/lon/lat
  - sort by individual and time
  - preserve and filter by `set`
  - derive `step_dt_s`, `step_length_m`, and `speed_m_s` via shared helpers from `movebench.trajectories` and `movebench.movement_math`
- Thumbnail rendering:
  - keep `color_by=None` as the grayscale default
  - add RGB rendering for `color_by="<column>"`
  - support derived movement vars plus numeric `gps:*` columns
  - color markers by fix value and segments by adjacent-value summary, with neutral gray for missing values
  - normalize color values per CSV+split using 5th/95th percentile clipping and `viridis`
- Explorer behavior:
  - use the active render mode in the filmstrip, active thumbnail, grid browser, and PDF exports
  - keep map overlays class-colored; only thumbnail imagery changes with explanatory-variable coloring

## Test Plan
- Unit-test split-aware loading and `split=None` chunk boundaries.
- Unit-test movement-feature derivation through the shared helper path.
- Unit-test color-variable detection for derived movement vars, numeric `gps:*` vars, and exclusion of nonnumeric GPS columns.
- Unit-test grayscale vs RGB thumbnail rendering and NaN fallback behavior.
- Smoke-test app browsing for `train`, `test`, and `all`, plus PDF export in both render modes.

## Defaults
- Source folder is fixed to the cleaned-outliers directory above.
- No writes occur inside the source folder.
- No environmental covariates are downloaded or used.
- Per-fix coloring is the default colorization behavior.
