# Roadmap

## Documentation
- Emphasise that a local CRS should be chosen to minimise projection error; accuracy degrades over large areas (city level: good, country level: less accurate)

## Open items

### Correctness / precision
- Validate user inputs at public API boundary

### Performance
- Investigate line-based comparisons as alternative to bilateral point distance checks
- Benchmark super-cell approach (e.g. 3×3 macro-cells) to reduce dict lookups at fine grid sizes

