# Data quality: Pydantic + Great Expectations

Why the project validates at two boundaries with two tools, what each one catches that the other structurally cannot, and how both are wired.

## Data quality: Pydantic + Great Expectations


The project validates at **two different boundaries**, with two different tools,
because they answer two different questions.

| | Pydantic | Great Expectations |
|---|---|---|
| question | *is this **record** well-formed?* | *is this **batch** fit to train or score on?* |
| unit | one row / one request | a whole dataframe |
| boundary | the HTTP edge (`api.py`) | the pipeline edge (`validation.py`) |
| sees | fields, types, per-field bounds | row counts, null rates, distributions, cardinality, medians |
| when | every request, synchronously | every `kedro run`, per checkpoint |
| failure | **422** to the caller | the run **stops**; a report lands in `08_reporting/` |
| cost | microseconds | ~10 ms per suite on 650 rows |
| declared in | `class Patient(BaseModel)` | `parameters.yml → data_quality` |

### Why neither can do the other's job

This is not a matter of preference — it is a **category difference**.

A Pydantic model validates one object. "48% of `Insulin` is missing", "the
positive class collapsed from 35% to 3%", "the feed returned 5 rows instead of
650" are all properties of a **collection**. There is no field you can annotate
to express them, because no single record is wrong. That is the entire blind
spot, and it is exactly where data pipelines fail in practice: not with garbage
rows, but with perfectly well-formed rows that no longer mean what they used to.

The reverse is just as true. Running a GX suite per HTTP request would be
nonsense: row-count and distribution expectations are undefined on a batch of
one, and you would pay ~10 ms and a validation context for a call that currently
takes 13 ms end to end. Pydantic costs microseconds and returns a precise
422 pointing at the offending field.

**So: Pydantic rejects a bad caller. Great Expectations rejects a bad dataset.**

### Demonstrated

Four realistic corruptions, run through both layers against the real inference
CSV. Reproduce it yourself:

```bash
uv run python scripts/data_quality_demo.py      # or: make quality
```

```
0. The real inference batch (baseline)
  Pydantic : 116/116 rows valid
  GX       : PASS (0/16 expectations failed)

1. Upstream job truncated the feed to 5 rows
  Pydantic : 5/5 rows valid                       <-- every row still perfect
  GX       : FAIL (4/16 expectations failed)
             - expect_table_row_count_to_be_between -> 5
             - expect_column_values_to_not_be_null(SKINTHICKNESS) -> 60.0% unexpected
             - expect_column_values_to_not_be_null(INSULIN) -> 80.0% unexpected
             - expect_column_mean_to_be_between(OUTCOME) -> 0.0

2. Insulin sensor stopped reporting (90% now unmeasured)
  Pydantic : 116/116 rows valid                   <-- 0 is a legal value!
  GX       : FAIL (1/16 expectations failed)
             - expect_column_values_to_not_be_null(INSULIN) -> 94.8% unexpected

3. Label pipeline broke: positives collapse to 2.6%
  Pydantic : 116/116 rows valid                   <-- 0 and 1 are both legal
  GX       : FAIL (1/16 expectations failed)
             - expect_column_mean_to_be_between(OUTCOME) -> 0.0258...

4. BMI arrives in the wrong unit (x10)
  Pydantic : 3/116 rows valid                     <-- here they overlap
  GX       : FAIL (1/16 expectations failed)
             - expect_column_values_to_be_between(BMI) -> 100.0% unexpected
```

Cases 1–3 are the argument. **Every single row is valid**, and the dataset is
ruined. A model trained on case 2 would quietly learn from a feature that is now
95% imputed; a model trained on case 3 would learn to always predict 0 and
report 97% accuracy. Both would pass every unit test in this repo and every
Pydantic check in the API.

Case 4 shows the deliberate overlap: bounds exist in both layers. That is
defence in depth, not duplication — the API stops one bad caller at the door,
the pipeline stops a bad upstream feed before it reaches the model.

### The asymmetry is on purpose

`Patient.Age` is `ge=21`; the GX suite allows `AGE >= 18`. That is not a drift
bug. **The API is strict about what it accepts; the pipeline is tolerant about
what history contains.** New requests must fall inside the range the model was
fitted on (the youngest training patient is 21, so a 19-year-old is an
extrapolation the service should refuse). Historical data is allowed to be
messier than what you accept today, because you cannot go back and re-collect
it. Postel's law, applied to a model boundary.

### What this would have caught in this very project

The notebook analysis turned up defects that dataset-level validation catches
directly:

- **The inference CSV has a different missingness profile** — `Glucose` has zero
  sentinel-zeros there, while the modelling CSV has 5. The notebook derived its
  "which columns are zero-coded" list from `min() == 0`, so it would silently
  treat `Glucose` differently on the two files. A null-rate expectation makes
  that visible instead of silent.
- **Schema drift**: replaying the notebook's `pd.get_dummies` chain on the
  inference CSV yields 24 columns, not 25. `ExpectTableColumnsToMatchSet` on the
  cleaned frame plus the fitted `OneHotEncoder` closes that hole from both ends.
- **Imputation silently not running**: the `master_table` suite asserts zero
  nulls in all five sentinel columns. If the imputer were bypassed, the run
  fails instead of scikit-learn raising something obscure ten nodes later.
- **Stratification regressing**: `ExpectColumnMeanToBeBetween(OUTCOME, 0.2, 0.5)`
  fails if the split stops preserving class balance.

### How it is wired here

Suites live in `conf/base/parameters.yml` as `{type, kwargs}` pairs and are
instantiated by name — the same config-over-code indirection the modelling
pipeline uses for `class_path`. Adding a check is a YAML edit.

```yaml
data_quality:
  cleaned:
    suite: cleaned_data
    fail_on_error: true
    expectations:
      - type: ExpectColumnValuesToNotBeNull
        kwargs: {column: INSULIN, mostly: 0.40}   # ~48% unmeasured is normal
      - type: ExpectColumnMeanToBeBetween
        kwargs: {column: OUTCOME, min_value: 0.20, max_value: 0.50}
```

Three checkpoints, 27 nodes total:

| checkpoint | suite | guards against |
|---|---|---|
| cleaned modelling data | `cleaned_data` | a bad training feed |
| `master_table` | `master_table` | the pipeline's own post-conditions |
| cleaned inference data | `cleaned_data` | **drift** — the same suite as training |

That third row is the one worth pausing on: **the inference batch is held to the
identical contract as the training data**. If next month's batch stops looking
like what the model was fitted on, the run fails loudly rather than producing
confident nonsense.

`validate_data` returns the dataframe *and* a report, so downstream nodes take
the validated frame as input. That dependency is what makes the check a **gate**
rather than a passive observation — a test (`test_validation_gates_downstream_work`)
asserts every validation node's output is actually consumed. Reports are written
to `data/08_reporting/*_quality.json`, and `fail_on_error: false` switches a
checkpoint from blocking to advisory if you would rather score and alert.

### An honest caveat on the choice

Great Expectations is not free: it adds ~60 MB and a sizeable dependency tree
(GX itself, `altair`, `cryptography`) to a project whose core is ~700 lines. For
a codebase this size, [**pandera**](https://pandera.readthedocs.io/) delivers
most of the same value at a fraction of the weight, and its schemas are
declarative classes that sit naturally beside Pydantic models.

GX earns its place here for three reasons: its expectation vocabulary is the
industry reference and reads as documentation; validation results are
first-class artifacts you can persist, diff and alert on; and the expectation
names themselves communicate intent to whoever inherits the pipeline. On a
larger project the calculus is clearer still. On a smaller one, reach for
pandera and keep the idea — the idea is the part that matters.

Two anti-patterns to avoid either way: do not run dataset validation inside the
request path, and do not assert the same bound in five places — pick the
boundary each rule belongs to and keep it there.

---

---

[← back to the README](../README.md)
