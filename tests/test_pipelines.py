"""Structural guarantees of the pipeline graph.

These tests assert the architecture rather than any numeric output: that the
inference pipeline reuses the training transforms, that it never fits anything,
and that a clean checkout can actually run.
"""

import importlib.util
from pathlib import Path

import yaml

from diabetes import validation
from diabetes.pipeline_registry import ALL, TRAINING
from diabetes.pipelines.data_engineering import nodes as de_nodes
from diabetes.pipelines.data_engineering import pipeline as de_pipeline
from diabetes.pipelines.inference import pipeline as inference_pipeline
from diabetes.pipelines.modelling import pipeline as modelling_pipeline

CONF = Path(__file__).resolve().parents[1] / "conf" / "base"
CATALOG = yaml.safe_load((CONF / "catalog.yml").read_text())
PARAMETERS = yaml.safe_load((CONF / "parameters.yml").read_text())

PIPELINES = {
    "data_engineering": de_pipeline.create_pipeline(),
    "modelling": modelling_pipeline.create_pipeline(),
    "inference": inference_pipeline.create_pipeline(),
}


def _inputs(*pipelines):
    """Every dataset any node reads. Pipeline.inputs() reports only free edges,
    which hides datasets produced and consumed inside the same pipeline."""
    return {name for p in pipelines for node in p.nodes for name in node.inputs}


def _outputs(*pipelines):
    return {name for p in pipelines for node in p.nodes for name in node.outputs}


def test_registry_names_match_the_pipeline_modules():
    assert set(ALL) == set(PIPELINES)
    assert set(TRAINING) == {"data_engineering", "modelling"}
    assert "inference" not in TRAINING, "a training run must not score as a side effect"


def test_every_node_is_uniquely_named():
    for name, pipeline in PIPELINES.items():
        node_names = [node.name for node in pipeline.nodes]
        assert len(node_names) == len(set(node_names)), f"duplicate node in {name}"


def test_inference_reuses_the_training_transforms():
    """Not 'an equivalent function' — the *same* function object. This is what
    makes training-serving skew structurally impossible."""
    inference_funcs = {node.func for node in PIPELINES["inference"].nodes}
    for shared in (
        de_nodes.clean_data,
        de_nodes.apply_imputer,
        de_nodes.apply_outlier_bounds,
        de_nodes.engineer_features,
        de_nodes.apply_encoder,
        de_nodes.apply_scaler,
    ):
        assert shared in inference_funcs, f"{shared.__name__} is not reused"


def test_inference_never_fits_an_artifact():
    """Scoring must only ever apply artifacts the training run produced."""
    fit_funcs = {
        de_nodes.fit_imputer,
        de_nodes.fit_outlier_bounds,
        de_nodes.fit_encoder,
        de_nodes.fit_scaler,
        de_nodes.split_data,
    }
    for node in PIPELINES["inference"].nodes:
        assert node.func not in fit_funcs, f"{node.name} fits during inference"


def test_every_fitted_artifact_is_persisted_and_reused():
    """Each fit_* output must be a catalog dataset and an inference input."""
    produced = _outputs(PIPELINES["data_engineering"])
    consumed = _inputs(PIPELINES["inference"])
    for artifact in ("imputer", "outlier_bounds", "encoder", "scaler"):
        assert artifact in produced, f"{artifact} is not produced by training"
        assert artifact in CATALOG, f"{artifact} is not persisted in the catalog"
        assert artifact in consumed, f"{artifact} is not reused at inference"


def test_everything_inference_loads_is_versioned_together():
    """Rollback is only coherent if the whole set rolls back together.

    Inference applies the encoder, scaler, imputer and fences that training
    fitted. Versioning the model alone would let a rollback pair an old
    estimator with new features — the failure is silent, because the shapes
    still line up. Kedro stamps one save_version per run, so versioning the set
    makes "the artifacts of run X" a directory, not a reassembly job.
    """
    loaded = _inputs(PIPELINES["inference"]) & set(CATALOG)
    fitted = {
        name
        for name in loaded
        if CATALOG[name]["filepath"].startswith("data/06_models/")
    }
    assert fitted == {
        "imputer",
        "outlier_bounds",
        "encoder",
        "scaler",
        "production_model",
    }

    unversioned = sorted(name for name in fitted if not CATALOG[name].get("versioned"))
    assert not unversioned, f"inference loads these unversioned: {unversioned}"


def test_inference_depends_on_the_promoted_model():
    assert "production_model" in _inputs(PIPELINES["inference"])
    assert "production_model" in _outputs(PIPELINES["modelling"])


def test_free_inputs_are_catalogued_or_parameterised():
    """A free input that is neither a catalog entry nor a parameter means a
    clean checkout cannot run the pipeline."""
    every = tuple(PIPELINES.values())
    for name in _inputs(*every) - _outputs(*every):
        if name.startswith("params:"):
            node = PARAMETERS
            for key in name.removeprefix("params:").split("."):
                assert key in node, f"{name} is undeclared"
                node = node[key]
        else:
            assert name in CATALOG, f"{name} is neither produced nor catalogued"


ARTIFACTS = frozenset(
    {"imputer", "outlier_bounds", "encoder", "scaler", "baseline_model", "tuned_model"}
)


def _layer(name):
    return (CATALOG[name].get("metadata") or {}).get("kedro-viz", {}).get("layer")


def test_persisted_datasets_declare_a_viz_layer():
    """kedro viz groups the graph by layer; an untagged dataset floats loose.

    The fitted transformers and the two candidates are the deliberate exception
    — see ``test_the_layer_graph_is_acyclic`` for why.
    """
    for name in CATALOG:
        if name in ARTIFACTS:
            assert not _layer(name), f"{name} must stay out of the layer graph"
            continue
        assert _layer(name), f"{name} has no kedro-viz layer"


def test_the_layer_graph_is_acyclic():
    """A cycle makes kedro viz disable layer bands for the WHOLE graph.

    Not a warning you notice: the app still renders, just flat, so the labels
    silently stop buying anything. The cycle is inherent to fit/apply — the
    encoder is fitted from feature data and then used to produce it — which is
    why fitted artifacts carry no layer at all.
    """
    edges, layers = set(), set()
    for pipeline in PIPELINES.values():
        for node in pipeline.nodes:
            for source in node.inputs:
                for target in node.outputs:
                    if source not in CATALOG or target not in CATALOG:
                        continue
                    a, b = _layer(source), _layer(target)
                    if a and b and a != b:
                        edges.add((a, b))
                        layers |= {a, b}

    # Kahn's algorithm: whatever cannot be ordered is in a cycle.
    remaining = set(layers)
    while True:
        free = {
            x for x in remaining if not any(b == x and a in remaining for a, b in edges)
        }
        if not free:
            break
        remaining -= free
    assert not remaining, f"layers are cyclic: {sorted(remaining)}"


def test_catalog_has_no_unused_entries():
    every = tuple(PIPELINES.values())
    used = _inputs(*every) | _outputs(*every)
    assert set(CATALOG) <= used, f"unused catalog entries: {set(CATALOG) - used}"


def test_both_data_branches_are_validated():
    """Training input and inference input must each pass a suite."""
    for name in ("data_engineering", "inference"):
        validators = [
            node
            for node in PIPELINES[name].nodes
            if node.func is validation.validate_data
        ]
        assert validators, f"{name} has no data quality gate"


def test_validation_gates_downstream_work():
    """A validation node must feed something. A terminal check that nobody
    consumes cannot stop bad data from reaching the model."""
    every = tuple(PIPELINES.values())
    all_nodes = [node for p in every for node in p.nodes]
    for node in all_nodes:
        if node.func is not validation.validate_data:
            continue
        frame_output = node.outputs[0]
        consumers = [n for n in all_nodes if frame_output in n.inputs]
        assert consumers, f"{node.name} produces {frame_output}, which nothing reads"


def test_quality_reports_are_persisted():
    for report in (
        "modelling_data_quality",
        "master_table_quality",
        "inference_data_quality",
    ):
        assert report in CATALOG, f"{report} is not persisted"


def test_the_committed_diagram_is_current():
    """docs/pipelines.md is generated; a stale diagram is worse than none.

    It is the only view of the graph a reader gets without starting kedro viz,
    so it has to be the graph the code actually builds.
    """
    spec = importlib.util.spec_from_file_location(
        "export_pipeline_diagram",
        Path(__file__).resolve().parents[1] / "scripts" / "export_pipeline_diagram.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.TARGET.read_text() == module.render()
    ), "docs/pipelines.md is out of date — run `make diagram`"
