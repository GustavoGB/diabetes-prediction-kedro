"""Raw modelling CSV -> master table, fitting every preprocessing artifact."""

from kedro.pipeline import Node, Pipeline

from ... import validation
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=nodes.clean_data,
                inputs=["raw_modelling_data", "params:columns"],
                outputs="cleaned_modelling_data",
                name="clean_data",
            ),
            Node(
                func=validation.validate_data,
                inputs=["cleaned_modelling_data", "params:data_quality.cleaned"],
                outputs=["validated_modelling_data", "modelling_data_quality"],
                name="validate_cleaned_modelling_data",
            ),
            Node(
                func=nodes.split_data,
                inputs=["validated_modelling_data", "params:columns", "params:split"],
                outputs="split_modelling_data",
                name="split_data",
            ),
            Node(
                func=nodes.fit_imputer,
                inputs=["split_modelling_data", "params:columns", "params:imputer"],
                outputs="imputer",
                name="fit_imputer",
            ),
            Node(
                func=nodes.apply_imputer,
                inputs=["split_modelling_data", "imputer"],
                outputs="imputed_modelling_data",
                name="apply_imputer",
            ),
            Node(
                func=nodes.fit_outlier_bounds,
                inputs=["imputed_modelling_data", "params:columns", "params:outliers"],
                outputs="outlier_bounds",
                name="fit_outlier_bounds",
            ),
            Node(
                func=nodes.apply_outlier_bounds,
                inputs=["imputed_modelling_data", "outlier_bounds"],
                outputs="primary_modelling_data",
                name="apply_outlier_bounds",
            ),
            Node(
                func=nodes.engineer_features,
                inputs=["primary_modelling_data", "params:feature_engineering"],
                outputs="featured_modelling_data",
                name="engineer_features",
            ),
            Node(
                func=nodes.fit_encoder,
                inputs=["featured_modelling_data", "params:columns"],
                outputs="encoder",
                name="fit_encoder",
            ),
            Node(
                func=nodes.apply_encoder,
                inputs=["featured_modelling_data", "encoder"],
                outputs="encoded_modelling_data",
                name="apply_encoder",
            ),
            Node(
                func=nodes.fit_scaler,
                inputs=["encoded_modelling_data", "params:columns"],
                outputs="scaler",
                name="fit_scaler",
            ),
            Node(
                func=nodes.apply_scaler,
                inputs=["encoded_modelling_data", "scaler"],
                outputs="scaled_modelling_data",
                name="apply_scaler",
            ),
            Node(
                func=validation.validate_data,
                inputs=["scaled_modelling_data", "params:data_quality.master_table"],
                outputs=["master_table", "master_table_quality"],
                name="validate_master_table",
            ),
        ]
    )
