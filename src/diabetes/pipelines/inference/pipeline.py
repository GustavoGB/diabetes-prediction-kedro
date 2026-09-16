"""Score unseen data with the promoted model.

Every transform node is the *same function object* the training pipeline used,
imported from data_engineering — that is the guarantee against skew.
"""

from kedro.pipeline import Node, Pipeline

from ... import validation
from ..data_engineering.nodes import (
    apply_encoder,
    apply_imputer,
    apply_outlier_bounds,
    apply_scaler,
    clean_data,
    engineer_features,
)
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=nodes.to_dataframe,
                inputs="raw_inference_data",
                outputs="raw_inference_dataframe",
                name="to_dataframe",
            ),
            Node(
                func=clean_data,
                inputs=["raw_inference_dataframe", "params:columns"],
                outputs="cleaned_inference_data",
                name="clean_inference_data",
            ),
            Node(
                func=validation.validate_data,
                inputs=["cleaned_inference_data", "params:data_quality.cleaned"],
                outputs=["validated_inference_data", "inference_data_quality"],
                name="validate_cleaned_inference_data",
            ),
            Node(
                func=apply_imputer,
                inputs=["validated_inference_data", "imputer"],
                outputs="imputed_inference_data",
                name="impute_inference_data",
            ),
            Node(
                func=apply_outlier_bounds,
                inputs=["imputed_inference_data", "outlier_bounds"],
                outputs="primary_inference_data",
                name="cap_inference_data",
            ),
            Node(
                func=engineer_features,
                inputs=["primary_inference_data", "params:feature_engineering"],
                outputs="featured_inference_data",
                name="engineer_inference_features",
            ),
            Node(
                func=apply_encoder,
                inputs=["featured_inference_data", "encoder"],
                outputs="encoded_inference_data",
                name="encode_inference_data",
            ),
            Node(
                func=apply_scaler,
                inputs=["encoded_inference_data", "scaler"],
                outputs="scaled_inference_data",
                name="scale_inference_data",
            ),
            Node(
                func=nodes.predict,
                inputs=[
                    "production_model",
                    "scaled_inference_data",
                    "params:inference",
                ],
                outputs="inference_predictions",
                name="predict",
            ),
        ]
    )
