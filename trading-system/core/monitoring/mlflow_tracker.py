"""MLflowTracker — clean wrapper around MLflow for experiment and model registry management."""

from __future__ import annotations

import os
from typing import Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------

try:
    import mlflow
    import mlflow.exceptions
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "MLflowTracker: mlflow is not installed — tracker will operate in no-op mode. "
        "Install with: pip install mlflow"
    )


# ---------------------------------------------------------------------------
# MLflowTracker
# ---------------------------------------------------------------------------


class MLflowTracker:
    """Clean wrapper around MLflow for experiment tracking and model registry.

    When mlflow is not installed (``ImportError`` at import time) the tracker
    degrades gracefully: all methods become no-ops and return safe defaults.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "trading_system",
    ) -> None:
        """Initialise the MLflow tracking URI and experiment.

        The tracking URI is resolved in this order:
        1. *tracking_uri* argument (if provided).
        2. ``MLFLOW_TRACKING_URI`` environment variable.
        3. MLflow's default (local ``./mlruns`` directory).

        Args:
            tracking_uri:    Optional MLflow server URI.
            experiment_name: Experiment to create or retrieve.  Defaults to
                             ``"trading_system"``.
        """
        self._available: bool = _MLFLOW_AVAILABLE
        self._experiment_name: str = experiment_name
        self._experiment_id: Optional[str] = None

        if not self._available:
            logger.warning(
                "MLflowTracker: mlflow unavailable — all tracking calls will be skipped"
            )
            return

        # Set tracking URI
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
            logger.info("MLflowTracker: tracking URI = %s", uri)
        else:
            logger.info(
                "MLflowTracker: no tracking URI set — using MLflow default (./mlruns)"
            )

        # Create or retrieve experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(
                    "MLflowTracker: created experiment '%s' (id=%s)",
                    experiment_name,
                    self._experiment_id,
                )
            else:
                self._experiment_id = experiment.experiment_id
                logger.info(
                    "MLflowTracker: using existing experiment '%s' (id=%s)",
                    experiment_name,
                    self._experiment_id,
                )
        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.__init__: failed to create/get experiment '%s' — %s",
                experiment_name,
                exc,
            )
            self._available = False

    # ------------------------------------------------------------------
    # Run tracking
    # ------------------------------------------------------------------

    def log_training_run(
        self,
        model_type: str,
        params: dict,
        metrics: dict,
        artifacts: Optional[list] = None,
    ) -> str:
        """Record a training run with parameters, metrics, and artifact files.

        Args:
            model_type: Short label for the model type, e.g. ``"xgb_regime"``.
            params:     Dict of hyper-parameters to log.
            metrics:    Dict of evaluation metrics to log.
            artifacts:  Optional list of local file paths to upload as artifacts.

        Returns:
            The MLflow ``run_id`` string, or an empty string when unavailable.
        """
        if not self._available:
            return ""

        run_id = ""
        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=model_type,
            ) as run:
                run_id = run.info.run_id

                mlflow.log_param("model_type", model_type)
                for key, value in params.items():
                    mlflow.log_param(key, value)

                for key, value in metrics.items():
                    mlflow.log_metric(key, float(value))

                if artifacts:
                    for artifact_path in artifacts:
                        try:
                            mlflow.log_artifact(artifact_path)
                            logger.debug(
                                "MLflowTracker.log_training_run: logged artifact %s",
                                artifact_path,
                            )
                        except mlflow.exceptions.MlflowException as exc:
                            logger.warning(
                                "MLflowTracker.log_training_run: could not log artifact "
                                "'%s' — %s",
                                artifact_path,
                                exc,
                            )

            logger.info(
                "MLflowTracker.log_training_run: run_id=%s model_type=%s params=%d metrics=%d",
                run_id,
                model_type,
                len(params),
                len(metrics),
            )
        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.log_training_run failed (model_type=%s): %s",
                model_type,
                exc,
            )

        return run_id

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(
        self, run_id: str, name: str, stage: str = "STAGING"
    ) -> None:
        """Register a model artifact from a completed run into the MLflow Model Registry.

        Args:
            run_id: MLflow run ID that produced the model.
            name:   Registered model name in the registry.
            stage:  Initial stage — one of ``"STAGING"``, ``"CHAMPION"``,
                    or ``"ARCHIVED"``.
        """
        if not self._available or not run_id:
            return

        try:
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri=model_uri, name=name)

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=name,
                version=model_version.version,
                stage=stage,
            )
            logger.info(
                "MLflowTracker.register_model: name=%s version=%s stage=%s",
                name,
                model_version.version,
                stage,
            )
        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.register_model failed (run_id=%s name=%s): %s",
                run_id,
                name,
                exc,
            )

    def promote_model(self, name: str, version: int, new_stage: str) -> None:
        """Transition a registered model version to a new lifecycle stage.

        Args:
            name:      Registered model name.
            version:   Version number to promote.
            new_stage: Target stage — ``"STAGING"``, ``"CHAMPION"``, or ``"ARCHIVED"``.
        """
        if not self._available:
            return

        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage=new_stage,
            )
            logger.info(
                "MLflowTracker.promote_model: name=%s version=%d → %s",
                name,
                version,
                new_stage,
            )
        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.promote_model failed (name=%s version=%d): %s",
                name,
                version,
                exc,
            )

    def get_champion(self, name: str) -> dict:
        """Retrieve metadata for the current CHAMPION model version.

        Args:
            name: Registered model name.

        Returns:
            Dict with keys ``"run_id"``, ``"version"``, ``"uri"``.
            Returns an empty dict when no CHAMPION version exists.
        """
        if not self._available:
            return {}

        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(name, stages=["CHAMPION"])
            if not versions:
                logger.info(
                    "MLflowTracker.get_champion: no CHAMPION version for '%s'", name
                )
                return {}

            v = versions[0]
            result = {
                "run_id": v.run_id,
                "version": int(v.version),
                "uri": f"models:/{name}/CHAMPION",
            }
            logger.info(
                "MLflowTracker.get_champion: name=%s version=%s run_id=%s",
                name,
                v.version,
                v.run_id,
            )
            return result
        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.get_champion failed (name=%s): %s", name, exc
            )
            return {}

    def rollback(self, name: str, db_manager=None) -> bool:
        """Roll back to the most recent ARCHIVED model version.

        Promotes the most recently archived version to CHAMPION and demotes
        the current CHAMPION to ARCHIVED.

        Args:
            name:       Registered model name.
            db_manager: Optional db_manager for logging the rollback event to
                        ``system_events``.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        if not self._available:
            return False

        try:
            client = mlflow.tracking.MlflowClient()

            # Find current CHAMPION
            champion_versions = client.get_latest_versions(name, stages=["CHAMPION"])
            current_champion = champion_versions[0] if champion_versions else None

            # Find most recent ARCHIVED version
            archived_versions = client.get_latest_versions(name, stages=["ARCHIVED"])
            if not archived_versions:
                logger.warning(
                    "MLflowTracker.rollback: no ARCHIVED versions found for '%s'", name
                )
                return False

            # Pick the ARCHIVED version with the highest version number
            restore_version = max(archived_versions, key=lambda v: int(v.version))

            # Promote archived → CHAMPION
            client.transition_model_version_stage(
                name=name,
                version=restore_version.version,
                stage="CHAMPION",
            )
            logger.info(
                "MLflowTracker.rollback: promoted version=%s to CHAMPION for '%s'",
                restore_version.version,
                name,
            )

            # Archive current CHAMPION (if it exists and is different)
            if current_champion and current_champion.version != restore_version.version:
                client.transition_model_version_stage(
                    name=name,
                    version=current_champion.version,
                    stage="ARCHIVED",
                )
                logger.info(
                    "MLflowTracker.rollback: archived previous CHAMPION version=%s for '%s'",
                    current_champion.version,
                    name,
                )

            # Persist rollback event
            if db_manager is not None:
                try:
                    db_manager.log_system_event(
                        event_type="MODEL_ROLLBACK",
                        severity="WARNING",
                        message=(
                            f"Model '{name}' rolled back from version "
                            f"{current_champion.version if current_champion else 'unknown'} "
                            f"to version {restore_version.version}"
                        ),
                        details={
                            "model_name": name,
                            "restored_version": int(restore_version.version),
                            "previous_champion": (
                                int(current_champion.version)
                                if current_champion
                                else None
                            ),
                        },
                    )
                except Exception as db_exc:
                    logger.error(
                        "MLflowTracker.rollback: failed to log system event — %s", db_exc
                    )

            return True

        except mlflow.exceptions.MlflowException as exc:
            logger.error(
                "MLflowTracker.rollback failed (name=%s): %s", name, exc
            )
            return False
