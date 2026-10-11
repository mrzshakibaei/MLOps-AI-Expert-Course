# json prépare le fichier d'entrée de test ; pytest fournit tmp_path comme dossier temporaire.
import json

from ml_pipeline.predict import predict
from ml_pipeline.train import train_model


def test_training_saves_model_and_metrics(tmp_path):
    """L'entraînement doit produire des métriques et des artefacts lisibles."""
    # tmp_path fournit un dossier isolé qui sera supprimé après le test.
    metrics = train_model(tmp_path)

    # Vérifie que l'évaluation produit des métriques valides et que les deux
    # artefacts attendus sont effectivement écrits.
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_f1_macro"] <= 1.0
    assert (tmp_path / "iris_pipeline.joblib").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics


def test_saved_pipeline_predicts_json_input(tmp_path):
    """Le modèle sauvegardé doit pouvoir servir une prédiction JSON connue."""
    # Prépare un modèle réellement entraîné et une fleur connue de type setosa.
    train_model(tmp_path)
    sample = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }
    input_path = tmp_path / "sample.json"
    input_path.write_text(json.dumps(sample), encoding="utf-8")

    # Vérifie le parcours de bout en bout : lecture JSON, chargement du modèle
    # et prédiction avec le prétraitement sauvegardé.
    result = predict(input_path, tmp_path / "iris_pipeline.joblib")

    assert result[0]["prediction"] == "setosa"
    assert 0.0 <= result[0]["confidence"] <= 1.0
