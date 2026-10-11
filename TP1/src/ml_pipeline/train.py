# Bibliothèques standard Python : options de commande, JSON, logs et chemins de fichiers.
import argparse
import json
import logging
from pathlib import Path

# joblib sauvegarde puis recharge les objets Python entraînés, ici la pipeline ML.
import joblib
# scikit-learn fournit le jeu Iris, les métriques d'évaluation et les outils de séparation/validation.
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

# Code du projet : fabrique le prétraitement et le classifieur réunis en pipeline.
from ml_pipeline.pipeline import create_pipeline

logger = logging.getLogger(__name__)


def train_model(output_dir: str | Path = "models", random_state: int = 42) -> dict:
    """Entraîne, évalue et sauvegarde le modèle et ses métriques.

    Ordre d'exécution : charger les données -> séparer train/test -> faire la
    validation croisée -> entraîner -> évaluer -> sauvegarder -> retourner
    les métriques à l'appelant.
    """
    # Charge le jeu Iris inclus dans scikit-learn : aucun fichier externe requis.
    dataset = load_iris(as_frame=True)
    features = dataset.data
    target = dataset.target
    logger.info("Jeu Iris chargé : %d exemples, %d caractéristiques.", *features.shape)

    # Conserve 20 % pour l'évaluation finale ; stratify garde les proportions
    # des trois espèces comparables entre les deux partitions.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    logger.info(
        "Données séparées : %d exemples d'entraînement, %d de test.",
        len(x_train),
        len(x_test),
    )

    pipeline = create_pipeline(random_state=random_state)
    # La validation croisée s'effectue uniquement sur l'entraînement. Chaque
    # pli réapprend ainsi ses transformations sans voir ses données de validation.
    logger.info("Validation croisée en 5 plis (F1 macro) en cours.")
    cv_scores = cross_val_score(
        pipeline, x_train, y_train, cv=5, scoring="f1_macro"
    )
    # Ajuste une pipeline finale sur toutes les données d'entraînement,
    # puis mesure ses résultats une seule fois sur le jeu de test tenu à part.
    logger.info("Entraînement du modèle final en cours.")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    logger.info("Évaluation sur le jeu de test terminée.")

    # Le F1 macro donne le même poids à chaque espèce, même si les classes
    # n'ont pas exactement le même nombre d'exemples.
    metrics = {
        "dataset": "Iris",
        "model": "LogisticRegression",
        "random_state": random_state,
        "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "class_names": dataset.target_names.tolist(),
        "feature_names": features.columns.tolist(),
    }
    logger.info(
        "Résultats test : accuracy=%.4f, F1 macro=%.4f.",
        metrics["test_accuracy"],
        metrics["test_f1_macro"],
    )
    logger.info(
        "Validation croisée : F1 macro moyen=%.4f (écart-type=%.4f).",
        metrics["cv_f1_macro_mean"],
        metrics["cv_f1_macro_std"],
    )

    # Crée le dossier de sortie si nécessaire et sauvegarde le prétraitement
    # avec le modèle pour pouvoir réutiliser exactement le même parcours.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path / "iris_pipeline.joblib")
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info("Modèle et métriques sauvegardés dans '%s'.", output_path)
    return metrics


def main() -> None:
    """Point d'entrée CLI : lit les options, lance train_model, affiche le bilan."""
    # Configure les messages de progression pour la commande en ligne de commande.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Ces options permettent de changer l'emplacement des artefacts ou la graine
    # sans modifier le code source.
    parser = argparse.ArgumentParser(description="Entraîne la pipeline Iris.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--random-state", type=int, default=42)
    arguments = parser.parse_args()

    # Affiche les métriques dans le terminal, en plus du fichier JSON sauvegardé.
    # Cet appel déclenche toutes les étapes documentées dans train_model().
    metrics = train_model(arguments.output_dir, arguments.random_state)
    print(json.dumps(metrics, indent=2))


# N'exécute la commande que si ce module est lancé directement, pas lors d'un import.
if __name__ == "__main__":
    main()
