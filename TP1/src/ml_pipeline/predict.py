# Bibliothèques standard Python : options de commande, lecture JSON, logs et chemins de fichiers.
import argparse
import json
import logging
from pathlib import Path

# joblib recharge le modèle entraîné ; pandas met les entrées JSON en tableau tabulaire.
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def predict(input_path: str | Path, model_path: str | Path) -> list[dict]:
    """Prédit l'espèce d'une ou plusieurs fleurs décrites en JSON.

    Ordre d'exécution : lire le JSON -> vérifier sa forme -> charger le modèle
    -> valider les colonnes -> créer le tableau d'entrée -> prédire -> retourner
    les espèces et leurs scores de confiance.
    """
    # Accepte soit un objet JSON unique, soit une liste d'objets pour traiter
    # une ou plusieurs fleurs avec la même commande.
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    logger.info("Fichier d'entrée '%s' chargé.", input_path)
    records = [payload] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("Le JSON doit être un objet ou une liste d'objets.")
    logger.info("Nombre de fleurs à prédire : %d.", len(records))

    # Le modèle sauvegardé contient toute la pipeline, prétraitement compris.
    model = joblib.load(model_path)
    logger.info("Modèle chargé depuis '%s'.", model_path)
    feature_names = list(model.feature_names_in_)
    expected = set(feature_names)
    # Vérifie le schéma avant l'inférence afin d'expliquer clairement les erreurs
    # de saisie au lieu de laisser pandas ou scikit-learn échouer plus tard.
    for index, record in enumerate(records):
        missing = expected - record.keys()
        unexpected = record.keys() - expected
        if missing or unexpected:
            raise ValueError(
                f"Entrée {index}: colonnes manquantes={sorted(missing)}, "
                f"colonnes inconnues={sorted(unexpected)}"
            )
    logger.info("Les données d'entrée ont les caractéristiques attendues.")

    # Respecte l'ordre des caractéristiques attendu à l'entraînement.
    frame = pd.DataFrame(records, columns=feature_names)
    probabilities = model.predict_proba(frame)
    predictions = model.predict(frame)
    logger.info("Prédictions terminées.")
    # Dans Iris, les identifiants 0, 1 et 2 correspondent à ces trois noms.
    class_names = ["setosa", "versicolor", "virginica"]
    # Associe chaque classe prédite à sa probabilité la plus élevée.
    results = [
        {
            "prediction": class_names[int(prediction)],
            "confidence": round(float(probabilities[row].max()), 4),
        }
        for row, prediction in enumerate(predictions)
    ]
    for index, result in enumerate(results, start=1):
        logger.info(
            "Fleur %d : espèce prédite=%s, confiance=%.4f.",
            index,
            result["prediction"],
            result["confidence"],
        )
    return results


def main() -> None:
    """Point d'entrée CLI : lit les chemins, appelle predict et affiche le JSON."""
    # Le chemin du modèle a une valeur par défaut ; le fichier d'entrée est requis.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Prédit l'espèce d'une fleur.")
    parser.add_argument("--model", type=Path, default=Path("models/iris_pipeline.joblib"))
    parser.add_argument("--input", type=Path, required=True, help="Fichier JSON d'entrée")
    arguments = parser.parse_args()
    print(json.dumps(predict(arguments.input, arguments.model), indent=2))


# Permet aussi d'importer predict() sans lancer l'interface en ligne de commande.
if __name__ == "__main__":
    main()
