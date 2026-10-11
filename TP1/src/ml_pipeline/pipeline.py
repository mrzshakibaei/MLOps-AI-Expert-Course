# scikit-learn fournit les étapes de préparation des données et le modèle de classification.
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(random_state: int = 42) -> Pipeline:
    """Assemble le prétraitement et le classifieur dans une seule pipeline.

    Garder ces étapes ensemble garantit que l'imputation et la standardisation
    sont apprises uniquement pendant fit, puis réutilisées de la même façon
    pendant predict. À chaque fit ou predict, scikit-learn applique les étapes
    dans cet ordre : imputer -> standardiser -> classifier.
    """
    return Pipeline(
        steps=[
            # Remplace une mesure manquante par la médiane de sa caractéristique.
            ("imputer", SimpleImputer(strategy="median")),
            # Centre et met à l'échelle les mesures pour faciliter l'apprentissage.
            ("scaler", StandardScaler()),
            (
                "classifier",
                # La graine rend l'entraînement reproductible ; max_iter évite
                # d'arrêter trop tôt l'optimisation sur les données d'entraînement.
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
