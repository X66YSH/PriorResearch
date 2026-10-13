"""Health check: did a run learn at all? Breast cancer AUC (the original nanoTabPFN sanity check) plus the AUC on
held-out synthetic tables (eval/synthetic.py). The thresholds come from the config and are fixed before any run."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

_BREAST_CANCER = None


def breast_cancer_scores(classifier) -> dict:
    """Same split and metrics as eval() in the original nanoTabPFN train.py."""
    global _BREAST_CANCER
    if _BREAST_CANCER is None:
        _BREAST_CANCER = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0)
    X_train, X_test, y_train, y_test = _BREAST_CANCER
    classifier.fit(X_train, y_train)
    prob = classifier.predict_proba(X_test)
    pred = prob.argmax(axis=1)
    return {"roc_auc": float(roc_auc_score(y_test, prob[:, 1])), "acc": float(accuracy_score(y_test, pred)),
            "balanced_acc": float(balanced_accuracy_score(y_test, pred))}


def health_label(base_prior_auc: float, breast_cancer_auc: float, thresholds: dict) -> str:
    """healthy / collapsed / unclear; base_prior_auc is the AUC on held-out tables of the run's base prior."""
    if base_prior_auc >= thresholds["healthy_prior_auc"] and breast_cancer_auc >= thresholds["healthy_breast_cancer_auc"]:
        return "healthy"
    if base_prior_auc < thresholds["collapsed_prior_auc"] or breast_cancer_auc < thresholds["collapsed_breast_cancer_auc"]:
        return "collapsed"
    return "unclear"
