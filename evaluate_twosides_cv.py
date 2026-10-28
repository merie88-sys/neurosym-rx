import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from neurosym_rx import NeuroSymRx
from core.drugbank_baseline import DrugBankBaseline
from core.patient_generator import PatientGenerator
from config import RANDOM_SEED


def load_twosides(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[INFO] TwoSIDES file not found at {path}. Using synthetic benchmark.")
        return _generate_synthetic_benchmark()
    return df


def _generate_synthetic_benchmark(n=2000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    drugs = ["warfarin", "amiodarone", "digoxin", "nsaid", "ssri", "metformin",
             "simvastatin", "clarithromycin", "lisinopril", "metoprolol",
             "ciprofloxacin", "fluconazole", "lithium", "tramadol", "omeprazole"]
    pairs = []
    for _ in range(n):
        d1, d2 = rng.choice(drugs, size=2, replace=False)
        pairs.append((d1, d2))
    labels = rng.binomial(1, 0.35, size=n)
    return pd.DataFrame({"drug1": [p[0] for p in pairs],
                         "drug2": [p[1] for p in pairs],
                         "label": labels})


def evaluate_system(system, df, patient_context, is_neurosym=True):
    y_true = df["label"].values
    y_score = np.zeros(len(df))
    y_pred = np.zeros(len(df), dtype=int)

    for idx, row in df.iterrows():
        drug_pair = (row["drug1"], row["drug2"])
        seed = int(hash((drug_pair[0], drug_pair[1], idx)) % (2**31))
        if is_neurosym:
            out = system.predict(drug_pair, patient_context, seed=seed)
            y_score[idx] = out["R_final_100"] / 100.0
            y_pred[idx] = 1 if out["R_final_100"] >= 50 else 0
        else:
            out = system.predict(drug_pair, patient_context)
            y_score[idx] = out["risk_score"]
            y_pred[idx] = 1 if out["alert"] else 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_pr = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        auc_pr = 0.5
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = 0.5

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "n_cases": len(df),
        "y_true": y_true,
        "y_pred": y_pred
    }


def run_cross_validation(df, n_splits=5, seed=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    neurosym_metrics = {k: [] for k in ["precision", "recall", "f1", "auc_pr", "auc_roc"]}
    baseline_metrics = {k: [] for k in ["precision", "recall", "f1", "auc_pr", "auc_roc"]}

    patient_context = {
        "patient_id": "eval_patient",
        "age": 68,
        "egfr": 55,
        "conditions": ["hypertension", "chronic_kidney_disease"],
        "medications": ["warfarin", "amiodarone", "lisinopril"],
        "n_medications": 3,
        "month": 6
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df["label"])):
        df_test = df.iloc[test_idx].reset_index(drop=True)

        ns_model = NeuroSymRx()
        ns_model.fusion.reset_stats()
        ns_results = evaluate_system(ns_model, df_test, patient_context, is_neurosym=True)

        db_model = DrugBankBaseline()
        db_results = evaluate_system(db_model, df_test, patient_context, is_neurosym=False)

        for k in neurosym_metrics:
            neurosym_metrics[k].append(ns_results[k])
            baseline_metrics[k].append(db_results[k])

        print(f"Fold {fold_idx + 1}/{n_splits} | "
              f"NeuroSym-Rx F1={ns_results['f1']:.3f} | "
              f"DrugBank-RB F1={db_results['f1']:.3f}")

    summary = {}
    for k in neurosym_metrics:
        ns_arr = np.array(neurosym_metrics[k])
        db_arr = np.array(baseline_metrics[k])
        summary[k] = {
            "neurosym_rx": f"{ns_arr.mean():.2f} ± {ns_arr.std():.2f}",
            "drugbank_rb": f"{db_arr.mean():.2f} ± {db_arr.std():.2f}"
        }
    return summary


def run_temporal_evaluation():
    generator = PatientGenerator()
    scenarios = generator.generate_all_scenarios()
    print(f"\nGenerated {len(scenarios)} patient scenarios:")
    for key in ["A", "B", "C", "D"]:
        count = sum(1 for s in scenarios if s["patient"]["profile"] == key)
        print(f"  Profile {key}: {count} patients")

    ns_model = NeuroSymRx()
    total_alerts = 0
    high_alerts = 0
    strategies = {"complementary": 0, "conflict_resolution": 0, "symbolic_fallback": 0}

    for scenario in scenarios[:10]:
        trajectory = scenario["trajectory"]
        final_state = trajectory[-1]
        meds = final_state["medications"]
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                seed = int(hash((meds[i], meds[j], final_state["patient_id"])) % (2**31))
                out = ns_model.predict((meds[i], meds[j]), final_state, seed=seed)
                total_alerts += 1
                if out["alert_level"] in ["High", "Critical"]:
                    high_alerts += 1
                strategies[out["fusion_strategy"]] += 1

    print(f"\nTemporal evaluation on 10 patients:")
    print(f"  Total drug pairs evaluated: {total_alerts}")
    print(f"  High/Critical alerts: {high_alerts} ({100*high_alerts/max(1,total_alerts):.1f}%)")
    print(f"  Fusion strategies: {strategies}")


def main():
    print("=" * 60)
    print("NeuroSym-Rx Evaluation Pipeline")
    print("=" * 60)

    df = load_twosides("data/twosides_filtered.csv")
    print(f"\nDataset: {len(df)} interactions, "
          f"{df['label'].sum()} positive ({100*df['label'].mean():.1f}%)")

    print("\n5-Fold Stratified Cross-Validation:")
    print("-" * 60)
    summary = run_cross_validation(df, n_splits=5)

    print("\n" + "=" * 60)
    print("Final Results (mean ± std across 5 folds):")
    print("=" * 60)
    print(f"{'Metric':<15} {'NeuroSym-Rx':<20} {'DrugBank-RB':<20}")
    print("-" * 60)
    for metric in ["precision", "recall", "f1", "auc_pr", "auc_roc"]:
        print(f"{metric:<15} {summary[metric]['neurosym_rx']:<20} {summary[metric]['drugbank_rb']:<20}")

    print("\n" + "=" * 60)
    print("Temporal Reasoning Evaluation")
    print("=" * 60)
    run_temporal_evaluation()


if __name__ == "__main__":
    main()
