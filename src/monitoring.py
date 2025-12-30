import pandas as pd
from pathlib import Path
from config.global_config import GlobalConfig

class MedicalMonitoring:
    def __init__(self):
        self.telemetry_path = GlobalConfig.METADATA_DIR / "inference_telemetry.csv"
        self.reference_path = GlobalConfig.METADATA_DIR / "training_baseline_stats.csv"
        self.report_output = GlobalConfig.REPORTS_DIR

    def generate_baseline(self, train_df: pd.DataFrame):
        """
        Saves the statistical distribution of the training set.
        Run this once after Phase 2 (Preprocessing).
        """
        train_df.to_csv(self.reference_path, index=False)
        print(f"✅ Training baseline saved to {self.reference_path}")

    def run_drift_analysis(self):
        """
        Compares recent inference data against the baseline.
        """
        if not self.telemetry_path.exists() or not self.reference_path.exists():
            print("❌ Telemetry or Baseline data missing. Cannot run drift analysis.")
            return

        # 1. Load Data
        reference_data = pd.read_csv(self.reference_path)
        current_data = pd.read_csv(self.telemetry_path)

        # 2. Define the Evidently Report (import at runtime to avoid top-level import errors)
        try:
            import importlib
            report_mod = importlib.import_module("evidently.report")
            preset_mod = importlib.import_module("evidently.metric_preset")
            Report = getattr(report_mod, "Report")
            DataDriftPreset = getattr(preset_mod, "DataDriftPreset")
            TargetDriftPreset = getattr(preset_mod, "TargetDriftPreset")
        except Exception as e:
            print(f"❌ Evidently package is not installed or could not be imported: {e}")
            print("Install it via: pip install evidently")
            return

        # We track intensity distribution and predicted volumes
        drift_report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset()
        ])

        print("📊 Analyzing Data Drift...")
        drift_report.run(reference_data=reference_data, current_data=current_data)

        # 3. Save Report as HTML
        report_filename = self.report_output / "drift_report.html"
        drift_report.save_html(str(report_filename))
        
        # 4. Expert Alerting Logic
        result = drift_report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        
        if drift_detected:
            print("⚠️ ALERT: Significant Data Drift detected! Retraining may be required.")
        else:
            print("✅ System Stable: No significant drift detected.")

        return report_filename

if __name__ == "__main__":
    monitor = MedicalMonitoring()
    # In a real workflow, this would be triggered via a Cron job or Airflow
    report = monitor.run_drift_analysis()