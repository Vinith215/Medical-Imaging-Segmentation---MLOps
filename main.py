import argparse
import sys
from src.ingestion import DataIngestion
from src.preprocess import MedicalPreprocessor
from src.train_pipe import TrainingPipeline
from src.inference import SegmentationInference

def main():
    parser = argparse.ArgumentParser(description="Med-Segmentation MLOps Pipeline")
    parser.add_argument(
        "phase", 
        choices=["ingest", "preprocess", "train", "infer"],
        help="The pipeline phase to execute"
    )
    parser.add_argument("--input", type=str, help="Path to input image (for inference)")
    parser.add_argument("--output", type=str, help="Path to output mask (for inference)")

    args = parser.parse_args()

    if args.phase == "ingest":
        print("🚀 Starting Phase 1: Ingestion...")
        ingestor = DataIngestion()
        ingestor.convert_dicom_to_nifti()

    elif args.phase == "preprocess":
        print("🚀 Starting Phase 2: Preprocessing...")
        preprocessor = MedicalPreprocessor()
        preprocessor.run_pipeline()

    elif args.phase == "train":
        print("🚀 Starting Phase 4/5: Training...")
        trainer = TrainingPipeline()
        trainer.run_training()

    elif args.phase == "infer":
        if not args.input or not args.output:
            print("❌ Error: Inference requires --input and --output paths.")
            sys.exit(1)
        print(f"🚀 Starting Phase 5: Inference on {args.input}...")
        engine = SegmentationInference()
        engine.predict(args.input, args.output)

if __name__ == "__main__":
    main()