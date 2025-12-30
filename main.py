def run_ingestion_pipeline():
    # 1. Initialize Configuration
    config = IngestionConfig()
    ingestor = DataIngestor(config)
    
    # 2. Example: Convert clinical DICOMs to ML-ready NIfTI
    # In a real scenario, you would loop through a list of patient IDs
    patient_list = ["patient_001", "patient_002"]
    
    for pid in patient_list:
        # Step A: Convert
        nifti_path = ingestor.convert_dicom_series(pid)
        
        if nifti_path:
            # Step B: Validate Metadata
            is_valid = ingestor.validate_volume(nifti_path)
            
            # Step C: Upload to Data Lake for Team Access
            if is_valid:
                s3_dest = f"{config.raw_s3_prefix}/{pid}.nii.gz"
                ingestor.s3_client.upload_file(nifti_path, config.bucket_name, s3_dest)
                logging.info(f"Pipeline complete for {pid}: Synced to Cloud.")
    

def run_preprocess_pipeline(patient_ids):
    config = PreprocessConfig()
    preprocessor = MedicalPreprocessor(config)
    guard = QualityGuard()
    
    # 1. Patient-Level Split
    train_ids, test_ids = preprocessor.patient_level_split(patient_ids)
    
    metadata_records = []

    for pid in train_ids:
        raw_path = f"./data/raw/{pid}.nii.gz"
        
        # 2. Artifact Correction (N4 Bias)
        corrected_img = preprocessor.n4_bias_correction(raw_path)
        
        # 3. Spatial Resampling
        resampled_img = preprocessor.resample_to_isotropic(corrected_img)
        
        # 4. Final Intensity Scaling
        img_array = sitk.GetArrayFromImage(resampled_img)
        final_array = preprocessor.intensity_normalization(img_array)
        
        # Save preprocessed volume
        save_path = os.path.join(config.processed_dir, f"{pid}_proc.npy")
        np.save(save_path, final_array)
        
        # 5. Collect metadata for Quality Guard
        metadata_records.append({
            "pid": pid,
            "mean_intensity": np.mean(final_array),
            "z_spacing": resampled_img.GetSpacing()[2],
            "file_path": save_path
        })

    # 6. Execute Great Expectations Validation
    df_meta = pd.DataFrame(metadata_records)
    if guard.validate_metadata(df_meta):
        logging.info("Quality Check Passed! Data is ready for training.")
    else:
        logging.error("Quality Check Failed! Check data for artifacts.")

if __name__ == "__main__":
    run_ingestion_pipeline()
    # Example patient list
    patients = ["p1", "p2", "p3", "p4", "p5"]
    run_preprocess_pipeline(patients)