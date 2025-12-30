import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from sklearn.model_selection import train_test_split
import logging

class MedicalPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        os.makedirs(self.config.processed_dir, exist_ok=True)

    def patient_level_split(self, patient_ids):
        """
        Prevents data leakage by splitting at the patient ID level.
        """
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=(1 - self.config.train_split), 
            random_state=self.config.seed
        )
        logging.info(f"Split complete: {len(train_ids)} Train, {len(test_ids)} Test")
        return train_ids, test_ids

    def n4_bias_correction(self, image_path):
        """
        Removes low-frequency intensity variations (shading artifacts) common in MRI.
        """
        input_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(input_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        output_image = corrector.Execute(input_image, mask)
        return output_image

    def resample_to_isotropic(self, sitk_image):
        """
        Resamples volume to a standard 1mm x 1mm x 1mm spacing.
        This is vital for 3D CNNs to understand physical scale.
        """
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()
        
        new_spacing = self.config.target_spacing
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetInterpolator(sitk.sitkLinear)
        
        return resample.Execute(sitk_image)

    def intensity_normalization(self, np_array):
        """
        Clips and scales intensities to [0, 1] range.
        """
        np_array = np.clip(np_array, self.config.intensity_min, self.config.intensity_max)
        np_array = (np_array - self.config.intensity_min) / (self.config.intensity_max - self.config.intensity_min)
        return np_array.astype(np.float32)