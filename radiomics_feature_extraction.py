# ==============================================================================
# 1. INSTALASI LIBRARY
# ==============================================================================
# Jalankan sekali: pip install pyradiomics SimpleITK pandas

import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from pathlib import Path

# ==============================================================================
# 2. KONFIGURASI (SESUAI DENGAN DATASET LOKAL FEDBCA)
# ==============================================================================

# Path ke dataset FedBCa lokal
BASE_DIR = r"D:\Research\DATASET\FedBCa"

# Output CSV (simpan di Research folder)
OUTPUT_CSV = r"D:\Research\radiomics_features_FedBCa_Centers1-4_final.csv"
OUTPUT_CSV_HARMONIZED = r"D:\Research\radiomics_features_FedBCa_Centers1-4_HARMONIZED_final.csv"

# Verifikasi path
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"[ERR] Base directory tidak ditemukan: {BASE_DIR}")

print(f"[INFO] Base directory: {BASE_DIR}")
print(f"[INFO] Output CSV akan disimpan ke: {OUTPUT_CSV}")

# Setup PyRadiomics Logger agar tidak berisik
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

# Konfigurasi Ekstraktor (Optimal untuk MRI T2WI + High-Order Features)
settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkBSpline,
    'resamplingInterpolator': sitk.sitkBSpline,
    'enableCExtensions': True,
    'normalize': True,
    'normalizeScale': 100,
    'correctMask': True,  
    'force2D': False
}

# Buat extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)


extractor.enableImageTypeByName('Original')  # Original image
extractor.enableImageTypeByName('Wavelet')   # Wavelet 
extractor.enableImageTypeByName('LoG', customArgs={'sigma': [2.0, 3.0, 4.0]})  # LoG 3 scales
extractor.enableImageTypeByName('Square')    # Square filter

# Enable feature classes
extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('ngtdm')

print("[INFO] PyRadiomics Extractor siap dengan high-order features:")
print("       ✓ Original image (7 feature classes)")
print("       ✓ Wavelet (8 decompositions x 6 texture classes)")
print("       ✓ LoG (3 sigma: 2.0, 3.0, 4.0 mm)")
print("       ✓ Square filter")
print("       ESTIMATED: 900-1000+ features total")

# Settings untuk HARMONIZED (dengan isotropic resampling + High-Order Features)
settings_harmonized = {
    'binWidth': 25,
    'resampledPixelSpacing': [1.0, 1.0, 1.0],  # Isotropic 1mm x 1mm x 1mm
    'interpolator': sitk.sitkBSpline,
    'resamplingInterpolator': sitk.sitkBSpline,
    'enableCExtensions': True,
    'normalize': True,
    'normalizeScale': 100,
    'correctMask': True,
    'force2D': False
}

# Buat harmonized extractor
extractor_harmonized = featureextractor.RadiomicsFeatureExtractor(**settings_harmonized)

extractor_harmonized.enableImageTypeByName('Original')
extractor_harmonized.enableImageTypeByName('Wavelet')
extractor_harmonized.enableImageTypeByName('LoG', customArgs={'sigma': [2.0, 3.0, 4.0]})
extractor_harmonized.enableImageTypeByName('Square')

# Enable feature classes
extractor_harmonized.enableFeatureClassByName('shape')
extractor_harmonized.enableFeatureClassByName('firstorder')
extractor_harmonized.enableFeatureClassByName('glcm')
extractor_harmonized.enableFeatureClassByName('glrlm')
extractor_harmonized.enableFeatureClassByName('glszm')
extractor_harmonized.enableFeatureClassByName('gldm')
extractor_harmonized.enableFeatureClassByName('ngtdm')

print("[INFO] PyRadiomics Extractor HARMONIZED siap (isotropic 1x1x1mm + high-order features).")
print("       ✓ Same configuration as normal extractor + resampling")

# ==============================================================================
# 3. FUNGSI PENCARIAN DATA 
# ==============================================================================

def find_data_pairs(base_dir):
    """
    Fungsi untuk mencari pasangan gambar (T2WI) dan mask (Annotation) di FedBCa.
    Hanya memproses Center2, Center3, Center4 (sesuai instruksi).
    
    Return: List of dict dengan struktur:
    {
        'patient_id': str,
        'image_paths': list[str],
        'mask_paths': list[str],
        'is_positive': bool,
        'center': str
    }
    """
    print(f"[INFO] Langkah 1: Mencari pasangan data di {base_dir}")
    print("[INFO] Centers yang diproses: Center2, Center3, Center4 (Center2 DILEWATI)")
    
    patient_data_list = []
    centers = ["Center1","Center2", "Center3", "Center4"]  # Center1 DILEWATI
    total_img_studies = 0
    total_mask_studies = 0
    total_matched_studies = 0

    for center in centers:
        center_path = os.path.join(base_dir, center)
        image_dir = os.path.join(center_path, "T2WI")
        mask_dir = os.path.join(center_path, "Annotation")

        # Verifikasi folder ada
        if not os.path.exists(image_dir):
            print(f"[WARN] {center} - Folder T2WI tidak ditemukan: {image_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"[WARN] {center} - Folder Annotation tidak ditemukan: {mask_dir}")
            continue

        print(f"\n--- Memproses {center} ---")

        # 1. PETAKAN GAMBAR (T2WI) - File langsung di T2WI folder
        image_map = {}
        try:
            for file in os.listdir(image_dir):
                if file.endswith(('.nii', '.nii.gz')):
                    # Skip file yang terlihat seperti mask/segmentasi
                    if any(skip in file.lower() for skip in ['mask', 'seg', 'annotation']):
                        continue
                    
                    img_path = os.path.join(image_dir, file)
                    
                    # Key = nama file tanpa extension
                    img_key = file.replace(".nii.gz", "").replace(".nii", "")
                    
                    if img_key not in image_map:
                        image_map[img_key] = []
                    image_map[img_key].append(img_path)
            
            print(f"    ✓ Menemukan {len(image_map)} file gambar")
        except Exception as e:
            print(f"    ✗ Error saat membaca T2WI: {e}")
            continue

        # 2. PETAKAN MASK (ANNOTATION) - Bisa file atau folder
        mask_map = {}
        try:
            for item_name in os.listdir(mask_dir):
                item_path = os.path.join(mask_dir, item_name)
                
                is_nifti = item_name.endswith(('.nii', '.nii.gz'))
                is_folder = os.path.isdir(item_path)

                if not (is_nifti or is_folder):
                    continue

                try:
                    # KASUS 1: Mask dalam folder (misal: Annotation/Patient123/mask.nii.gz)
                    if is_folder:
                        mask_files = [
                            os.path.join(item_path, f) 
                            for f in os.listdir(item_path) 
                            if f.endswith(('.nii', '.nii.gz'))
                        ]
                        
                        if mask_files:
                            mask_key = item_name
                            if mask_key not in mask_map:
                                mask_map[mask_key] = []
                            mask_map[mask_key].extend(mask_files)
                    
                    # KASUS 2: Mask file langsung (misal: Annotation/Patient123.nii.gz)
                    elif is_nifti and os.path.isfile(item_path):
                        mask_base = item_name.replace(".nii.gz", "").replace(".nii", "")
                        mask_key = mask_base
                        if mask_key not in mask_map:
                            mask_map[mask_key] = []
                        mask_map[mask_key].append(item_path)
                
                except Exception as e:
                    print(f"    [Warn] Gagal memproses mask {item_name}: {e}")
                    continue
            
            print(f"    ✓ Menemukan {len(mask_map)} studi mask")
        except Exception as e:
            print(f"    ✗ Error saat membaca Annotation: {e}")
            continue

        # 3. COCOKKAN GAMBAR DENGAN MASK (dengan flexible matching)
        matched_count = 0
        for img_key, img_paths in image_map.items():
            # Strategi 1: Exact match
            mask_paths = mask_map.get(img_key, [])
            
            # Strategi 2: Prefix/suffix match
            if not mask_paths:
                for mask_key in mask_map.keys():
                    if img_key in mask_key or mask_key in img_key:
                        mask_paths = mask_map[mask_key]
                        print(f"    [Match] {img_key} ↔ {mask_key}")
                        break
            
            # Strategi 3: Extract numbers dan coba match
            if not mask_paths:
                import re
                img_numbers = re.findall(r'\d+', img_key)
                if img_numbers:
                    img_num = img_numbers[-1]  # Ambil number terakhir (misal: 001 dari BC_001)
                    for mask_key in mask_map.keys():
                        mask_numbers = re.findall(r'\d+', mask_key)
                        if mask_numbers and img_num in mask_numbers:
                            mask_paths = mask_map[mask_key]
                            print(f"    [Match by number] {img_key} ({img_num}) ↔ {mask_key}")
                            break
            
            is_positive = len(mask_paths) > 0
            
            patient_data_list.append({
                "patient_id": f"{center}_{img_key}",
                "image_paths": img_paths,
                "mask_paths": mask_paths,
                "is_positive": is_positive,
                "center": center
            })
            
            if is_positive:
                matched_count += 1

        total_img_studies += len(image_map)
        total_mask_studies += len(mask_map)
        total_matched_studies += matched_count
        
        print(f"    ✓ Cocok: {matched_count}/{len(image_map)} pasien")

    # RINGKASAN
    print("\n" + "="*60)
    print("[SUMMARY] Hasil Pencarian Data")
    print("="*60)
    print(f"Total Pasien (semua):     {len(patient_data_list)}")
    print(f"Total Positif (ada mask): {total_matched_studies}")
    print(f"Total Negatif (no mask):  {len(patient_data_list) - total_matched_studies}")

    if total_matched_studies == 0:
        raise ValueError("[CRITICAL] Tidak ada pasangan data ditemukan! Periksa struktur folder!")

    return patient_data_list

# ==============================================================================
# 4. FUNGSI EKSTRAKSI RADIOMICS
# ==============================================================================

def run_extraction():
    """
    Ekstrak radiomics features dari semua pasien positif (yang punya mask).
    """
    try:
        # Cari data pairs
        patients = find_data_pairs(BASE_DIR)
    except Exception as e:
        print(f"[ERR] Gagal mencari data: {e}")
        return

    # Filter hanya pasien positif (yang punya mask)
    positive_patients = [p for p in patients if p['is_positive']]
    negative_patients = [p for p in patients if not p['is_positive']]
    
    print(f"\n[INFO] Akan memproses {len(positive_patients)} pasien POSITIF (abaikan {len(negative_patients)} negatif)")

    all_features = []
    success_count = 0
    error_count = 0

    print(f"\n[START] Memulai ekstraksi fitur radiomics...")
    print("="*70)

    for i, patient in enumerate(positive_patients):
        pid = patient['patient_id']
        center = patient['center']
        
        # Ambil gambar pertama (biasanya 1 T2WI per pasien)
        if not patient['image_paths']:
            print(f"[{i+1:3d}/{len(positive_patients)}] ✗ {pid:30s} - TIDAK ADA GAMBAR")
            error_count += 1
            continue
        
        img_path = patient['image_paths'][0]
        
        # Loop semua mask (jika ada multiple lesi per pasien)
        for mask_idx, mask_path in enumerate(patient['mask_paths']):
            try:
                # Progress indicator
                print(f"[{i+1:3d}/{len(positive_patients)}] ➜ {pid:30s} (Mask {mask_idx+1})...", end=" ", flush=True)
                
                # === PRE-VALIDATION: Check image and mask compatibility ===
                try:
                    img_sitk = sitk.ReadImage(img_path)
                    mask_sitk = sitk.ReadImage(mask_path)
                    
                    # Check if mask is empty (all zeros)
                    mask_array = sitk.GetArrayFromImage(mask_sitk)
                    if mask_array.sum() == 0:
                        print("✗ Mask kosong (all zeros)")
                        error_count += 1
                        continue
                    
                    # Check if dimensions match (after potential resampling)
                    img_size = img_sitk.GetSize()
                    mask_size = mask_sitk.GetSize()
                    
                    # If sizes don't match, try to resample mask to image
                    if img_size != mask_size:
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(img_sitk)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        mask_sitk = resampler.Execute(mask_sitk)
                        
                        # Check again after resampling
                        mask_array = sitk.GetArrayFromImage(mask_sitk)
                        if mask_array.sum() == 0:
                            print("✗ Mask kosong setelah resample")
                            error_count += 1
                            continue
                    
                except Exception as val_error:
                    print(f"✗ Validation error: {str(val_error)[:40]}")
                    error_count += 1
                    continue
                
                # === EKSTRAKSI PYRADIOMICS ===
                result = extractor.execute(img_path, mask_path)
                
                # Bersihkan hasil (buang metadata/diagnostic)
                clean_result = {
                    k: v for k, v in result.items() 
                    if "diagnostic" not in k and not k.startswith("Image-")
                }
                
                # Tambahkan metadata penting
                clean_result['PatientID'] = pid
                clean_result['Center'] = center
                clean_result['MaskIndex'] = mask_idx
                clean_result['ImagePath'] = img_path
                clean_result['MaskPath'] = mask_path
                
                all_features.append(clean_result)
                success_count += 1
                print("✓")
                
            except Exception as e:
                error_count += 1
                print(f"✗ Error: {str(e)[:50]}")
                continue

    # === HASIL AKHIR ===
    print("\n" + "="*70)
    print(f"[DONE] Ekstraksi selesai!")
    print(f"  ✓ Berhasil: {success_count}")
    print(f"  ✗ Error:   {error_count}")
    print(f"  Total:     {len(all_features)} fitur dihasilkan")
    print("="*70)

    # === SIMPAN KE CSV ===
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Atur urutan kolom: ID dulu, baru feature
        id_cols = ['PatientID', 'Center', 'MaskIndex', 'ImagePath', 'MaskPath']
        feature_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + feature_cols]
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SAVED] File berhasil disimpan!")
        print(f"  Path: {OUTPUT_CSV}")
        print(f"  Size: {len(df)} rows × {len(df.columns)} columns")
        print(f"\n[PREVIEW] 5 baris pertama:")
        print("-" * 70)
        print(df[id_cols].head())
        print(f"\n[FEATURES] Total fitur radiomics: {len(feature_cols)}")
        print(f"  Shape    class: {sum(1 for c in feature_cols if 'shape' in c.lower())}")
        print(f"  1st Order class: {sum(1 for c in feature_cols if 'firstorder' in c.lower())}")
        print(f"  GLCM     class: {sum(1 for c in feature_cols if 'glcm' in c.lower())}")
        print(f"  GLRLM    class: {sum(1 for c in feature_cols if 'glrlm' in c.lower())}")
        print(f"  GLSZM    class: {sum(1 for c in feature_cols if 'glszm' in c.lower())}")
        print(f"  GLDM     class: {sum(1 for c in feature_cols if 'gldm' in c.lower())}")
        print(f"  NGTDM    class: {sum(1 for c in feature_cols if 'ngtdm' in c.lower())}")
    else:
        print("[WARN] Tidak ada fitur yang berhasil diektrak!")

def run_extraction_harmonized():
    """
    Fungsi untuk ekstraksi fitur radiomics dengan HARMONIZED settings (isotropic resampling).
    """
    all_features = []
    success_count = 0
    error_count = 0

    # Cari data
    all_patients = find_data_pairs(BASE_DIR)
    positive_patients = [p for p in all_patients if p['is_positive']]

    print(f"\n[START] Memulai ekstraksi fitur radiomics (HARMONIZED)...")
    print("="*70)

    for i, patient in enumerate(positive_patients):
        pid = patient['patient_id']
        center = patient['center']
        
        if not patient['image_paths']:
            print(f"[{i+1:3d}/{len(positive_patients)}] ✗ {pid:30s} - TIDAK ADA GAMBAR")
            error_count += 1
            continue
        
        img_path = patient['image_paths'][0]
        
        for mask_idx, mask_path in enumerate(patient['mask_paths']):
            try:
                print(f"[{i+1:3d}/{len(positive_patients)}] ➜ {pid:30s} (Harmonized, Mask {mask_idx+1})...", end=" ", flush=True)
                
                # Pre-validation
                try:
                    img_sitk = sitk.ReadImage(img_path)
                    mask_sitk = sitk.ReadImage(mask_path)
                    
                    mask_array = sitk.GetArrayFromImage(mask_sitk)
                    if mask_array.sum() == 0:
                        print("✗ Mask kosong (all zeros)")
                        error_count += 1
                        continue
                    
                    img_size = img_sitk.GetSize()
                    mask_size = mask_sitk.GetSize()
                    
                    if img_size != mask_size:
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(img_sitk)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        mask_sitk = resampler.Execute(mask_sitk)
                        
                        mask_array = sitk.GetArrayFromImage(mask_sitk)
                        if mask_array.sum() == 0:
                            print("✗ Mask kosong setelah resample")
                            error_count += 1
                            continue
                    
                except Exception as val_error:
                    print(f"✗ Validation error: {str(val_error)[:40]}")
                    error_count += 1
                    continue
                
                # EKSTRAKSI dengan HARMONIZED extractor
                result = extractor_harmonized.execute(img_path, mask_path)
                
                clean_result = {
                    k: v for k, v in result.items() 
                    if "diagnostic" not in k and not k.startswith("Image-")
                }
                
                clean_result['PatientID'] = pid
                clean_result['Center'] = center
                clean_result['MaskIndex'] = mask_idx
                clean_result['ImagePath'] = img_path
                clean_result['MaskPath'] = mask_path
                
                all_features.append(clean_result)
                success_count += 1
                print("✓")
                
            except Exception as e:
                error_count += 1
                print(f"✗ Error: {str(e)[:50]}")
                continue

    print("\n" + "="*70)
    print(f"[DONE] Ekstraksi HARMONIZED selesai!")
    print(f"  ✓ Berhasil: {success_count}")
    print(f"  ✗ Error:   {error_count}")
    print(f"  Total:     {len(all_features)} fitur dihasilkan")
    print("="*70)

    # SIMPAN KE CSV HARMONIZED
    if all_features:
        df = pd.DataFrame(all_features)
        
        id_cols = ['PatientID', 'Center', 'MaskIndex', 'ImagePath', 'MaskPath']
        feature_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + feature_cols]
        
        df.to_csv(OUTPUT_CSV_HARMONIZED, index=False)
        print(f"\n[SAVED] File HARMONIZED berhasil disimpan!")
        print(f"  Path: {OUTPUT_CSV_HARMONIZED}")
        print(f"  Size: {len(df)} rows × {len(df.columns)} columns")
    else:
        print("[WARN] Tidak ada fitur yang berhasil diektrak!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RADIOMICS FEATURE EXTRACTION - FedBCa Dataset (Center2-4)")
    print("DUAL MODE: Normal + Harmonized (Isotropic Resampling)")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Output File 1 (Normal):     {OUTPUT_CSV}")
    print(f"Output File 2 (Harmonized): {OUTPUT_CSV_HARMONIZED}")
    print("="*70 + "\n")
    
    try:
        # EKSTRAKSI 1: NORMAL (tanpa isotropic resampling)
        print("\n[PHASE 1/2] EKSTRAKSI NORMAL")
        print("="*70)
        run_extraction()
        
        # EKSTRAKSI 2: HARMONIZED (dengan isotropic resampling 1x1x1mm)
        print("\n\n[PHASE 2/2] EKSTRAKSI HARMONIZED")
        print("="*70)
        run_extraction_harmonized()
        
        print("\n" + "="*70)
        print("[SUCCESS] Program selesai dengan sukses!")
        print("  ✓ File 1: " + OUTPUT_CSV)
        print("  ✓ File 2: " + OUTPUT_CSV_HARMONIZED)
        print("="*70)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()