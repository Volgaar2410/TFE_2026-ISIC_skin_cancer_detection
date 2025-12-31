import pandas as pd
from sklearn.impute import KNNImputer

num_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
    'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
]

cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general", 'image_type']

features_absents_test = [
    "lesion_id",
    "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
    "mel_mitotic_index", "mel_thick_mm",
    "tbp_lv_dnn_lesion_confidence"
]

non_feature_cols = ["target", "file_path", "patient_id"]


def get_train_file_path(image_id, TRAIN_DIR):
    return f"{TRAIN_DIR}/{image_id}.jpg"


def prepare_features(df):
    df = df.copy()

    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("missing")
    if num_cols:
        imputer = KNNImputer()
        df[num_cols] = imputer.fit_transform(df[num_cols])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    

    ohe_cols = [c for c in df.columns if any(c.startswith(col + "_") for col in cat_cols)]
    feature_cols = num_cols + ohe_cols

    return df, feature_cols