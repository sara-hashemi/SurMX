#!/bin/bash

set -e

INPUT_DIR="MAMA-MIA/images"
LABEL_DIR="MAMA-MIA/segmentations/expert"
RAW_DIR="MAMA-MIA/raw/Dataset002_MAMA_MIA"
CLINICAL_XLSX="MAMA-MIA/clinical_and_imaging_info.xlsx"
MAPPING_CSV="$RAW_DIR/image_mapping_with_clinical.csv"
TMP_MAP="$RAW_DIR/tmp_id_map.csv"
MISSING_LIST="$RAW_DIR/missing_modality_0003.txt"

rm -rf "$RAW_DIR/imagesTr" "$RAW_DIR/labelsTr" "$TMP_MAP" "$MAPPING_CSV" "$MISSING_LIST"
mkdir -p "$RAW_DIR/imagesTr" "$RAW_DIR/labelsTr"

# 📋 Extract patient IDs (all for training now)
python3 <<EOF
import pandas as pd
df = pd.read_excel("$CLINICAL_XLSX")
df["patient_id"].to_csv("$RAW_DIR/all_patient_ids.txt", index=False, header=False)
EOF

# Read all patient folders
mapfile -t ALL_PATIENTS < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Prepare temporary mapping file
echo "nnunet_id,patient_id,split" > "$TMP_MAP"
touch "$MISSING_LIST"

valid_count=0
for PATIENT_DIR in "${ALL_PATIENTS[@]}"; do
    PATIENT_ID=$(basename "$PATIENT_DIR")
    MODALITY_SRC="$PATIENT_DIR/${PATIENT_ID}_0003.nii.gz"
    LABEL_SRC="$LABEL_DIR/${PATIENT_ID}.nii.gz"

    if [[ -f "$MODALITY_SRC" ]]; then
        ID_PADDED=$(printf "%03d" "$valid_count")
        NNUNET_ID="Dataset002_MAMA_MIA_${ID_PADDED}"

        # Copy modality 0003 as 0000
        DST_IMG="$RAW_DIR/imagesTr/${NNUNET_ID}_0000.nii.gz"
        cp "$MODALITY_SRC" "$DST_IMG"

        # Copy label
        DST_LABEL="$RAW_DIR/labelsTr/${NNUNET_ID}.nii.gz"
        cp "$LABEL_SRC" "$DST_LABEL"

        echo "$NNUNET_ID,$PATIENT_ID,tr" >> "$TMP_MAP"
        valid_count=$((valid_count + 1))

        echo "✅ Copied $PATIENT_ID"
    else
        echo "$PATIENT_ID" >> "$MISSING_LIST"
        echo "❌ Missing modality 0003 for $PATIENT_ID"
    fi
done

echo ""
echo "📋 Total patients with modality 0003: $valid_count"
echo "📋 Missing patients saved to: $MISSING_LIST"

# Merge temporary map with clinical info
python3 <<EOF
import pandas as pd
map_df = pd.read_csv("$TMP_MAP")
clinical_df = pd.read_excel("$CLINICAL_XLSX")
merged_df = map_df.merge(clinical_df, on="patient_id", how="left")
merged_df.to_csv("$MAPPING_CSV", index=False)
EOF

# Create dataset.json for nnUNet
cat <<EOF > "$RAW_DIR/dataset.json"
{
  "channel_names": {
    "0": "modality4"
  },
  "labels": {
    "background": "0",
    "lesion": "1"
  },
  "modality": {
    "0": "modality4"
  },
  "file_ending": ".nii.gz",
  "numTraining": $valid_count,
  "training": [
$(for ((i=0; i<$valid_count; i++)); do
    printf '    { "image": "./imagesTr/Dataset002_MAMA_MIA_%03d_0000.nii.gz", "label": "./labelsTr/Dataset002_MAMA_MIA_%03d.nii.gz" }%s\n' "$i" "$i" "$( [[ $i -lt $((valid_count - 1)) ]] && echo , )"
done)
  ],
  "test": []
}
EOF

echo "✅ Done! Saved dataset with modality 4 to: $RAW_DIR"