from dataset.load import ds_root
from dataset.dataset import loaded_and_decoded_ds, without_bg_ds, without_bg_and_shadows_ds, triple_ds, augmented_ds

# TODO:
# - resize images: which scale ?
# - flatten labels to vector
# - augment
# --- remove background ✔️
# --- remove shadows ✔️
# --- flip ✔️
# ----- horizontally ✔️
# ----- vertically ✔️
# ----- both ✔️
# --- color ✔️
# ----- grayscale ✔️
# ----- inverse color ✔️
# ----- remove random channel ✔️
# --- other
# ----- add noise
# ----- blur
# ----- grid mask
# --- distortions
# ----- brightness ✔️
# ----- saturation ✔️
# ----- contrast ✔️
# ----- hue ✔️
# ----- rotation
# ----- sharpening
# --- smart translations (depending on labels)
# --- synthetic objects
# - tune performance (https://www.tensorflow.org/guide/data_performance#the_dataset)
# --- cache
# --- prefetch
# --- interleave
# --- parallel map
# --- map on batches? - vectorize
# - prepare TFRecords ?
