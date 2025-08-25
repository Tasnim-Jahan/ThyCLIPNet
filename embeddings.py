

# ✅ Define the function to load CLIP features

def load_clip_npz(split):

    clip_feature_path = "/kaggle/input/tn3kaug71256d32b"

    data = np.load(f"{clip_feature_path}/biomedclip_tn3k_{split}_features.npz", allow_pickle=True)

    features = data["features"]

    filenames = [f.decode() if isinstance(f, bytes) else f for f in data["image_filenames"]]

    return features, filenames


# ✅ Load CLIP features (with augmented image paths for training)

clip_train_features, full_train_x = load_clip_npz("train")

clip_valid_features, valid_x = load_clip_npz("valid")

clip_test_features, test_x = load_clip_npz("test")
