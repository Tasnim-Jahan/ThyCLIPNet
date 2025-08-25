
def create_combined_dataset(image_paths, clip_features, mask_paths, batch_size, augment=False):

    def _parse(image_path, clip_feature, mask_path):

        image = read_image(image_path)

        mask = read_mask(mask_path)

        if augment:

            image, mask = apply_augmentation(image, mask)

        clip_feature = tf.reshape(clip_feature, (512,))

        return (image, clip_feature), mask




    dataset = tf.data.Dataset.from_tensor_slices((image_paths, clip_features, mask_paths))

    dataset = dataset.shuffle(1000, seed=42)  # ✅ Added shuffle before cache for training

    dataset = dataset.cache()                 # ✅ Cache before batch (image size = 256x256)

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)



    return dataset



# ✅ Enable GPU Memory Growth (for TensorFlow 2.x)

gpus = tf.config.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth to prevent out-of-memory errors

        print("✅ TensorFlow GPU memory growth enabled.")

    except RuntimeError as e:

        print(f"⚠️ Error enabling memory growth: {e}")
