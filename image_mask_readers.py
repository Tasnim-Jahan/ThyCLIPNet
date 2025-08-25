

# ✅ Image Readers
def read_image(path):

    x = tf.io.read_file(path)

    x = tf.image.decode_jpeg(x, channels=3)

    x = tf.image.resize(x, (H, W))

    x = tf.cast(x, tf.float32) / 255.0

    return x



# ✅ Mask Readers
def read_mask(path):

    x = tf.io.read_file(path)

    x = tf.image.decode_jpeg(x, channels=1)

    x = tf.image.resize(x, (H, W))

    x = tf.cast(x, tf.float32) / 255.0

    x = tf.where(x > 0.5, 1.0, 0.0)

    return x
