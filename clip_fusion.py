

# ✅ Cross-Attention Fusion for CLIP

def cross_attention_block(clip_features, segmentation_features, num_filters):

    clip_proj = Dense(num_filters, activation="relu")(clip_features)

    clip_proj = Reshape((1, 1, num_filters))(clip_proj)

    clip_proj = UpSampling2D(size=(segmentation_features.shape[1], segmentation_features.shape[2]), interpolation="bilinear")(clip_proj)




    attention_weights = Conv2D(num_filters, (1, 1), activation="sigmoid")(segmentation_features)

    gated_features = 0.8 * clip_proj + 0.2 * segmentation_features  # Increase CLIP's weight to 50%

    gated_features = Multiply()([gated_features, attention_weights])



    return Concatenate()([segmentation_features, gated_features])
