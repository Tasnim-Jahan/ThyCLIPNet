
def build_model_with_clip(input_shape, clip_features_shape):

    image_input = Input(input_shape, name="image_input")

    clip_input = Input((clip_features_shape[0],), name="clip_input")




    encoder = MobileNetV2(include_top=False, weights="imagenet", input_tensor=image_input, input_shape=input_shape)

    skip1 = Dropout(0.3)(eca_block(encoder.get_layer("block_1_expand_relu").output))  # Reduce dropout in encoder

    skip2 = Dropout(0.3)(eca_block(encoder.get_layer("block_3_expand_relu").output))

    skip3 = Dropout(0.3)(eca_block(encoder.get_layer("block_6_expand_relu").output))

    encoder_output = Dropout(0.3)(eca_block(encoder.get_layer("block_13_expand_relu").output))  # Reduce dropout in encoder




    aspp = aspp_block(encoder_output)

    aspp = SpatialDropout2D(0.2)(aspp)

    bridge = customized_cbam_block(aspp)





    decoder1 = cross_attention_block(clip_input, decoder_block(bridge, skip3, 256), 256)

    decoder2 = cross_attention_block(clip_input, decoder_block(decoder1, skip2, 128), 128)

    decoder3 = decoder_block(decoder2, skip1, 64)





    final_output = Conv2D(1, (1, 1), activation="sigmoid")(decoder3)

    final_output = UpSampling2D(size=(2, 2), interpolation="bilinear")(final_output)



    return tf.keras.models.Model(inputs=[image_input, clip_input], outputs=final_output)
