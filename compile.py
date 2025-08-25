
# ✅ Compile the Model

model.compile(

    optimizer=Adam(learning_rate=initial_lr),

    loss=hybrid_loss,

    metrics=["accuracy", iou_metric, dice_coefficient, Precision(), Recall()]  # Include relevant metrics

)
