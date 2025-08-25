

aug_count = 0

for img_path, mask_path in zip(train_x, train_y):

    img = cv2.imread(img_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (W, H))

    mask = cv2.resize(mask, (W, H))

    base = os.path.splitext(os.path.basename(img_path))[0]

    cv2.imwrite(f"{aug_img_dir}/{base}_orig.jpg", img)

    cv2.imwrite(f"{aug_mask_dir}/{base}_orig.jpg", mask)


    for i in range(AUGMENTATION_FACTOR):

        aug = augment_fn(image=img, mask=mask)

        aug_img, aug_mask = aug["image"], aug["mask"]

        cv2.imwrite(f"{aug_img_dir}/{base}_aug{i+1}.jpg", aug_img)

        cv2.imwrite(f"{aug_mask_dir}/{base}_aug{i+1}.jpg", aug_mask)

        aug_count += 1


# ✅ Map and Combine
aug_img_paths = sorted(glob(os.path.join(aug_img_dir, "*.jpg")))

aug_mask_paths = sorted(glob(os.path.join(aug_mask_dir, "*.jpg")))

aug_train_x = [x for x in aug_img_paths if "_orig.jpg" not in x]

aug_train_y = [y for y in aug_mask_paths if "_orig.jpg" not in y]


orig_train_x = [os.path.join(aug_img_dir, os.path.basename(p).replace(".jpg", "_orig.jpg")) for p in train_x]

orig_train_y = [os.path.join(aug_mask_dir, os.path.basename(p).replace(".jpg", "_orig.jpg")) for p in train_y]


full_train_x = orig_train_x + aug_train_x

full_train_y = orig_train_y + aug_train_y


# ✅ Summary

print(f"✅ Saved {len(train_x)} originals and {aug_count} augmented copies.")

print(f"📦 Total = {len(full_train_x)} samples.")

print(f"✅ Validation set: {len(valid_x)} images | {len(valid_y)} masks")

print(f"✅ Test set: {len(test_x)} images | {len(test_y)} masks")
