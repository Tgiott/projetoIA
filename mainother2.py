# Display the results
exp_dir = os.path.join(output_path, 'runs', 'train')
if os.path.exists(exp_dir):
    exp = sorted(os.listdir(exp_dir))[-1]
    exp_path = os.path.join(exp_dir, exp)

    img_path = os.path.join(exp_path, 'val_batch0_pred.jpg')
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 15))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
    else:
        print("Prediction image not found.")
else:
    print("No training results found.")