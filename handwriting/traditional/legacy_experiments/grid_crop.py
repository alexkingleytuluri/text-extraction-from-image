import cv2
import os

def grid_crop(image_path, rows, cols, save_as_digit_folders):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open: {image_path}")

    height, width = img.shape[:2]
    cell_h = height // rows
    cell_w = width // cols

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = img[y1:y2, x1:x2]

            if save_as_digit_folders:
                # training grid: row number = the digit label
                folder = os.path.join("data", str(r))
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, f"{c+1}.png"), cell)
            else:
                # test row: column number = sequence position
                os.makedirs("test", exist_ok=True)
                cv2.imwrite(os.path.join("test", f"{c}.png"), cell)

    print("Done slicing:", image_path)

# Training grid: 10 rows (digits 0-9), 10 columns (samples)
grid_crop("0-9.jpeg", rows=10, cols=10, save_as_digit_folders=True)

# Test row: 1 row, 10 columns (your written sequence 0123456789)
grid_crop("test_img.jpeg", rows=1, cols=10, save_as_digit_folders=False)