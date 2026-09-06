import os
import cv2

test_folder = "test_clean"

errors = [
    ("0.png", "0", "3"),
    ("1.png", "1", "l"),
    ("2.png", "2", "9"),
    ("5.png", "5", "T"),
    ("6.png", "6", "4"),
    ("7.png", "7", "P"),
    ("8.png", "8", "e"),
    ("J.png", "J", "1"),
    ("O.png", "O", "o"),
    ("s-a.png", "a", "u"),
    ("s-c.png", "c", "e"),
    ("s-e.png", "e", "R"),
    ("s-f.png", "f", "c"),
    ("s-g.png", "g", "9"),
    ("s-h.png", "h", "r"),
    ("s-j.png", "j", "f"),
    ("s-k.png", "k", "B"),
    ("s-l.png", "l", "1"),
    ("s-s.png", "s", "o"),
    ("s-t.png", "t", "k"),
    ("s-u.png", "u", "4"),
    ("s-v.png", "v", "9"),
]

os.makedirs("error_images", exist_ok=True)

for filename, true_label, predicted in errors:

    source = os.path.join(test_folder, filename)
    image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    output_name = f"true_{true_label}_pred_{predicted}.png"
    output_path = os.path.join("error_images", output_name)

    cv2.imwrite(output_path, image)

    print(f"Saved: {output_path}")

print("\nDone. Check the error_images folder.")