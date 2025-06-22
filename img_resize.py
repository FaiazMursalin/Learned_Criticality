from PIL import Image, ImageOps

def pad_to_201(image_path, output_path):
    img = Image.open(image_path)
    padded_img = ImageOps.pad(img, (201, 201), color=(0, 0, 0))  # Pads with black
    padded_img.save(output_path)


input_imgs = ["home.png", "Office.png", "library.png"]

for img in input_imgs:
    pad_to_201(img, f"{img.split('.')[0]}_resized.png")