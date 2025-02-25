def convert_to_rgb(image):
    if isinstance(image, np.ndarray):  
        image=Image.fromarray(image)
    if image.mode in ("RGBA", "P"):  
        image = image.convert("RGB")
    return np.array(image)
