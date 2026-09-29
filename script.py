import os
import shutil
import cv2

# Шляхи до директорій датасету IRSTD1k
MASKS_DIR = "C:/filter-script/IRSTD1k_Label"           # Папка з масками
IMAGES_DIR = "C:/filter-script/IRSTD1k_Img"             # Папка з ІЧ-зображеннями

# Папки для валідних даних
OUTPUT_MASKS_DIR = "C:/filter-script/filtered_masks" 
OUTPUT_IMAGES_DIR = "C:/filter-script/filtered_images"

# Папки для даних, що НЕ пройшли перевірку
REJECTED_MASKS_DIR = "C:/filter-script/rejected_masks"
REJECTED_IMAGES_DIR = "C:/filter-script/rejected_images"

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def copy_paired_image(images_dir, output_dir, base_name, orig_ext, size_suffix):
    """Допоміжна функція для пошуку та копіювання відповідного зображення з новим ім'ям."""
    img_path = os.path.join(images_dir, base_name + orig_ext)
    
    if os.path.exists(img_path):
        new_img_name = f"{base_name}{size_suffix}{orig_ext}"
        shutil.copy(img_path, os.path.join(output_dir, new_img_name))
    else:
        # Якщо розширення зображення відрізняється від маски
        for ext in VALID_EXTENSIONS:
            alt_img_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(alt_img_path):
                new_img_name = f"{base_name}{size_suffix}{ext}"
                shutil.copy(alt_img_path, os.path.join(output_dir, new_img_name))
                break


def filter_by_mask_objects(
    masks_dir, 
    output_masks_dir, 
    rejected_masks_dir,
    images_dir=None, 
    output_images_dir=None, 
    rejected_images_dir=None,
    min_size=(5, 5), 
    max_size=(30, 30)
):
    # Створюємо директорії для виходу
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(rejected_masks_dir, exist_ok=True)
    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)
    if rejected_images_dir:
        os.makedirs(rejected_images_dir, exist_ok=True)

    matched_count = 0
    skipped_count = 0

    for filename in os.listdir(masks_dir):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        mask_path = os.path.join(masks_dir, filename)
        
        # Завантажуємо маску в одноканальному режимі (градації сірого)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Не вдалося прочитати маску: {filename}")
            continue

        # В бінарній масці білі об'єкти > 127
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Знаходимо зовнішні контури всіх білих плям
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_valid_object = False
        valid_size = None
        max_invalid_size = (0, 0)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Перевіряємо умови розміру
            if (min_size[0] <= w <= max_size[0]) and (min_size[1] <= h <= max_size[1]):
                has_valid_object = True
                valid_size = (w, h)
                break  # Достатньо одного об'єкта потрібного розміру
            else:
                # Фіксуємо розмір найбільшого об'єкта для відсіяних масок
                if (w * h) > (max_invalid_size[0] * max_invalid_size[1]):
                    max_invalid_size = (w, h)

        base_name, ext = os.path.splitext(filename)

        if has_valid_object:
            size_suffix = f"_W{valid_size[0]}xH{valid_size[1]}"
            new_mask_name = f"{base_name}{size_suffix}{ext}"

            # Зберігаємо валідну маску з розміром у назві
            shutil.copy(mask_path, os.path.join(output_masks_dir, new_mask_name))

            # Зберігаємо відповідну картинку
            if images_dir and output_images_dir:
                copy_paired_image(images_dir, output_images_dir, base_name, ext, size_suffix)

            matched_count += 1
        else:
            size_suffix = f"_W{max_invalid_size[0]}xH{max_invalid_size[1]}"
            new_mask_name = f"{base_name}{size_suffix}{ext}"

            # Зберігаємо невалідну маску в папку відхилених
            shutil.copy(mask_path, os.path.join(rejected_masks_dir, new_mask_name))

            # Зберігаємо відповідну невалідну картинку
            if images_dir and rejected_images_dir:
                copy_paired_image(images_dir, rejected_images_dir, base_name, ext, size_suffix)

            skipped_count += 1

    print("Фільтрація завершена!")
    print(f"Отобрано кадрів з потрібним розміром ({min_size[0]}x{min_size[1]} - {max_size[0]}x{max_size[1]}): {matched_count}")
    print(f"Відсіяно кадрів: {skipped_count}")


if __name__ == "__main__":
    filter_by_mask_objects(
        masks_dir=MASKS_DIR,
        output_masks_dir=OUTPUT_MASKS_DIR,
        rejected_masks_dir=REJECTED_MASKS_DIR,
        images_dir=IMAGES_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        rejected_images_dir=REJECTED_IMAGES_DIR,
        min_size=(5, 5),
        max_size=(30, 30)
    )