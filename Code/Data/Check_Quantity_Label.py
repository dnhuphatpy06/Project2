import os

def print_labels_in_directory(directory):
    """
    Liệt kê các file trong thư mục và in ra số lần xuất hiện của các nhãn trong tên file.

    Args:
        directory (str): Đường dẫn đến thư mục chứa các file âm thanh.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} không phải là thư mục hợp lệ.")
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    label_count = {}
    
    for file in files:
        label = file.split("_")[-1].split(".")[0]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    for label, count in label_count.items():
        print(f"Nhãn '{label}' xuất hiện {count} lần.")

# Ví dụ sử dụng:
directory_path = "" 
print_labels_in_directory(directory_path)