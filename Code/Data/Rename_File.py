import os

def rename_wav_files(directory, index_fist):
    """
    Đổi tên tất cả các tệp .wav trong thư mục theo định dạng mới, dựa trên nhãn từ tên tệp gốc.
    
    Tham số:
        directory (str): Đường dẫn đến thư mục chứa các tệp âm thanh (.wav).
        index_fist (int): Chỉ số bắt đầu để đổi tên tệp, mỗi tệp sẽ được đánh số từ chỉ số này.
    
    Trả về:
        None: Hàm không trả về giá trị gì, chỉ thực hiện việc đổi tên tệp trong thư mục.
        
    Lưu ý:
        - Tên tệp sẽ được đổi theo định dạng: "{index}_{label}.wav"
        - Hàm này sử dụng hàm `extract_labels` để lấy nhãn từ tên tệp gốc.
    """
    if not os.path.exists(directory):
        print("Thư mục không tồn tại!")
        return
    for filename in os.listdir(directory):
        if filename.lower().endswith(".wav"):
            old_file_path = os.path.join(directory, filename)
            label = extract_labels(filename)
            new_filename = f"{index_fist}_{label}.wav"
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Đổi tên tệp: {filename} -> {new_filename}")
            index_fist += 1

def extract_labels(file_name, index_data = None):
    """
    Trích xuất nhãn từ tên tệp âm thanh dựa trên quy tắc đặt tên của từng bộ dữ liệu.
    
    Tham số:
        file_name (str): Tên tệp âm thanh (ví dụ: "file_HAP_01.wav").
        index_data (int, optional): Chỉ số xác định bộ dữ liệu, có thể là:
            - 1: Bộ dữ liệu đầu tiên, nhãn theo kiểu "01", "03", v.v.
            - 2: Bộ dữ liệu thứ hai, nhãn theo kiểu "happy", "sad", v.v.
            - 3: Bộ dữ liệu thứ ba, nhãn theo kiểu "HAP", "FEA", v.v.
    
    Trả về:
        str: Nhãn tương ứng với tên tệp 
        
    Lưu ý:
        - Hàm này sử dụng `index_data` để xác định cách trích xuất nhãn từ tên tệp tuỳ thuộc vào các bộ dữ liệu.
        - Hàm sẽ trả về nhãn tương ứng dựa trên định dạng của tệp.
    """
    if index_data == 1:
        label = file_name.split("-")[-5]
        if label == "01":
            return "NEU"
        elif label == "03":
            return "HAP"
        elif label == "04":
            return "SAD"
        elif label == "05":
            return "ANG"
        elif label == "06":
            return "FEA"
        
    elif index_data == 2:
        label = file_name.split("_")[-1].split(".")[0]
        if label == "sad":
            return "SAD"
        elif label == "happy":
            return "HAP"
        elif label == "fear":
            return "FEA"
        elif label == "angry":
            return "ANG"
        elif label == "neutral":
            return "NEU"
        
    elif index_data == 3:
        label = file_name.split["_"][-2]
        if label == "HAP":
            return "HAP"
        elif label == "FEA":
            return "FEA"
        elif label == "ANG":
            return "ANG"
        elif label == "NEU":
            return "NEU"
        elif label == "SAD":
            return "SAD"
    
# Khai báo đường dẫn và các chỉ số bắt đầu
input_directory = ""
index_fist = 0
index_data = 3

# Thực hiện quá trình đổi tên
rename_wav_files(input_directory)
