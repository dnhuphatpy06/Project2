import librosa
import numpy as np
import soundfile as sf
import os

def shift_audio(audio, sr, n_steps=0):
    """
    Chỉ thay đổi cao độ của âm thanh.
    Args:
        audio (numpy.ndarray): Mảng tín hiệu âm thanh.
        sr (int): Tần số lấy mẫu.
        n_steps (int): Số cung thay đổi cao độ
    Returns:
        numpy.ndarray: Âm thanh sau khi thay đổi cao độ.
    """
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return shifted_audio

def shift_and_add_noise(audio, sr, n_steps=0, noise_level=0):
    """
    Kết hợp thay đổi cao độ và thêm nhiễu.
    Args:
        audio (numpy.ndarray): Mảng tín hiệu âm thanh.
        sr (int): Tần số lấy mẫu.
        n_steps (int): Số cung thay đổi cao độ 
        noise_level (float): Mức độ nhiễu
    Returns:
        numpy.ndarray: Âm thanh sau khi biến đổi.
    """
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    noise = np.random.normal(0, noise_level, shifted_audio.shape)
    return shifted_audio + noise

def augment_audio(audio, sr, n_steps=0, noise_level=0):
    """
    Kết hợp thay đổi cao độ và thêm nhiễu.
    Args:
        audio (numpy.ndarray): Mảng tín hiệu âm thanh.
        sr (int): Tần số lấy mẫu.
        n_steps (int): Số cung thay đổi cao độ 
        noise_level (float): Mức độ nhiễu 
    Returns:
        numpy.ndarray: Âm thanh sau khi biến đổi.
    """
    shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    noise = np.random.normal(0, noise_level, shifted_audio.shape)
    return shifted_audio + noise

def add_reverb(audio, sr, delay=0, decay=0):
    """
    Thêm hiệu ứng tiếng vang vào âm thanh.
    Args:
        audio (numpy.ndarray): Mảng tín hiệu âm thanh.
        sr (int): Tần số lấy mẫu.
        delay (float): Thời gian trễ giữa các tiếng vang (tính bằng giây).
        decay (float): Mức độ giảm âm lượng của tiếng vang (0.0 đến 1.0).
    Returns:
        numpy.ndarray: Âm thanh sau khi thêm tiếng vang.
    """
    delay_samples = int(delay * sr)
    reverb_audio = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb_audio[i] += decay * audio[i - delay_samples]
    return reverb_audio

def process_files_in_directory(input_directory, output_directory, n_steps_values=(4, -2, 3), noise_levels=(0.005, 0.005, 0.005)):
    """
    Duyệt qua tất cả các file trong thư mục đầu vào và ghi các file kết quả vào thư mục đầu ra.
    
    Args:
        input_directory (str): Thư mục chứa các file âm thanh đầu vào.
        output_directory (str): Thư mục lưu tất cả các kết quả (cả bốn phép biến đổi).
        n_steps_values (tuple): Các giá trị n_steps cho các hàm biến đổi (default: (4, -2, 3)).
        noise_levels (tuple): Các giá trị noise_level cho các hàm biến đổi (default: (0.005, 0.005, 0.005)).
    """
    os.makedirs(output_directory, exist_ok=True)
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        if os.path.isfile(file_path):
            audio, sr = librosa.load(file_path, sr=None)
            
            result_1 = shift_audio(audio, sr, n_steps=4)
            sf.write(os.path.join(output_directory, f"shift_{file_name}"), result_1, sr)
            
            result_2 = shift_and_add_noise(audio, sr, n_steps=-2, noise_level=0.005)
            sf.write(os.path.join(output_directory, f"shift_and_add_noise_{file_name}"), result_2, sr)
            
            result_3 = augment_audio(audio, sr, n_steps=3, noise_level=0.005)
            sf.write(os.path.join(output_directory, f"augment_{file_name}"), result_3, sr)
            
            result_4 = add_reverb(audio, sr, delay=0.08, decay=0.5)
            sf.write(os.path.join(output_directory, f"add_reverb_{file_name}"), result_4, sr)
            
            print(f"Đã xử lý file: {file_name}")

# Khai báo được dẫn
input_directory = ""
output_directory = ""

# Thực hiện quá trình tăng cường
process_files_in_directory(input_directory, output_directory)
