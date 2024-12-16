import glob
import librosa
import librosa.display
import numpy as np

def extract_feature(file_name):
    """
    Trích xuất đặc trưng âm thanh từ một tệp âm thanh.
    Hàm này trích xuất các đặc trưng âm thanh từ tệp âm thanh đã cho, bao gồm:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Chroma
    - Mel spectrogram
    - Spectral contrast
    - Tonnetz

    Tham số:
    file_name (str): Đường dẫn đến tệp âm thanh.

    Trả về:
    tuple: Một tuple chứa các mảng đặc trưng sau:
        - mfccs (np.array): MFCC trung bình của âm thanh.
        - chroma (np.array): Đặc trưng Chroma trung bình của âm thanh.
        - mel (np.array): Mel spectrogram trung bình của âm thanh.
        - contrast (np.array): Spectral contrast trung bình của âm thanh.
        - tonnetz (np.array): Tonnetz trung bình của âm thanh.
    """
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def extract_labels(file_name):
    """
    Trích xuất nhãn từ tên tệp âm thanh.
    Nhãn được giả định là phần tên tệp nằm sau dấu gạch dưới cuối cùng và trước phần mở rộng tệp.
    
    Tham số:
    file_name (str): Tên tệp âm thanh (bao gồm đường dẫn).

    Trả về:
    str: Nhãn được trích xuất từ tên tệp.
    """
    label = file_name.split("_")[-1].split(".")[0]
    return label

def parse_audio_files(path):
    """
    Duyệt qua các tệp âm thanh trong thư mục và trích xuất đặc trưng và nhãn.
    Hàm này duyệt qua tất cả các tệp `.wav` trong thư mục đã cho, trích xuất các đặc trưng từ mỗi tệp 
    bằng hàm `extract_feature()` và trích xuất nhãn tương ứng bằng hàm `extract_labels()`.

    Tham số:
    path (str): Đường dẫn đến thư mục chứa các tệp âm thanh `.wav`.

    Trả về:
    tuple: Một tuple chứa:
        - features (np.array): Mảng numpy chứa các đặc trưng đã trích xuất cho tất cả các tệp âm thanh.
        - labels (np.array): Mảng numpy chứa các nhãn tương ứng cho mỗi tệp âm thanh.
    """
    features_list = []  
    labels_list = []   
    for fn in glob.glob(path  + "/*.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features_list.append(ext_features)
        label = extract_labels(fn)
        labels_list.append(label)
    features = np.array(features_list)
    labels = np.array(labels_list)
    return features, labels

def shuffle_data(features, labels):
    """
    Xáo trộn dữ liệu đặc trưng và nhãn một cách ngẫu nhiên.
    Hàm này sử dụng một phép hoán vị ngẫu nhiên các chỉ số để xáo trộn thứ tự của cả đặc trưng và nhãn đồng thời.
    Điều này hữu ích để đảm bảo rằng dữ liệu được xáo trộn trước khi huấn luyện mô hình học máy.

    Tham số:
    features (np.array): Mảng đặc trưng cần xáo trộn.
    labels (np.array): Mảng nhãn cần xáo trộn.

    Trả về:
    tuple: Một tuple chứa:
        - shuffled_features (np.array): Các đặc trưng đã được xáo trộn.
        - shuffled_labels (np.array): Các nhãn đã được xáo trộn.
    """
    indices = np.random.permutation(len(features))
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]
    return shuffled_features, shuffled_labels

# Khai báo đường dẫn
directory_path = ""
features_path = "" 
labels_path = ""

# Trích xuất và xáo trộn dữ liệu
features, labels = parse_audio_files(directory_path)
features, labels = shuffle_data(features, labels)

# Kiểm tra dữ liệu
print(features.shape)
print(labels.shape)
print(features[:5])
print(labels[:5])

# Lưu dữ liệu
np.save(features_path, features)
np.save(labels_path, labels)



