from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkinter.ttk import Combobox
import pandas as pd
import _pickle as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model
import sounddevice
from scipy.io.wavfile import write
import librosa
import librosa.display
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("""Chào mừng đến với ỨNG DỤNG NHẬN DIỆN CẢM XÚC TỪ GIỌNG NÓI!

Ban đầu có thể mô hình sẽ load hơi lâu, hãy đợi một chút!      

Hướng dẫn sơ bộ:
1. Nút 'Thoát' ở góc trên bên trái: Thoát ứng dụng.
2. Nút 'Ghi âm ngắn': Ghi âm âm thanh trực tiếp trong 2 giây và sau đó đưa ra dự đoán.
3. Nút 'Ghi âm dài': Ghi âm âm thanh trực tiếp cho đến khi bấm dừng.
4. Nút 'Upload file .wav': Tải file âm thanh .wav để mô hình đưa ra dự đoán.
5. Nút 'Dự đoán 1 lần': Điều chỉnh việc file âm thanh được dự đoán 1 lần hay được chia ra thành các đoạn để dự đoán.
6. Ô 'Chọn mô hình': Chọn mô hình mà bạn muốn dùng để dự đoán. Lưu ý: Hãy chọn đúng tên mô hình!

Chúc bạn có trải nghiệm tốt nhất!""")

#Đường dẫn đến folder APP:
path_to_APP = os.path.dirname(__file__)

#Các tuple lưu những giá trị cố định:
cam_xuc = ('Giận dữ', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn bã')
CAM_XUC = ('ANG', 'FEA', 'HAP', 'NEU', 'SAD')
model_path = (f'{path_to_APP}/model/ModelKNN.sav', f'{path_to_APP}/model/DecisionTree.sav', f'{path_to_APP}/model/ExtraTree.sav', f'{path_to_APP}/model/keras_model.h5')


#Biến chung cho cả chương trình: (Biến được lưu dưới dạng danh sách để có thể truy cập trực tiếp trong hàm)
modelpath = [f'{path_to_APP}/model/ModelKNN.sav'] #Đường dẫn đến mô hình

#Các hàm xử lý âm thanh
"""Các tham số chung:
    X: Một mảng NumPy chứa các giá trị biên độ âm thanh (audio samples).
    sr: Giá trị tần số mẫu (sampling rate) của âm thanh sau khi tải.
"""
def extract_feature(X, sr):
    """
    Trích xuất đặc trưng âm thanh từ một tệp âm thanh.
    Hàm này trích xuất các đặc trưng âm thanh từ tệp âm thanh đã cho, bao gồm:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Chroma
    - Mel spectrogram
    - Spectral contrast
    - Tonnetz

    Tham số: X, sr
    
    Trả về:
    tuple: Một tuple chứa các mảng đặc trưng sau:
        - mfccs (np.array): MFCC trung bình của âm thanh.
        - chroma (np.array): Đặc trưng Chroma trung bình của âm thanh.
        - mel (np.array): Mel spectrogram trung bình của âm thanh.
        - contrast (np.array): Spectral contrast trung bình của âm thanh.
        - tonnetz (np.array): Tonnetz trung bình của âm thanh.
    """
    X, sample_rate = X, sr
    stft = np.abs(librosa.stft(X, n_fft=min(len(X), 1024)))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(X, sr):
    """
    Duyệt qua mảng Numpy âm thanh, sau đó trích xuất đặc trưng và nhãn.
    Hàm này trích xuất các đặc trưng từ mỗi tệp bằng hàm `extract_feature()`.

    Tham số: X, sr

    Trả về:
    tuple: Một tuple chứa:
        - features (np.array): Mảng numpy chứa các đặc trưng đã trích xuất cho tất cả các tệp âm thanh.
    """
    features = np.empty((0, 193))

    mfccs, chroma, mel, contrast, tonnetz = extract_feature(X, sr)

    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])
    return np.array(features)


#Các hàm toán học:
def max_appear(list_emo):
    """
    Hàm này trả về phần tử xuất hiện nhiều lần nhất trong 1 danh sách.
    Nếu có nhiều hơn 1 phần tử có tần suất xuất hiện nhiều nhất, hàm sẽ trả về phần tử xuất hiện đầu tiên.
    """
    dic = {}
    for i in list_emo:
        dic[i] = dic.get(i,0) + 1
    return max(dic.items(), key = lambda a: a[1])[0]


#Các hàm dự đoán:
def predict(X, sr):
    """
    Hàm sẽ dự đoán cảm xúc từ âm thanh, thông qua mô hình ở đường dẫn 'modelpath'

    Tham số:
        X, sr: Như đã nói ở trên
        modelpath: Đường dẫn đến mô hình

    Trả về:
        prediction: Kết quả dự đoán. Vd: HAP
    """
    ts_features= parse_audio_files(X, sr)
    ts_features = np.array(ts_features, dtype=pd.Series)
    if modelpath[0] != f'{path_to_APP}/model/keras_model.h5':
        model = pickle.load(open(modelpath[0], 'rb'))
        prediction = model.predict(ts_features)[0]
    else:
        ts_features = np.array(ts_features, dtype=np.float32)
        model = load_model(modelpath[0])
        prediction = model.predict(ts_features, verbose=0)
        predicted_class = np.argmax(prediction[0]) 
        prediction = CAM_XUC[predicted_class]
    return prediction

def predict_true(X, sr):
    """
    Là hàm dự đoán chính thức của chương trình.
    Hàm sử dụng hàm 'predict' ở phía trên để đưa ra dự đoán trên 1 hay nhiều đoạn âm thanh

    Tham số: X, sr.

    Trar về:
        list_emotes: Danh sách các cảm xúc được dự đoán tương ứng với các đoạn âm thanh
    """
    secs = librosa.get_duration(y=X, sr=sr)
    list_emotes = []
    if secs <= 1:
        list_emotes.append(predict(X, sr))
    else:
        for i in range(int((secs - 1) // 0.5) + 3):
            cat = X[max(0, int(sr *( i * 0.5 - 1))): min(int((i * 0.5 + 1) * sr), int(secs * sr))]
            list_emotes.append(predict(cat, sr))
    return list_emotes


#Các hàm liên quan đến hiện thị trên giao diện:
def chay_thanh():
    """
    Hàm hiện thanh chạy thời gian
    """
    progress_bar['value'] = 0
    progress_bar.grid(row=7, column=0, pady=20)

def xoa_hien_thi():
    """
    Xóa hết các hiển thị về dự đoán và trạng thái các nút, đưa giao diện về lại ban đầu.
    """
    hang_7.grid_forget()
    duong_dan.config(text="")
    hien_thi_1.config(text="")
    option.update()
    du_doan.config(text="")
    image = Image.open(f'{path_to_APP}/emotion/NON.png').resize((50,50))
    photo = ImageTk.PhotoImage(image=image)
    anh.image = photo
    anh.config(image=photo)
    dudoan.update()
    progress_bar.grid_forget()

def hien_du_doan(ABC):
    """
    Hàm hiện dự đoán lên Label 'dudoan'

    Tham số:
        ABC: Dự đoán ở dạng 'HAP',...
    """
    du_doan.config(text=cam_xuc[CAM_XUC.index(ABC)])
    image = Image.open(f"{path_to_APP}/emotion/{ABC}.png").resize((50,50))
    photo = ImageTk.PhotoImage(image=image)
    anh.image = photo
    anh.config(image=photo)
    dudoan.update()

def hien_thi_du_doan_lien_tuc(path):
    """
    Hàm hiển thị liên tục các dự đoán trên các đoạn âm thanh nhỏ
    Sau đó trả về kết quả xuất hiện nhiều lần nhất: Kết quả tổng quát cuối cùng

    Tham số:
        path: Đường dẫn đến tệp âm thanh cần dự đoán
    """
    X, sr = librosa.load(path)
    list_emo = predict_true(X, sr)
    if not check_don.get():
        sounddevice.play(X, sr)
        chay_thanh()
        step = 100 / (len(list_emo) - 1)
        for i in range(len(list_emo)):
            hien_du_doan(list_emo[i])
            time.sleep(0.5)
            if progress_bar['value'] < 100:
                progress_bar['value'] += step
        time.sleep(0.5)
        progress_bar.grid_forget()
        hien_chu_hang_7.config(text='Kết quả tổng quát cuối cùng:',font=('cambria', 15))
        hien_chu_hang_7.grid(row=0, column=0)
        hang_7.grid(row=7, column=0,pady=10)
        hien_du_doan(max_appear(list_emo))
    else:
        hien_du_doan(max_appear(list_emo))


#Các hàm ghi âm:
def callback(indata, frames, time, status):
    """
    Hàm bổ trợ cho quá trình ghi âm liên tục
    """
    recorded_data.append(indata.copy())

def record():
    """
    Hàm sẽ thực hiện ghi âm cho đến khi nhận được tín hiệu dừng.
    Sau đó thực hiện dự đoán liên tục và hiển thị lên màn hình.
    
    Vị trí hàm được gọi: Hàm 'start_record'.
    Trạng thái: Hàm được chạy trên luồng phụ.

    Trả về:
        - Tệp audio.wav lưu âm thanh sau khi được ghi âm.
        - Báo lỗi nếu không kết nối được với âm thanh.
    """
    try:
        global recorded_data
        recorded_data = []
        stream = sounddevice.InputStream(callback=callback, channels=2, samplerate=44100)
        stream.start()
        nut_dung.grid(row=0, column=1, padx=10)
        hien_thi_1.config(text='Đang ghi âm...')
        option.update()
        while not stop_event.is_set():
            pass
        stream.stop()
        audio_data = np.concatenate(recorded_data, axis=0)  # Kết hợp các mảng con lại

        write(f'{path_to_APP}/audio.wav', 44100, audio_data)
        stream.close()
        hien_thi_1.config(text='Đang xử lý...')
        option.update()
        hien_thi_1.config(text='Hoàn tất!')
        option.update()
        hien_thi_du_doan_lien_tuc(f'{path_to_APP}/audio.wav')
    except sounddevice.PortAudioError:
        hien_thi_1.config(text='Không thể ghi âm!')
        option.update()

"""Phần chương trình chính"""

# Gọi giao diện:
root = Tk()
root.title('NHẬN DIỆN CẢM XÚC TỪ GIỌNG NÓI') #Tên của giao diện
root.geometry("925x500") #Kích thước giao diện
root.resizable(False, False) #Cố định kích thước

#Biến của Tkinter:
check_don = IntVar() #Biến biểu thị có hiển thị quá trình dự đoán hay không.

"""Code từng hàng"""
### Hàng 0:
Button(root, text='Thoát', command=root.quit).grid(row=0, column=0,sticky='w') #Nút thoát

### Hàng 1:
Label(root, text='Chọn chế độ:',font=('cambria', 24), width=50).grid(row=1,column=0,pady=10) #Hiển thị 'Chọn chế đố'

### Hàng 2:
stop_event = threading.Event() #Biến giúp dừng luồng phụ chạy hàm record

option = Frame(root) #Tạo Frame Option trên giao diện chính
    
def start_record():
    """
    Hàm được gọi khi kích vào nút 'Ghi âm'
    Bắt đầu ghi âm và đồng thời hiện nút 'Dừng' lên
    """
    xoa_hien_thi()
    global stop_event
    stop_event.clear()
    try:
        threading.Thread(target=record, daemon=True).start()
    except:
        pass
    
def stop_recording():
    """
    Hàm được gọi ghi kích vào nút 'Dừng'
    Có tác dụng dừng quá trình ghi âm, trả lại nút 'Ghi âm'
    """
    hien_thi_1.config(text='')
    option.update()
    stop_event.set()
    nut_dung.grid_forget()

Button(option, text='Ghi âm', width=30, command=start_record).grid(row=0, column=1, padx=10) #Tạo và hiển thị nút ghi âm
nut_dung = Button(option, text='Dừng', width=30, command=stop_recording) #Khởi tạo nút dừng

def upload_file():
    """
    Hàm mở giao diện chọn file lên để người dùng chọn tệp âm thanh cần dự đoán
    Sau đó thực hiện dự đoán liên tục và hiển thị lên màn hình.
    """
    xoa_hien_thi()
    # Mở hộp thoại để chọn file
    file_path = filedialog.askopenfilename(title="Chọn file", filetypes=[("WAV Files", "*.wav")])
    if file_path:
        duong_dan.config(text=f"File đã chọn: {file_path.split('/')[-1]}")
        option.update()
        hien_thi_du_doan_lien_tuc(file_path)
        
Button(option, text='Upload file .wav', width=30, command=upload_file).grid(row=0, column=2) #Tạo và hiển thị nút 'Upload file'

hien_thi_1 = Label(option, text="",font=('cambria', 10)) #Label hiển thị bên dưới nút 'Ghi âm'
hien_thi_1.grid(row=1, column=1,pady=5)
duong_dan = Label(option, text="",font=('cambria', 10)) #Label hiển thị bên dưới nút 'Upload file'
duong_dan.grid(row=1, column=2,pady=5)

option.grid(row=2, column=0, pady=10) #Hiển thị Frame 'opiton' lên.

### Hàng 4
chon_mo_hinh = Frame(root) #Khởi tạo Frame 'chon_mo_hinh'

#Tạo ô chọn mô hình
Label(chon_mo_hinh, text="Chọn mô hình:").grid(row=0, column=0, padx=25)
combo = Combobox(chon_mo_hinh)
combo["values"] = ("KNN", "Decision Tree", "Extra Tree", 'Neural Network')
combo.current(0)
combo.grid(row=0, column=1,pady=10)

def lay_mo_hinh():
    """
    Hàm có tác dụng thay đổi biến chung 'modelpath[0]' tương ứng với mô hình mà ta chọn
    """
    if combo.get() in ("KNN", "Decision Tree", "Extra Tree", 'Neural Network'):
        mo_hinh = combo.get()
        messagebox.showinfo("Thông báo", f"Bạn đã chuyển sang mô hình {mo_hinh}")
        if mo_hinh == 'KNN':
            modelpath[0] = model_path[0]
        elif mo_hinh == "Decision Tree":
            modelpath[0] = model_path[1]
        elif mo_hinh == 'Extra Tree':
            modelpath[0] = model_path[2]
        elif mo_hinh == 'Neural Network':
            modelpath[0] = model_path[3]
    else:
        messagebox.showinfo("Thông báo", "Mô hình không hợp lệ.")

Button(chon_mo_hinh, text="Chọn", command=lay_mo_hinh).grid(row=0,column=2) #Khởi tạo và hiện thị nút 'Chọn'

chon_mo_hinh.grid(row=4, column=0, pady=10) #Hiển thị Frame 'chon_mo_hinh'

### Hàng 5
du_doan_don = Checkbutton(root, text='Chỉ hiện kết quả cuối', variable=check_don)
du_doan_don.grid(row=5, column=0, pady=5) #Tạo và hiển thị CheckBox 'Chỉ hiện kết quả cuối'

### Hàng 6
Label(root, text='Dự đoán:',fg='red', font=('cambria', 16), width=40).grid(row=6,column=0,pady=10) #Tạo và hiển thị Label 'Dự đoán'

### Hàng 7
progress_bar = Progressbar(root, orient="horizontal", length=300, mode="indeterminate") #Tạo thanh chạy
hang_7 = Frame(root)
hien_chu_hang_7 = Label(hang_7, text='') #Có tác dụng thay thế vị trí thanh chạy để hiện 'Kết quả dự đoán cuối cùng'
hang_7.grid(row=7, column=0, pady=5) #Hiển thị Frame 'hang_7'

### Hàng 8
dudoan = Frame(root)
anh = Label(dudoan)
anh.grid(row=1, column=0, padx=50) #Label 'anh' sẽ hiển thị Icon cảm xúc
du_doan = Label(dudoan, text='',font=('cambria', 15))
du_doan.grid(row=1, column=1) #Label 'du_doan' sẽ hiển thị kết quả dự đoán ở dạng 'Hạnh phúc'
dudoan.grid(row=8, column=0) 

"""Chạy chương trình"""
root.mainloop()