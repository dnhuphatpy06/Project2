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

# check_don = IntVar()

print("""Chào mừng đến với ỨNG DỤNG NHẬN DIỆN CẢM XÚC TỪ GIỌNG NÓI!

Ban đầu có thể mô hình sẽ load hơi lâu, hãy đợi một chút!      

Hướng dẫn sơ bộ:
1. Nút 'Thoát' ở góc trên bên trái: Thoát ứng dụng.
2. Nút 'Ghi âm ngắn': Ghi âm âm thanh trực tiếp trong 2 giây và sau đó đưa ra dự đoán.
3. Nút 'Ghi âm dài': Ghi âm âm thanh trực tiếp cho đến khi bấm dừng.
3. Nút 'Upload file .wav': Tải file âm thanh .wav để mô hình đưa ra dự đoán.
4. Ô 'Chọn mô hình': Chọn mô hình mà bạn muốn dùng để dự đoán. Lưu ý: Hãy chọn đúng tên mô hình!

Chúc bạn có trải nghiệm tốt nhất!""")

cam_xuc = ('Giận dữ', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn bã')
tuong_ung = ('ANG','FEA', 'HAP', 'NEU', 'SAD')

def chay_thanh():
    progress_bar['value'] = 0
    progress_bar.grid(row=6, column=0, pady=20)

def extract_feature(X, sr):
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
    features = np.empty((0, 193))

    mfccs, chroma, mel, contrast, tonnetz = extract_feature(X, sr)

    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, ext_features])
    return np.array(features)

def predict(X, sr, modelpath):
    ts_features= parse_audio_files(X, sr)
    ts_features = np.array(ts_features, dtype=pd.Series)
    if modelpath != './model/keras_model.h5':
        model = pickle.load(open(modelpath, 'rb'))
        prediction = model.predict(ts_features)[0]
    else:
        ts_features = np.array(ts_features, dtype=np.float32)
        model = load_model(modelpath)
        prediction = model.predict(ts_features, verbose=0)
        predicted_class = np.argmax(prediction[0]) 
        prediction = tuong_ung[predicted_class]
    return prediction

modelpath = ['./model/ModelKNN.sav']

def predict_dai(path, modelpath):
    y, sr = librosa.load(path, sr=None)
    secs = librosa.get_duration(y=y, sr=sr)
    list_emotes = []
    if secs <= 1:
        list_emotes.append(predict(y, sr, modelpath))
    else:
        for i in range(int((secs - 1) // 0.5) + 3):
            cat = y[max(0, int(sr *( i * 0.5 - 1))): min(int((i * 0.5 + 1) * sr), int(secs * sr))]
            list_emotes.append(predict(cat, sr, modelpath))
    return list_emotes


def callback(indata, frames, time, status):
    # Lưu dữ liệu âm thanh vào danh sách recorded_data
    recorded_data.append(indata.copy())

def record_lien_tuc():
    global recorded_data
    recorded_data = []
    hien_thi_2.config(text='Đang ghi âm...')
    option.update()
    stream = sounddevice.InputStream(callback=callback, channels=2, samplerate=44100)
    stream.start()
    while not stop_event.is_set():
        pass
    stream.stop()
    audio_data = np.concatenate(recorded_data, axis=0)  # Kết hợp các mảng con lại

    write('./audioL.wav', 44100, audio_data)
    stream.close()
    hien_thi_2.config(text='Đang xử lý...')
    option.update()
    list_emo = predict_dai('./audioL.wav', modelpath[0])
    hien_thi_2.config(text='Hoàn tất!')
    option.update()
    y, sr = librosa.load('./audioL.wav')
    sounddevice.play(y, sr)
    chay_thanh()
    time.sleep(0.5)
    step = 100 / len(list_emo)
    for i in range(len(list_emo)):
        hien_du_doan(list_emo[i])
        progress_bar['value'] += step
        time.sleep(0.5)
    hien_thi_2.config(text='')
    option.update()
    
def record_it_giay():
    hien_thi_1.config(text='Bắt đầu')
    option.update()
    fs = 44100
    seconds= 2
    myrecording = sounddevice.rec(int(seconds*fs), samplerate = fs, channels = 2)
    sounddevice.wait()
    write('./audio.wav', fs, myrecording)
    hien_thi_1.config(text='Kết thúc')
    option.update()
    y, sr = librosa.load('./audio.wav', sr=None)
    return predict(y, sr, modelpath[0])

root = Tk()

root.title('NHẬN DIỆN CẢM XÚC TỪ GIỌNG NÓI')
root.minsize(height=500, width=500)

#Hang 0:
Button(root, text='Thoát', command=root.quit).grid(row=0, column=0,sticky='w')
### 1
Label(root, text='Chọn chế độ:',font=('cambria', 24), width=50).grid(row=1,column=0,pady=10)
### 2
stop_event = threading.Event()

option = Frame(root)
def record_ngan():
    xoa_hien_thi()
    hien_du_doan(record_it_giay())
    
def record_dai():
    xoa_hien_thi()
    global stop_event
    stop_event.clear()
    threading.Thread(target=record_lien_tuc, daemon=True).start()
    # time.sleep(4)
    nut_dung.grid(row=0, column=1, padx=10)

def xoa_hien_thi():
    duong_dan.config(text="")
    hien_thi_1.config(text="")
    hien_thi_2.config(text="")
    option.update()
    du_doan.config(text="")
    image = Image.open("./emotion/NON.png").resize((50,50))
    photo = ImageTk.PhotoImage(image=image)
    anh.image = photo
    anh.config(image=photo)
    dudoan.update()
    progress_bar.grid_forget()

def hien_du_doan(ABC):
    du_doan.config(text=cam_xuc[tuong_ung.index(ABC)])
    image = Image.open(f"./emotion/{ABC}.png").resize((50,50))
    photo = ImageTk.PhotoImage(image=image)
    anh.image = photo
    anh.config(image=photo)
    dudoan.update()
    
def stop_recording():
    """Kích hoạt cờ để dừng luồng."""
    hien_thi_2.config(text='')
    option.update()
    stop_event.set()
    nut_dung.grid_forget()

Button(option, text="Ghi âm ngắn", width=30, command=record_ngan).grid(row=0, column=0, padx=10)
Button(option, text='Ghi âm dài', width=30, command=record_dai).grid(row=0, column=1, padx=10)
nut_dung = Button(option, text='Dừng', width=30, command=stop_recording)
def upload_file():
    xoa_hien_thi()
    # Mở hộp thoại để chọn file
    file_path = filedialog.askopenfilename(title="Chọn file", filetypes=[("WAV Files", "*.wav")])
    if file_path:
        duong_dan.config(text=f"File đã chọn: {file_path.split('/')[-1]}")
        option.update()
        list_emo = predict_dai(file_path, modelpath[0])
        y, sr = librosa.load(file_path)
        sounddevice.play(y, sr)
        chay_thanh()
        step = 100 / len(list_emo)
        time.sleep(0.5)
        for i in range(len(list_emo)):
            hien_du_doan(list_emo[i])
            progress_bar['value'] += step
            time.sleep(0.5)
        
Button(option, text='Upload file .wav', width=30, command=upload_file).grid(row=0, column=2)
option.grid(row=2, column=0, pady=10)
# bien_check = [0]
# def check_1_lan():
#     if check_don.get():
#         bien_check[0] = 1
#     else:
#         bien_check[0] = 0
# du_doan_don = Checkbutton(option, text='Dự đoán 1 lần', variable=check_don, command=check_1_lan)
#
hien_thi_1 = Label(option, text="",font=('cambria', 10))
hien_thi_1.grid(row=1, column=0,pady=5)
hien_thi_2 = Label(option, text="",font=('cambria', 10))
hien_thi_2.grid(row=1, column=1,pady=5)
duong_dan = Label(option, text="",font=('cambria', 10))
duong_dan.grid(row=1, column=2,pady=5)
### 4
chon_mo_hinh = Frame(root)
Label(chon_mo_hinh, text="Chọn mô hình:").grid(row=0, column=0, padx=25)
combo = Combobox(chon_mo_hinh)
combo["values"] = ("KNN", "Decision Tree", "Extra Tree", 'Neural Network')
combo.current(0)
combo.grid(row=0, column=1,pady=10)

model_path = ('./model/ModelKNN.sav', './model/DecisionTree.sav', './model/ExtraTree.sav', './model/keras_model.h5')
def lay_mo_hinh():
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
Button(chon_mo_hinh, text="Chọn", command=lay_mo_hinh).grid(row=0,column=2)
chon_mo_hinh.grid(row=4, column=0, pady=10)
### 5
Label(root, text='Dự đoán:',fg='red', font=('cambria', 16), width=40).grid(row=5,column=0,pady=10)
### 6
progress_bar = Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
progress_bar.grid_forget()
### 7
dudoan = Frame(root)
anh = Label(dudoan)
anh.grid(row=1, column=0, padx=50)
du_doan = Label(dudoan, text='',font=('cambria', 15))
du_doan.grid(row=1, column=1)
dudoan.grid(row=7, column=0)
###

root.mainloop()