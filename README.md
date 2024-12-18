# Trường Đại học Khoa học Tự nhiên - ĐHQG TPHCM
## Khoa Công nghệ Thông tin. Ngành Trí tuệ Nhân tạo.
## Môn: Thực hành - Giới thiệu ngành Trí tuệ Nhân tạo.
### Project 2: Đề tài: Nhận diện cảm xúc từ giọng nói - Emotion Detection through Speech

### Giới thiệu:
Trong thời đại cách mạng công nghiệp 4.0 ngày nay, ứng dụng của trí tuệ nhân tạo AI đóng vai quan trọng và mang lại nhiều giá trị hiệu quả trong đa dạng lĩnh vực. Phát hiện cảm xúc từ giọng nói Emotion Detection from Speech là một lĩnh vực nghiên cứu đầy triển vọng trong ứng dụng trí tuệ nhân tạo vào đời sống.

### Các thành viên thực hiện dự án:
| STT | Tên | MSSV |
| :----- | :---------- | :-------------- |
| 1 | Đinh Như Phát | 24122002 |
| 2 | Võ Lê Gia Huy | 24122004 |
| 3 | Từ Văn Khôi | 24122019 |
| 4 | Lê Bảo Phúc | 24122021 |
| 5 | Trần Xuân Bách | 24122028 |
| 6 | Trần Lê Minh Đức | 24122031 |

### Dữ liệu:
Dữ liệu là các tệp âm thanh với các cảm xúc khác nhau, được chúng tôi thu thập từ nhiều nguồn. Một số nguồn như:
[Nguồn 1](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [Nguồn 2](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess), [Nguồn 3](https://www.kaggle.com/datasets/ejlok1/cremad).

Các nguồn được chúng tôi tải về, điều chỉnh cho phù hợp và tạo được nhiều dữ liệu huấn luyện hơn. Kết quả là 23750 mẫu âm thanh được lưu ở [Dữ liệu](https://drive.google.com/file/d/1cPrOLobqJcs_wTEpcE4cwGnnx5R6-9uo/view). Ngoài ra, để tiện cho quá trình huấn luyện mô hình, chúng tôi còn lưu những dữ liệu trên dưới dạng tệp **_.npy_** ở [Data](https://github.com/dnhuphatpy06/Project2/tree/main/Data).

### Các thuật toán mô hình được sử dụng:
Chúng tôi đã sử dụng 4 thuật toán khác nhau để giải quyết bài toán này, lần lượt là: Decision Tree, Extremely Random Trees, K-Nearest Neighbors và Neural Network. Đoạn mã dùng để huấn luyện các mô hình nằm trong thư mục: [Code](https://github.com/dnhuphatpy06/Project2/tree/main/Code)

### Sử dụng ứng dụng:
Clone repository về hoặc tải [thư mục](https://github.com/dnhuphatpy06/Project2/tree/main/App) về máy để sử dụng ứng dụng.
- Vì file **_app.py_** có sử dụng thư viện **_os_** nên không cần phải di chuyển đến thư mục trước khi chạy chương trình mà chỉ cần mở file lên chạy là được. Trước khi chạy đề nghị tải tất cả các thư viện cần thiết được _import_ trong chương trình.
- Để sử dụng ứng dụng bằng file **_app.exe_**, tải về tại [Link tải app.exe](http://www.mediafire.com/file/eoyyhxcqb7olvvr/app.zip), giải nén và đặt chung một thư mục với **_app.py_**. Đồng thời, vì mô hình _Extra Tree_ có kích thước lớn (hơn 100mb), không thể push trực tiếp lên GitHub nên hãy giải nén tệp [ExtraTree.zip](https://github.com/dnhuphatpy06/Project2/blob/main/App/model/ExtraTree.zip) và đặt vào thư mục **_model_**.
Cấu trúc của thư mục chứa **_app.py_** có dạng như sau:
![](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/App/emotion/C%E1%BA%A5u%20tr%C3%BAc%20th%C6%B0%20m%E1%BB%A5c.png)

### Kết quả dự đoán:
#### Mô hình Decision Tree:
![Đường cong học tập](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Decision%20Tree/Learning_Curve_Decision_Tree.png)
#### Mô hình Extremely Random Trees
![Đường cong học tập](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Extremely%20Randomized%20Trees/Learning_Curve_ExtraTree.png)
#### Mô hình K-Nearest Neighbors
![Đường cong học tập](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/K-Nearest%20Neighbors/Learning_Curve_KNN.png)
#### Mô hình Neural Network
![Đồ thị mất mát](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Neural%20Network/Loss_NN.jpg)
![Đồ thị chính xác](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Neural%20Network/Accuracy_NN.png)

#### Ma trận nhầm lẫn:
![](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Decision%20Tree/Confusion_Matrix_Decision_Tree.png)
![](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Extremely%20Randomized%20Trees/Confusiong_Matrix_ExtraTree.png)
![](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/K-Nearest%20Neighbors/Confusion_Matrix_KNN.png)
![](https://raw.githubusercontent.com/dnhuphatpy06/Project2/refs/heads/main/Code/Neural%20Network/Confusion_Matrix_NN.png)