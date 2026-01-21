import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve

# Hazır bir derin öğrenme modelini kullanarak gerçek zamanlı bir yüz tespit (face detection) uygulaması


# ========================-Downloading Assets-========================

# https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector den indirilebilir

# Yukarıdaki dosya yolundan manuel indirebilirsin. Dosyaların olduğu yolu belirt (Aynı klasördeyse direkt isimlerini yaz)
# proto_path = "indirilen_klasor/deploy.prototxt"
# model_path = "indirilen_klasor/res10_300x300_ssd_iter_140000_fp16.caffemodel"
 
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)


URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assest ZIP does not exists.
# download kodunu bir kere calıstırdıktan sonra " Download if assest ZIP does not exists." kontrolunu yapan kodu ekledik.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================



# Bu kod bloğu, programın video kaynağını (kamera mı yoksa bir video dosyası mı olacağını) dinamik olarak seçmesini sağlar. Yani kodu her seferinde değiştirmek zorunda kalmadan, farklı kaynaklarla çalıştırabilmene olanak tanır.
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

if not source.isOpened():
    print(f"Hata: Kaynak açılamadı! (Kaynak: {s})")
    sys.exit()

#sys.argv, programı terminalden (komut satırından) çalıştırırken yanına yazdığın argümanları tutan bir listedir.

# Senaryo A: Sadece python dosya_adi.py yazarsan, listenin uzunluğu 1'dir. Bu blok çalışmaz ve s sıfır olarak kalır (kamera açılır).

# Senaryo B: python dosya_adi.py video.mp4 yazarsan, listenin 2. elemanı (sys.argv[1]) "video.mp4" olur. Kod bu durumda s değerini "video.mp4" yapar.

source = cv2.VideoCapture(s)

# Bu satır, OpenCV'nin video yakalama motorunu çalıştırır:

# Eğer s = 0 ise: Canlı kamera yayını başlar.

# Eğer s = "video.mp4" ise: Belirttiğin video dosyasını kare kare okumaya başlar.

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#.prototxt: Ağın mimarisini (katmanların nasıl dizildiğini) tanımlar.

# .caffemodel: Eğitilmiş ağırlıkları (öğrenilmiş yüz özelliklerini) içerir.

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# OpenCV'nin DNN (Derin Sinir Ağları) modülünü kullanarak Caffe formatındaki modeli belleğe yükler. Bu, sistemin "beynini" hazırladığı andır.

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7
# Bu sayılar rastgele seçilmemiştir. Kullandığımız bu model, ImageNet adı verilen ve milyonlarca fotoğraftan oluşan devasa bir veri setiyle eğitildi.

# 104: Tüm o milyonlarca resimdeki Mavi (Blue) kanalların ortalamasıdır.

# 117: Tüm o resimdeki Yeşil (Green) kanalların ortalamasıdır.

# 123: Tüm o resimdeki Kırmızı (Red) kanalların ortalamasıdır. (Not: OpenCV BGR formatını kullandığı için sıralama Mavi-Yeşil-Kırmızı şeklindedir.)

# Eğer biz bu "ortalama" değerleri çıkarırsak, resmin genel parlaklık seviyesini (aydınlık mı karanlık mı olduğunu) bir nevi yok saymış oluruz. Böylece yapay zeka, ışığın şiddetine değil, yüzün hatlarına ve şekline odaklanır.


while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    # Görüntüyü aynalar (genelde görüntülü görüşme hissi için tercih edilir).
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame. Ham görüntüyü yapay zekaya vermeden önce standart hale getirir:
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()
    # net.setInput(blob): Hazırlanan "blob" (sayısallaştırılmış veri paketi) ağa gönderilir.

    # net.forward(): Ağ, veriyi katmanlar arasından geçirir ve sonuçları detections değişkenine atar. Bu değişken, görüntüde bulunan nesnelerin koordinatlarını ve güven skorlarını içeren bir matristir.

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Kod, detections içindeki her bir tahmini döngüye alır:

        # Confidence (Güven Skoru): "Bu gördüğüm şey % kaç ihtimalle bir yüz?" sorusunun cevabıdır. Eğer bu oran conf_threshold (0.7 yani %70) değerinden büyükse, işlem başlar.

        if confidence > conf_threshold:
            x_top_left = int(detections[0, 0, i, 3] * frame_width)
            y_top_left = int(detections[0, 0, i, 4] * frame_height)
            x_bottom_right  = int(detections[0, 0, i, 5] * frame_width)
            y_bottom_right  = int(detections[0, 0, i, 6] * frame_height)

            # Koordinat Hesaplama: Model bize 0.0 ile 1.0 arasında normalize edilmiş değerler verir. Kod, bu değerleri orijinal pencere genişliği (frame_width) ve yüksekliği (frame_height) ile çarparak gerçek piksel koordinatlarına (x_top_left vb.) dönüştürür.

            cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - label_size[1]),
                (x_top_left + label_size[0], y_top_left + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )        

            # Çizim: cv2.rectangle ile yüzün etrafına yeşil bir kutu, cv2.putText ile de güven oranı yazılır.

            cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()

cv2.destroyWindow(win_name)
