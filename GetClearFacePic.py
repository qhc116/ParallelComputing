import cv2
import os

# 要提取的视频路径
video_path = r"C:\Users\86155\Desktop\bxjsdzy\cmtk.flv"
# 清晰图片输出路径
outPutDirName = r"C:\Users\86155\Desktop\bxjsdzy\cmtk\clearpic"
# 人脸图片的输出路径
facePutDir = r"C:\Users\86155\Desktop\bxjsdzy\cmtk\clearface"
# 创建路径
if not os.path.exists(outPutDirName):
    # 如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)

if not os.path.exists(facePutDir):
    # 如果文件目录不存在则创建目录
    os.makedirs(facePutDir)

times = 0  # 当前读取帧数
now = 0    # 当前为第x次查找清晰图片
# 提取视频的频率，每x帧提取一个
frameFrequency = 25
# 取得视频源
camera = cv2.VideoCapture(video_path)
# 人脸侦测器
face_classfier = cv2.CascadeClassifier(r"D:\development\python3.6\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
eyes_classfier = cv2.CascadeClassifier(r"D:\development\python3.6\Lib\site-packages\cv2\data\haarcascade_eye.xml")

flag = True
while flag == True:
    pic_list = []
    blur_list = []
    o = 0
    now += 1
    # 读取帧
    while o < frameFrequency :
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        else:
            pic_list.append(image)
            o += 1

    if pic_list:
        for img in pic_list:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.Laplacian(img, cv2.CV_64F).var()
            blur_list.append(blur)
        index = blur_list.index(max(blur_list))
        clear_image = pic_list[index]
        hasface = face_classfier.detectMultiScale(cv2.cvtColor(clear_image, cv2.COLOR_RGB2GRAY), scaleFactor=1.3, minNeighbors=10, minSize=(32, 32))
        if len(hasface) > 0:
            facenum = 1
            print("当前提取至" + str(int(now/60)) +"分" + str(now%60) + "秒处。  当前图片人脸数量：" + str(len(hasface)))
            # 测试画框，不需要的话可以去掉
            for (x,y,w,h) in hasface:
                clear_face = clear_image[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(facePutDir + r'\face_' + str(now) + str(facenum) + '.jpg', clear_face)
                facenum += 1
        cv2.imwrite(outPutDirName + r'\img_' + str(now) + '.jpg', clear_image)
    else:
        flag = False

print('图片提取结束')
camera.release()