import cv2
from threading import Thread, Lock
class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
def extract_frames(video_path):
    # save image names to .txt file
    # f_img_name_train = open(img_names+"_train", 'w')
    # f_img_name_val = open(img_names+"val", 'w')
    # f_img_folder_train = img_folder + "_train.txt"
    # f_img_folder_val = img_folder + "_val.txt"

    # f2 = open(frame_set_path,'r')
    video = cv2.VideoCapture()
    # vs = WebcamVideoStream(src=video_path).start()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)

    frame_index = 0
    frame_index_str = "{}".format(frame_index)
    count = 0
    # frame_tag = f2.readline().split(',')[0]
    frame_index = 0
    while True:
        _, frame = video.read()
        frame_index+=1
        print('frame:', frame_index)
        if frame is None:
            break
        if frame_index%25==0:
            img_folder = 'images_val'
            img_name = "{:>04d}".format(count)
            save_img_path = "{}/{}.jpg".format(img_folder, img_name)
            cv2.imwrite(save_img_path, frame)
            frame_index += 1
            count+=1
            print('frame:',count)
        elif frame_index%5==0:
            img_folder = 'images_train'
            img_name = "{:>04d}".format(count)
            save_img_path = "{}/{}.jpg".format(img_folder, img_name)
            cv2.imwrite(save_img_path, frame)
            frame_index += 1
            count+=1
            print('frame:',count)


        # if(frame_index_str==frame_tag):
        #     print("frame:", count, "index:", frame_tag)
        #     test_tag+=1
        #     if test_tag == 5:
        #         f_label_train = labe_path + "_train.txt"
        #         f_label_val = labe_path + "_val.txt"
        #         img_name = "{:>04d}".format(count)
        #         save_img_path = "{}/{}.jpg".format(f_img_folder_val, img_name)
        #         cv2.imwrite(save_img_path, frame)
        #         f_img_name_train.writelines(img_name+'\n')
        #         test_tag = 0
        #     else:
        #         img_name = "{:>04d}".format(count)
        #         save_img_path = "{}/{}.jpg".format(f_img_folder_train, img_name)
        #         cv2.imwrite(save_img_path, frame)
        #
        #         f_img_name_val.writelines(img_name + '\n')
        #
        #     count += 1
        #     # frame_tag = f2.readline().split(',')[0]
        # frame_index += 1
        # frame_index_str = "{}".format(frame_index)
    # f_img_name_train.close()
    # f_img_name_val.close()
    # f_label_train.close()
    # f_label_val.close()
    # f2.close()
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(count))

if __name__ == '__main__':
    video_path = "VID_20180727_131347.mp4"
    img_path = 'images'
    img_name = 'img_name'
    # frame_set_path = 'VID_20180720_101418_XY12gt.txt'
    extract_frames(video_path)