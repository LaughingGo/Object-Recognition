import matplotlib.pyplot as plt
from skimage import measure,data,color
import cv2
import os



output_label_txt = '/home/jianfenghuang/Myproject/CarPlateRecog/deep-anpr-master/label_data_train.txt'
f_output = open(output_label_txt,'w')
img_path = '/home/jianfenghuang/Myproject/CarPlateRecog/deep-anpr-master/Letters/'
imgs = os.listdir(img_path)
for img_name in imgs:

    a=cv2.imread(img_path+'/'+img_name)
    img=color.rgb2gray(a)
    # if img_name =='model1444.jpg':
    #     a=0
    # img=color.rgb2gray(data.horse())

    #detect all contours
    contours = measure.find_contours(img, 0.5)
    if len(contours)<1:
        continue
    edge = contours[0]

    # write txt
    print(img_name)
    f_output.writelines(img_name)
    f_output.writelines(' 1')
    curClass=img_name[9]
    f_output.writelines(' '+curClass+' ')

    pts_num=0
    for i in range(len(edge)):
        if i%5==0:
            pts_num+=1
    print('pts_num:', pts_num)
    f_output.writelines(str(pts_num))
    f_output.writelines(' ')
    t1=0
    t2=0
    for i in range(len(edge)):
        if i%5==0:
            f_output.writelines(str(int(edge[i,1])))
            f_output.writelines(' ')
            t1+=1
    # print('t1:',t1)
    for i in range(len(edge)):
        if i % 5 == 0:
            f_output.writelines(str(int(edge[i,0])))
            f_output.writelines(' ')
            t2 += 1
    # print('t2:', t2)
    f_output.writelines('\n')
f_output.close()

    # draw contour
    # fig, axes = plt.subplots(1,2,figsize=(8,8))
    # ax0, ax1= axes.ravel()
    # ax0.imshow(img,plt.cm.gray)
    # ax0.set_title('original image')
    #
    # rows,cols=img.shape
    # ax1.axis([0,rows,cols,0])
    # for n, contour in enumerate(contours):
    #     if n<1:
    #         print(contour)
    #         ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax1.axis('image')
    # ax1.set_title('contours')
    # plt.show()
