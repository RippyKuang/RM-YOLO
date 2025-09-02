import os 
import xml.etree.ElementTree as ET
import random
import math
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import time
import  shutil
from threading import Thread, Lock

sns.set() 
IMG_TYPE = ['png','jpg','bmp']
NORMAL_TYPE = ['png','jpg','bmp','txt','xml']
rune_dic = {
		'notact_red':0,
		'act_red':1,
		'notact_blue':2,
		'act_blue':3,
             }
dic = {'armor_sentry_blue':0,
       'armor_sentry_red':1,
       'armor_sentry_none':2,
       'armor_hero_blue':3,
       'armor_hero_red':4,
       'armor_hero_none':5,
       'armor_engine_blue':6,
       'armor_engine_red':7,
       'armor_engine_none':8,
       'armor_infantry_3_blue':9,  #28
       'armor_infantry_3_red':10,  #29
       'armor_infantry_3_none':11,
       'armor_infantry_4_blue':12,  #25
       'armor_infantry_4_red':13,    #26
       'armor_infantry_4_none':14,
       'armor_infantry_5_blue':15,  #31
       'armor_infantry_5_red':16,   #32
       'armor_infantry_5_none':17,
       'armor_outpost_blue':18,
       'armor_outpost_red':19,
       'armor_outpost_none':20,
       'armor_base_blue':21,
       'armor_base_red':22,
       'armor_base_none':23,
       'armor_bal_3_blue':24,
       'armor_bal_3_red':25,
       'armor_bal_3_none':26,
       'armor_bal_4_blue':27,
       'armor_bal_4_red':28,
       'armor_bal_4_none':29,
       'armor_bal_5_blue':30,
       'armor_bal_5_red':31,
       'armor_bal_5_none':32,
       }
old_dic = {'armor_sentry_blue':0,
       'armor_sentry_red':1,
       'armor_sentry_none':2,
       'armor_hero_blue':3,
       'armor_hero_red':4,
       'armor_hero_none':5,
       'armor_engine_blue':6,
       'armor_engine_red':7,
       'armor_engine_none':8,
       'armor_infantry_3_blue':9,  #28
       'armor_infantry_3_red':10,  #29
       'armor_infantry_3_none':11,
       'armor_infantry_4_blue':12,  #25
       'armor_infantry_4_red':13,    #26
       'armor_infantry_4_none':14,
       'armor_infantry_5_blue':15,  #31
       'armor_infantry_5_red':16,   #32
       'armor_infantry_5_none':17,
       'armor_outpost_blue':18,
       'armor_outpost_red':19,
       'armor_outpost_none':20,
       'armor_base_blue':21,
       'armor_base_red':22,
       'armor_base_purple':23,
       'armor_base_none':24,
       }
sjdic = {'armor_sentry_blue':0,
       'armor_sentry_red':9,
       'armor_sentry_none':18,
       'armor_hero_blue':1,
       'armor_hero_red':10,
       'armor_hero_none':19,
       'armor_engine_blue':2,
       'armor_engine_red':11,
       'armor_engine_none':20,
       'armor_infantry_3_blue':3,  #28
       'armor_infantry_3_red':12,  #29
       'armor_infantry_3_none':21,
       'armor_infantry_4_blue':4,  #25
       'armor_infantry_4_red':13,    #26
       'armor_infantry_4_none':22,
       'armor_infantry_5_blue':5,  #31
       'armor_infantry_5_red':14,   #32
       'armor_infantry_5_none':23,
       'armor_outpost_blue':6,
       'armor_outpost_red':15,
       'armor_outpost_none':24,
       'armor_base_blue':8,
       'armor_base_red':17,
       'armor_base_purple':26,
       'armor_base_none':35,
       }
r_dic = {v:k for k,v in dic.items()}
r_sjdic = {v:k for k,v in sjdic.items()}
r_rune_dic=  {v:k for k,v in rune_dic.items()}
big_dic = {
    "23":"24",
    "24":"25",
    "25":"27",
    "26":"28",
    "27":"30",
    "28":"31",
}
rm_bal = {
    "13":"9", #b3
    "10":"12",#b4
    "16":"15",#b5
    "14":"10",
    "11":"13",
    "17":"16",
}

re_bal = {
    "23":"9", #b3
    "24":"10",#b4
    "25":"12",#b5
    "26":"13",
    "27":"15",
    "28":"16",
}
old_dic = {
    "0" :"0",
    "9" :"1",
    "18" :"2",
    "1" :"3",
    "10" :"4",
    "19" :"5",
    "2" :"6",
    "11" :"7",
    "20" :"8",
    "3" :"9",
    "12" :"10",
    "21" :"11",
    "4" :"12",
    "13" :"13",
    "22" :"14",
    "5" :"15",
    "14" :"16",
    "23" :"17",
    "6":"18",
    "15":"19",
    "24":"20",
    "8":"21",
    "17":"22",
    "35":"23",
}
def loadfolder(path,target_suffix):
    target = []
    for dirpath, dirnames,filenames in os.walk(path):
        for fn in filenames:
            suf =fn.split('.')[-1]
            if suf in target_suffix:
                target.append((dirpath,fn))
            elif suf not in NORMAL_TYPE:
                print(dirpath+'/'+fn)
    return target

def remove012(files):
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                if data.split(' ')[0] not in ['2']:
                    d.append(data)
        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()

def convertBigNum(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                try:
                    line[0] =re_bal[line[0]]
                
                except  Exception:
                    d.append(data)
                  #  print("error: "+line[0]+" : "+file)
                    continue
                data = " ".join(line)
                d.append(data)
        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()

def convertInv(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                p1_x = line[1]
                p1_y = line[2]
                p2_x = line[3]
                p2_y = line[4]

                p3_x = line[5]
                p3_y = line[6]
                p4_x = line[7]
                p4_y = line[8][:-1]

                line[1] =p2_x
                line[2] =p2_y
                line[3] =p1_x
                line[4] =p1_y
                line[5] =p4_x
                line[6] =p4_y
                line[7] =p3_x
                line[8] =p3_y+'\n'    
                print(line)
                data = " ".join(line)    
                d.append(data) 

        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()

def CLAMP(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
          

                line[1] =clamp(line[1])
                line[2] =clamp(line[2])
                line[3] =clamp(line[3])
                line[4] =clamp(line[4])
                line[5] =clamp(line[5])
                line[6] =clamp(line[6])
                line[7] =clamp(line[7])
                line[8] =clamp(line[8][:-1])+'\n'    
            
                data = " ".join(line)    
                d.append(data) 

        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()
def Five2Four(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            if len(datas) == 0:
                print(file)
                continue
            for data in datas:
                line = data.split(' ')
                if len(line)<7:
                    print("error "+file)
                    continue
                data = " ".join(line[:5])+" "+" ".join(line[7:])
                d.append(data) 

        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()
def face2Four(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            if len(datas) == 0:
                print(file)
                continue
            for data in datas:
                line = data.split(' ')
                if len(line)<7:
                    print("error "+file)
              
            
                data = line[0]+' '+" ".join(line[5:])    
                d.append(data) 

        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()
def ExpandFour(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            if len(datas) == 0:
                print(file)
                continue
            for data in datas:
                line = data.split(' ')
                if(len(line)<8):
                    print(file)
                iline  = [float(x) for x in line[1:]]
                iline =  expand_label(iline,ratio=1)
                iline  = [clamp(str(x)) for x in iline]
                data = line[0]+' '+" ".join(iline)+'\n'
                d.append(data) 

        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()
def clamp(str_):
    if float(str_)<0:
        return str(0)
    elif float(str_)>1:
        return str(1)
    else:
        return str_
    
def expand_label(label,ratio=0.5):
    exp_label =[]
    arg = np.array([1.0,1.0,-1.0,-1.0]) * ratio
    deta_xy1 = np.asarray([label[0]-label[2],label[1]-label[3]] *2)
    exp_label[:4] = label[:4] + arg * deta_xy1
    deta_xy2 = np.asarray([label[4]-label[6],label[5]-label[7]] *2)
    exp_label[4:] = label[4:] + arg * deta_xy2
  
    return np.asarray(exp_label)
  
def rotate_and_pad(src,src_label):
    max_x = int(np.max(src_label[[0,2,4,6]]))
    min_x = int(np.min(src_label[[0,2,4,6]]))
    max_y = int(np.max(src_label[[1,3,5,7]]))
    min_y = int(np.min(src_label[[1,3,5,7]]))

   
    h, w =max_y-min_y,max_x-min_x
    hm = math.ceil(h*2.0)
    wm = math.ceil(w*1.4)


    padding_h = (hm - h) // 2
    padding_w = (wm - w) // 2
    center = (hm // 2, wm // 2)
    pminy = min_y-padding_h if min_y-padding_h>0 else 0
    pmaxy = max_y+padding_h if max_y+padding_h<src.shape[:2][0] else src.shape[:2][0]
    pminx = min_x-padding_w if min_x-padding_w>0 else 0
    pmaxx = max_x+padding_w if max_x+padding_w<src.shape[:2][1] else src.shape[:2][1]
   

    img_padded = src[pminy:pmaxy,pminx:pmaxx,:]
    src_label = src_label - [min_x - padding_w,min_y - padding_h]*4 

    M = np.eye(3)
    rad = math.atan((src_label[7]-src_label[1])/(src_label[6]-src_label[0]))
    M[:2] = cv.getRotationMatrix2D(center, rad*57.3, 1)
    rotated_padded = cv.warpAffine(img_padded, M[:2], (wm, hm))

    xy = np.ones((4, 3))
    xy[:, :2] = src_label.reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    src_label = xy[:,:2].reshape(8)


    S = np.eye(3)
    S[0, 1] = (src_label[2]-src_label[0])/(src_label[1]-src_label[3])
    rotated_padded = cv.warpAffine(rotated_padded, (S )[:2], (wm, hm))

    xy = np.ones((4, 3))
    xy[:, :2] = src_label.reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ S.T  # transform
    src_label = xy[:,:2].reshape(8)
  

    return rotated_padded,src_label

    

def copyPaste(src,src_label,dst,dst_labels):
   src = cv.imread(src)
   dst = cv.imread(dst)
   msk = np.zeros(dst.shape[:2],np.uint8)
   src_shape = np.asarray(src.shape[:2] * 4)[[1,0,3,2,5,4,7,6]]
   dst_shape = np.asarray(dst.shape[:2] * 4)[[1,0,3,2,5,4,7,6]]
   src_label = [float(x) for x in src_label.split(' ')[1:]]* src_shape
   

   src,src_label = rotate_and_pad(src,src_label)
   src_shape =  np.asarray(src.shape[:2])[[1,0]]


   exp_label = expand_label(src_label,src_shape,0.5)
   exp_label = expand_label(exp_label[[3,2,5,4,7,6,1,0]],src_shape[[1,0]],0.2)[[7,6,1,0,3,2,5,4]]

   max_x = int(np.max(exp_label[[0,2,4,6]]))
   min_x = int(np.min(exp_label[[0,2,4,6]]))
   max_y = int(np.max(exp_label[[1,3,5,7]]))
   min_y = int(np.min(exp_label[[1,3,5,7]]))

   src = src[min_y:max_y,min_x:max_x]

   src_label = src_label - [min_x,min_y]*4


   pst1=np.float32([[src_label[0],src_label[1]],[src_label[2],src_label[3]],[src_label[4],src_label[5]],[src_label[6],src_label[7]]])
   rmsks = []
   tmsks = []
   for dst_label in dst_labels:
        dst_label = [float(x) for x in dst_label.split(' ')[1:]]* dst_shape

        pst2=np.float32([[dst_label[0],dst_label[1]],[dst_label[2],dst_label[3]],[dst_label[4],dst_label[5]],[dst_label[6],dst_label[7]]])
        matrix=cv.getPerspectiveTransform(pst1,pst2)
        gsrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)+1
        msk=cv.warpPerspective(src,matrix,(dst.shape[1],dst.shape[0]))
        qmsk=cv.warpPerspective(gsrc,matrix,(dst.shape[1],dst.shape[0]))
        _,temp_msk= cv.threshold(qmsk, 0, 255,cv.THRESH_BINARY)
    
        rmsks.append(cv.bitwise_not(temp_msk))
        tmsks.append(msk)
   for i,tmsk in enumerate(tmsks):
        dst = cv.merge([cv.bitwise_and(s,rmsks[i]) for s in cv.split(dst)])
        dst = cv.add(dst,tmsk)
   return dst
    
    

  
   
def xml2txt(files):
    for (p,n) in files:
        xmlfilename = p+'/'+n
        tree = ET.ElementTree(file=xmlfilename)
        node = tree.getroot()
        objects =node.findall("object")
        txt_contents =[]

        for obj in objects:
            name = str(rune_dic[obj.find("name").text])

            node = obj.find("bndbox")
            top_left_x = clamp(node.find("top_left_x").text)
            top_left_y = clamp(node.find("top_left_y").text)
            bottom_left_x =  clamp(node.find("bottom_left_x").text)
            bottom_left_y =  clamp(node.find("bottom_left_y").text)
            bottom_right_x =  clamp(node.find("bottom_right_x").text)
            bottom_right_y =  clamp(node.find("bottom_right_y").text)
            top_right_x =  clamp(node.find("top_right_x").text)
            top_right_y =  clamp(node.find("top_right_y").text)
            txt_contents.append(" ".join([name,top_left_x,top_left_y,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y,top_right_x,top_right_y])+'\n')
        file = open(p+'/'+n.split('.')[-2]+'.txt','w')
        file.writelines(txt_contents)  
        file.close()

def removexml(files):
    for (p,n) in files:
        os.remove(p+'/'+n)   


def removenone(files):
    for (p,n) in files:
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            if len(datas) == 0:
                f.close()
                os.remove(p+'/'+n)
                try:
                    os.remove(p+'/'+txt2png(n)) 
                except  Exception:
                    os.remove(p+'/'+txt2jpg(n)) 
                continue  
            line = datas[0].split(' ')
            if(len(line)>9):
                f.close()
                os.remove(p+'/'+n)
                try:
                    os.remove(p+'/'+txt2png(n)) 
                except  Exception:
                    os.remove(p+'/'+txt2jpg(n)) 
           
        f.close()    
def removeNoLabel(files):
    for (p,n) in files:
        txt_file = p+'/'+png2txt(n)
        
        if not os.path.exists(txt_file):
             shutil.move(p+'/'+n, "./bad/"+n)
             print("remove :"+n)


def makeSentry(sentry_path,oths_path,num,dst):
    sts = loadfolder(sentry_path,['txt'])
    _oths = loadfolder(oths_path,['txt'])
    oths = []

    for x in range(len(_oths)):
        oth,othn = _oths[x]
        cnt = 0 
        with open(oth+'/'+othn, 'r', encoding='utf-8') as f:
            temp_datas = f.readlines()
            for l in temp_datas:
                if l.split(' ')[0] not in ['23','24','25','26','27','28','21','22','3','4','5']:
                    cnt=cnt+1
        if cnt != 0:
            oths.append(_oths[x])
                    
    if  num>len(oths):
        print("太多了"+str(len(oths)))
        num = len(oths)
    print(str(len(sts))+"哨兵")
    print(str(len(oths))+"其他")
    random.shuffle(sts)
    random.shuffle(oths)
    for x in tqdm(list(range(num))):
        st,stn = sts[x% len(sts)]
        oth,othn = oths[x]
        with open(st+'/'+stn, 'r', encoding='utf-8') as f:
            stl = f.readlines()[0]
        f.close()
        datas =[ ]
        ori_dta = []
        with open(oth+'/'+othn, 'r', encoding='utf-8') as f:
            temp_datas = f.readlines()
            for l in temp_datas:
                if l.split(' ')[0] in ['23','24','25','26','27','28','21','22','3','4','5']:
                    ori_dta.append(l)
                else:
                    datas.append(l)
        f.close()
    
        try:
            res =  copyPaste(st+'/'+txt2jpg(stn), stl ,oth+'/'+txt2png(othn),datas)
        except Exception:
            res =  copyPaste(st+'/'+txt2jpg(stn), stl ,oth+'/'+txt2jpg(othn),datas)
        if x<5000:
            cv.imwrite(dst+'/train/'+str(x)+".png",res)
        else:
            cv.imwrite(dst+'/val/'+str(x)+".png",res)
        for d in datas:
            ld = d.split(' ')
            ld[0] = stl[0]
            ori_dta.append(' '.join(ld))
        if x<5000:
            file = open(dst+'/train/'+str(x)+".txt",'w')
            file.writelines(ori_dta)  
            file.close()
        else:
            file = open(dst+'/val/'+str(x)+".txt",'w')
            file.writelines(ori_dta)  
            file.close()

        

def motion_blur(image, degree=10, angle=20):
    image = np.array(image)
    
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))
    
    motion_blur_kernel = motion_blur_kernel / degree        
    blurred = cv.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def five_viewer(path,gpath=None,bpath=None,exp=2):
    txts = loadfolder(path,['txt'])
    gcnt =0
    bcnt =0
    for tp,tn in txts:
        try:
            imgp = tp+'/'+tn.split('.')[-2]+'.png' 
            img0 = cv.imread(imgp)
            img = cv.resize(img0,(img0.shape[1]*exp,img0.shape[0]*exp))
        except Exception:
            imgp = tp+'/'+tn.split('.')[-2]+'.jpg' 
            img0 = cv.imread(imgp)
            img = cv.resize(img0,(img0.shape[1]*exp,img0.shape[0]*exp))
        

        with open(tp+'/'+tn, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                name = r_rune_dic[int(line[0])]
                loc = [float(x) for x in line[1:]]* np.asarray(img.shape[:2] * 5)[[1,0,3,2,5,4,7,6,9,8]] 
                loc = [int(x) for x in loc]
                img = cv.circle(img, (loc[0],loc[1]), 2, (0,255,255), 2) 
                img = cv.circle(img, (loc[2],loc[3]), 2, (255,255,0), 2) 
                # img = cv.circle(img, (loc[4],loc[5]), 2, (0,255,0), 2) 
                # img = cv.circle(img, (loc[6],loc[7] ), 2, (0,255,255), 2) 
                # img = cv.circle(img, (loc[8],loc[9] ), 2, (255,0,0), 2) 
                img = cv.circle(img, (loc[6],loc[7] ), 2, (0,255,0), 2) 
                img = cv.circle(img, (loc[8],loc[9] ), 2, (0,0,255), 2)  #黄蓝绿红

                cv.putText(img,name,(loc[2],loc[3]+5),cv.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),1)
          #  img = motion_blur(img,30,225)  #45 135 225 315
            cv.imshow("show", img)
            key = cv.waitKey(0) 
            #print(key)
            #if key == 108:
              #  saveLabel(bpath,str(bcnt),img0,datas)
            #    bcnt += 1
           # elif key ==120:
            #    os.remove(tp+'/'+tn)   
            #    os.remove(imgp)   

          #  else:  
            #    saveLabel(gpath,str(gcnt),img0,datas)
                #gcnt += 1
                
        f.close()

     
def res_viewer(path,gpath=None,bpath=None,exp=2):
    txts = loadfolder(path,['txt'])
    gcnt =0
    bcnt =0
    for tp,tn in txts:
        try:
            imgp = tp+'/'+tn.split('.')[-2]+'.png' 
            img0 = cv.imread(imgp)
            img = cv.resize(img0,(img0.shape[1]*exp,img0.shape[0]*exp))
        except Exception:
            imgp = tp+'/'+tn.split('.')[-2]+'.jpg' 
            img0 = cv.imread(imgp)
            img = cv.resize(img0,(img0.shape[1]*exp,img0.shape[0]*exp))
        

        with open(tp+'/'+tn, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                name = r_rune_dic[int(line[0])]
                loc = [float(x) for x in line[1:]]* np.asarray(img.shape[:2] * 4)[[1,0,3,2,5,4,7,6]] 
                loc = [int(x) for x in loc]
                img = cv.circle(img, (loc[0],loc[1]), 2, (0,255,255), 2) 
                img = cv.circle(img, (loc[2],loc[3]), 2, (255,255,0), 2) 
                img = cv.circle(img, (loc[4],loc[5]), 2, (0,255,0), 2) 
                img = cv.circle(img, (loc[6],loc[7] ), 2, (0,0,255), 2) 
                cv.putText(img,name,(loc[2],loc[3]+5),cv.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),1)
          #  img = motion_blur(img,30,225)  #45 135 225 315
            cv.imshow("show", img)
            key = cv.waitKey(0) 
            #print(key)
            #if key == 108:
              #  saveLabel(bpath,str(bcnt),img0,datas)
            #    bcnt += 1
           # elif key ==120:
            #    os.remove(tp+'/'+tn)   
            #    os.remove(imgp)   

          #  else:  
            #    saveLabel(gpath,str(gcnt),img0,datas)
                #gcnt += 1
                
        f.close()

def png2txt(s):
    return s.split('.')[-2]+'.txt' 
def txt2png(s):
    return s.split('.')[-2]+'.png' 
def txt2jpg(s):
    return s.split('.')[-2]+'.jpg' 
lock = Lock()
def showdisplot(dis,dis2,fid=0):
    lock.acquire()
    f = plt.figure()                            
    f.add_subplot(1,2,1)
    sns.distplot(dis,kde=False)               
    plt.title("(area)", fontsize=20)            
    f.add_subplot(1,2,2)
    sns.distplot(dis2,kde=False)                             
    plt.title("(dist)", fontsize=20)    
    plt.subplots_adjust(wspace=0.3)        
    plt.savefig('./'+str(fid)+'.jpg')
    plt.cla()
    lock.release()

def getArea(label):
    max_x = int(np.max(label[[0,2,4,6]]))
    min_x = int(np.min(label[[0,2,4,6]]))
    max_y = int(np.max(label[[1,3,5,7]]))
    min_y = int(np.min(label[[1,3,5,7]]))
    return math.log10((max_x-min_x)*(max_y-min_y))
def getDist(label):
    left_x,left_y = (label[0]+label[2])//2 , (label[1]+label[3])//2
    right_x,right_y = (label[4]+label[6])//2 , (label[5]+label[7])//2

    return math.sqrt((left_x-right_x)*(left_x-right_x) + (left_y-right_y)*(left_y-right_y))
def _analyze(path,target,fid=0):
    txts = loadfolder(path,['txt'])
   

    areas = []
    dist  = []
    for tp,tn in tqdm(txts,position =fid):
          try:
            imgp = tp+'/'+tn.split('.')[-2]+'.png' 
            img = cv.imread(imgp)
            shape = img.shape[:2]
          except Exception:
            imgp = tp+'/'+tn.split('.')[-2]+'.jpg' 
            img = cv.imread(imgp)
            shape = img.shape[:2]
          with open(tp+'/'+tn, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                if  int(line[0]) in target:
                    loc = [float(x) for x in line[1:]]* np.asarray(shape * 4)[[1,0,3,2,5,4,7,6]] 
                    loc = [int(x) for x in loc]
                    areas.append(getArea(np.asarray(loc)))
                    dist.append(getDist(np.asarray(loc)))
    showdisplot(areas,dist,fid)
    
    
def analyze(pathes,targets):
    for i,p in enumerate(pathes):
        t1 = Thread(target=_analyze,args=(p,targets[i],i,))
        t1.start()
        

if __name__ == '__main__':
    sourcepath =  "./rune"
    datasetpath = "../rune"
    # jpg = loadfolder(datasetpath,["jpg"])
    # removeNoLabel(jpg)
#     src = "./xmlDataset/23sentry/HERO-23-OTH-0.jpg" 
#     src_label="1 0.423553 0.526173 0.420942 0.551289 0.46648 0.555357 0.468751 0.530057"
#     dst = "./xmlDataset/SJTU/SJTU-21/UC_N/Armor/SJTU-21-UC_N-3213.png"
#     dst_label = ["14 0.664582 0.517403 0.665947 0.562215 0.73865 0.560478 0.736923 0.513947",
# "7 0.593877 0.465479 0.593162 0.491937 0.63035 0.492271 0.630407 0.465778",
# "22 0.0342471 0.346396 0.0330575 0.365014 0.0753888 0.369237 0.0755143 0.349704",
# "5 0.760138 0.488119 0.759425 0.513455 0.834388 0.50855 0.835429 0.482262"
# ]
  #  makeSentry("./detaset/train/images","./xmlDataset",3000,"./temp")
  #  res_viewer(datasetpath,exp=1)
    #res_viewer(datasetpath,exp=1)
 #   res_viewer("D:\Project\datasets\detaset\train\pingbu",exp=2)
    #analyze(["./detaset/train","./detaset/train"],[[6,7,8],[9,10,11]])
 #   _analyze("./detaset/train",[0,1,2])
 #   bigpath = "./sj_gkd"
   # png = loadfolder(sourcepath,['png','jpg'])
    # xml = loadfolder(sourcepath,['xml'])
    # xml2txt(xml)
    txt = loadfolder(datasetpath,['txt'])
  #  ExpandFour(txt)
    #addisbig(txt)
  #  convertInv(txt)
  #  CLAMP(txt)
  
    
   # remove012(txt)
   # removenone(txt)
  #  bigtxt = loadfolder(bigpath,['txt'])
    remove012(txt)


   #  png = loadfolder(sourcepath,['png','jpg'])#
    # random.seed(2024116)
    # random.shuffle(png)
    # thre = len(png)*0.8
    # for i,(p,n) in enumerate(png):
    #     if i<thre:
    #         try:
    #             shutil.copy(p+'/'+n,datasetpath+"/train/"+n)
    #             shutil.copy(p+'/'+png2txt(n),datasetpath+"/train/"+png2txt(n))
    #         except Exception:
    #             print("未找到"+p+'/'+n)
    #             continue
    #     else:
    #         try:
    #             shutil.copy(p+'/'+n,datasetpath+"/val/"+n)
    #             shutil.copy(p+'/'+png2txt(n),datasetpath+"/val/"+png2txt(n))
    #         except Exception:
    #             print("未找到"+p+'/'+n)
    #             continue
    
