import numpy as np
import os,sys
import caffe
import cv2
import pyttsx3
import time

classifier=[]
sub_classifier=[]
drawing=False
lock=False
ix,iy=-1,-1
ex,ey=-1,-1
frame=[]
roi=[]
bufferSize=1
bufferImg={}
r_scale=256
g_scale=256
b_scale=256
r_offset=256
g_offset=256
b_offset=256
game_scene='gamescene/scene.jpg'
game_pose_roi=np.zeros((7,4),dtype=np.int32)
game_pose_box=(125,153)
crow=0
scol=4
ccol=0
box_startAt=(175,15)
weight=[]
for i in range(7):
    game_pose_roi[i,:]=(crow*game_pose_box[0],ccol*game_pose_box[1],(crow+1)*game_pose_box[0],(ccol+1)*game_pose_box[1])
    ccol+=1
    if (i+1)%scol==0:
        crow+=1
        ccol=0  

def cost(y,y_pri):
    return np.linalg.norm(y_pri-y,2)/2.0

def dcostdy(y,y_pri):
    diff=y_pri-y
    return diff/np.linalg.norm(diff, 1)

def updateWeight(w,y,y_pri,lr=1,expendMat=False):
    dw=-lr*dcostdy(y, y_pri)
    if expendMat:
        if len(dw.shape)==1 or dw.shape[1]!=1:
            dw.reshape((1,np.size(dw)))
        newdw=np.repeat(dw,np.size(dw),axis=0)
        dw=newdw*np.identity((w.shape[0]),dtype=np.float32)
    return w-dw

def fx(w,x,b=0):
    return np.add(np.dot(x,w),b)

def objective(errs,thresholds):
    meet=True
    for i in range(len(errs)):
        if errs[i]>thresholds[i]:
            meet=False
            break
    return meet

def reinforcement(expected):
    index_map_path='indexMap'
    img_list={}
    for fname in os.listdir(index_map_path):
        class_id,fext=os.path.splitext(fname)
        img_list[class_id]=cv2.imread(os.path.join(index_map_path,fname))[0:256,0:256]
    dim=len(expected)
    weights=np.identity(dim,dtype=np.float32)
    threshold=1e-1
    tried=0
    maxTried=1000
    nsample=10
    all_x=[]
    cam=cv2.VideoCapture(1)
    cv2.namedWindow('calibration')
    cv2.setMouseCallback('calibration',draw_roi)
    totalSample=dim*nsample
    labels=[]
    idx=np.arange(totalSample)
    np.random.shuffle(idx)
    ready=0
    for k in range(len(expected)):
        n=0
        while True:
            ret,img=cam.read()
            frame=img.copy()
            frame=preprocess(frame,2)
            cv2.rectangle(img,(ix,iy),(ex,ey),(0,255,0),3)
            if ready>100:
                roi=frame[iy:ey,ix:ex]
                roi=cv2.resize(roi,(256,256))
                cdata=roi.reshape(roi.shape[0],roi.shape[1],1)
                cv2.imshow('roi',roi)
                x=classifier.predict([cdata])
                all_x.append(x)
                label=np.zeros(x.shape,dtype=np.float32)
                label[0,expected[k]]=1
                labels.append(label)
                if n<nsample-1:
                    cv2.putText(img,'Please do pose %d for calibration'%k,(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
                else:
                    cv2.putText(img,'Please do wrong pose/no pose for calibration',(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
                if cv2.waitKey(10)==27:
                    exit()
                n+=1
                if n>=nsample:
                    break
            else:
                cv2.putText(img,'Please Select the ROI',(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
                if cv2.waitKey(1)==27:
                    exit()
            if lock==False and compute_area((ix,iy), (ex,ey))>1 and ready==0:
                ready=1
            if ready>0 and ready<=100:
                cv2.putText(img,'Ready for pose %d'%k,(250,250),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,255,0))
                cv2.imshow('Class Image',img_list[str(k+1)])
                cv2.waitKey(1)
                ready+=1
            cv2.imshow('calibration',img)
            cv2.waitKey(1)
        if ready>0:
            ready=1
        #y=fx(weights,)
    cv2.destroyAllWindows()
    print('%d samples are captured for calibration'%totalSample)
    lr=0.01
    while tried<maxTried:
        sumErr=0
        for i in idx:
            y=fx(weights,all_x[i])
            err=cost(y,labels[i])
            weights=updateWeight(weights, y, labels[i], 0.01,True)
            sumErr+=err*err
        sumErr=np.sqrt(sumErr)
        print('total loss=%f at %d iteration'%(sumErr,tried+1))
        if sumErr<threshold:
            break
        tried+=1
        if tried%100==0:
            lr*=10
    return weights

def judge(pose,expected):
    return pose==expected
    
def preprocess(src,mode):
    if mode==2:
        gs=cv2.GaussianBlur(src,ksize=(11,11),sigmaX=10)
        gs=cv2.cvtColor(gs,cv2.COLOR_BGR2GRAY)
        #gs=cv2.fastNlMeansDenoising(gs,None,3,9,9)
        #gs=cv2.bilateralFilter(gs,9,75,75)
        #swimg=cv2.resize(gs,(256,256))
        pimg=cv2.Laplacian(gs,cv2.CV_32F,ksize=5)
        #pimg=cv2.Laplacian(gs,cv2.CV_8U,ksize=3)
        return pimg
    elif mode==1:
        return cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    else:
        return src
            

def load_model(model_path,prototxt_path):
    global classifier
    #net=caffe.Net(prototxt_path,model_path,caffe.TEST)
    classifier=caffe.Classifier(prototxt_path,model_path)
    
def load_submodel(model_path,prototxt_path):
    global sub_classifier
    #net=caffe.Net(prototxt_path,model_path,caffe.TEST)
    sub_classifier=caffe.Classifier(prototxt_path,model_path)

def compute_area(pt1,pt2):
    return np.linalg.norm(np.subtract(pt2,pt1))

def draw_roi(event,x,y,flags,param):
    global ix,iy,ex,ey,drawing,frame,roi,lock
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
        ex,ey=x,y
        lock=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            lock=True
            ex,ey=x,iy+x-ix
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        ex,ey=x,iy+x-ix
        lock=False

    
def realtime(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt
    game_scene_img=cv2.imread(game_scene)
    cam=cv2.VideoCapture(1)
    expected_pose=(0,2,1,3,6,4,5)
    # cv format: [H,W,C]
    #cam.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cv2.namedWindow('real time classifier')
    cv2.namedWindow('roi')
    cv2.namedWindow('Class Image')
    index_map_path='indexMap'
    img_list={}
    mode=2
    back_data_dir='back_data'
    if not os.path.isdir(back_data_dir):
        os.mkdir(back_data_dir)
    pose_dirs=[]
    for ipose in expected_pose:
        pose_dir=os.path.join(back_data_dir,'P%02d'%ipose)
        if not os.path.isdir(pose_dir):
            os.mkdir(pose_dir)
        pose_dirs.append(pose_dir)
    engine = pyttsx3.init()
    img_count=0
    modes=('RGB Mode','Grayscale Mode','Edge Mode')
    for fname in os.listdir(index_map_path):
        class_id,fext=os.path.splitext(fname)
        img_list[class_id]=cv2.imread(os.path.join(index_map_path,fname))[0:256,0:256]
    cv2.setMouseCallback('real time classifier',draw_roi)
    countImg=0
    for i in range(bufferSize):
        bufferImg[i]=[]
        
    countThreshold=(5,5,5,5,5,2,3)
    truecount=np.zeros((len(expected_pose)),dtype=np.int32)
    currentPose=0
    while True:
        ret,img=cam.read()
        frame=img.copy()
        rawimg=img.copy()
        cv2.putText(img,modes[mode],(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
        frame=preprocess(frame,mode)
        cv2.rectangle(img,(ix,iy),(ex,ey),(0,255,0),3)
        cv2.imshow('real time classifier',img)
        scene=game_scene_img.copy()
        pbbox=game_pose_roi[currentPose]
        sroi_color=(0,0,255)
        if compute_area((ix,iy), (ex,ey))>1:
            #print countImg
            #print bufferSize
            #print len(bufferImg)
            roi=frame[iy:ey,ix:ex]
            rawroi=rawimg[iy:ey,ix:ex]
            roi=cv2.resize(roi,(256,256))
            cv2.imshow('roi',roi)
            bufferImg[countImg]=roi
            countImg+=1
            if countImg>=bufferSize:
                countImg=0
                if lock==False:
                    if classifier !=[]:
                        class_scs=[]
                        for ii in range(bufferSize):
                            proi=bufferImg[ii]
                            cimg=proi
                            if len(cimg.shape)>2:
                                cimg=cimg[:,:,(2,1,0)]
                            else:
                                cimg=cimg.reshape(cimg.shape[0], cimg.shape[1],1 )
                            predicts=classifier.predict([cimg])
                            if w!=[]:
                                predicts=np.dot(predicts,w)
                            #predicts[:,predicts.shape[1]-1]*=0.00000005
                            #predicts[:,0]*=0.9
                            #predicts[:,1]*=0.8
                            if class_scs==[]:
                                class_scs=predicts
                            else:
                                class_scs=np.add(class_scs,predicts)
                        print(class_scs)
                        max_idx=np.argmax(class_scs)
                        if max_idx==0 and sub_classifier!=[]:
                            bin_pred=sub_classifier.predict([cimg])
                            max_idx=np.argmax(bin_pred)
                        #print max_idx
                        if max_idx!=len(expected_pose)-1:
                            ts = time.time()
                            tsstr=str(ts).replace('.','_')
                            cv2.imwrite(os.path.join(pose_dirs[currentPose],'%s.jpg'%tsstr),rawroi)
                        if max_idx==expected_pose[currentPose]:
                            truecount[currentPose]+=1
                            sroi_color=(0,255,0)
                            if truecount[currentPose]>=countThreshold[currentPose]:
                                currentPose+=1
                            if currentPose>=len(expected_pose):
                                currentPose=0
                                engine.say('Well Done')
                                engine.runAndWait()
                                truecount=np.zeros((len(expected_pose)),dtype=np.int32)
                                #cv2.putText(scene,'Well Done!',(250,250),cv2.FONT_HERSHEY_DUPLEX,3,(0,255,0))
                        class_idx=max_idx+1
                        class_str='%d'%class_idx
                        if class_str in img_list:
                            cv2.imshow('Class Image',img_list[class_str])
            else:
                key=cv2.waitKey(10)
                if key==27:
                    break
                elif key==103:
                    mode+=1
                    if mode>2:
                        mode=0
        cv2.rectangle(scene,(box_startAt[1]+pbbox[1],box_startAt[0]+pbbox[0]),(box_startAt[1]+pbbox[3],box_startAt[0]+pbbox[2]),sroi_color,3)
        cv2.imshow('Scene',scene)
        key=cv2.waitKey(10)
        if key==27:
            break
        elif key==103:
            mode+=1
            if mode>2:
                mode=0
    cv2.destroyAllWindows()


caffe.set_mode_gpu()
model_path=''
prototxt_path=''
if len(sys.argv)>2:
    model_path=sys.argv[1]
    prototxt_path=sys.argv[2]
    load_model(model_path, prototxt_path)

if len(sys.argv)>3:
    bufferSize=int(sys.argv[3])
    for i in range(bufferSize):
        bufferImg[i]=[]
    
if len(sys.argv)>5:
    model_path=sys.argv[4]
    prototxt_path=sys.argv[5]
    load_submodel(model_path, prototxt_path)

weight=np.identity(8,dtype=np.float32)
#weight[7,7]=0.03
#weight[7,7]=0.001
#weight[7,7]=0.000065
#weight[5,5]=2
#weight[4,4]=0.4
#weight[6,6]=1.2
#weight[3,3]=1.5
#weight[1,1]=1.2

weight[0,0]=2.3
weight[1,1]=2.6
weight[2,2]=33.3
weight[3,3]=63.2
weight[4,4]=7.95
weight[5,5]=35.65
weight[6,6]=2.67
weight[7,7]=0.064

#weight=reinforcement((0,1,2,3,4,5,6,7))
#print(weight)
#exit()

#game_scene_img=cv2.imread(game_scene)
#display_scene=[]
#for bbox in game_pose_roi:
#    display_scene=game_scene_img.copy()
#    cv2.rectangle(display_scene,(box_startAt[1]+bbox[1],box_startAt[0]+bbox[0]),(box_startAt[1]+bbox[3],box_startAt[0]+bbox[2]),(0,255,0),3)
#    cv2.imshow('scene',display_scene)
#    cv2.waitKey(2000)
realtime(weight)
