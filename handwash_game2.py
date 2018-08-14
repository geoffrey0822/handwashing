import numpy as np
import os,sys
import caffe
import cv2
import pyttsx
import time
import Queue
from PySide.QtGui import QApplication, QPixmap
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

def toQImage(im, copy=False):
    if im is None:
        return QImage()
    if im.dtype==np.uint8:
        if len(im.shape)==2:
            qim=QImage(im.data,im.shape[1],im.shape[0],im.strides[0],QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() 
        elif len(im.shape)==3:
            if im.shape[2]==3:
                qim=QImage(im.copy().tostring(), im.shape[1],im.shape[0],im.strides[0],QImage.Format_RGB888)
                return qim 
            elif im.shape[2]==4:
                qim=QImage(im.copy().tostring(), im.shape[1],im.shape[0],im.strides[0],QImage.Format_ARGB32)
                return qim 

class ImageWidget(QtGui.QWidget):
    def __init__(self,path,parent):
        super(ImageWidget,self).__init__(parent)
        #self.picture=QtGui.QPixmap(path)
        roi=QRect(0,0,256,256)
        self.picture=QtGui.QPixmap(path).copy(roi)
        self.picture=self.picture.scaled(96,96)
        print 'ok'
        
    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0,0, self.picture)



def getmask(img,step=100.0):
    gs=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,mask=cv2.threshold(gs,0,255,cv2.THRESH_BINARY)
    noise=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.int8)
    cv2.randn(noise,0,5)
    onoise=noise.astype('uint8')
    onoise=cv2.GaussianBlur(onoise,(9,9),3)
    res=cv2.bitwise_and(onoise,mask,mask=mask)
    #cv2.imshow('mask',onoise)
    return res,(mask-res).astype('float')/step

def updateHand(img,mask,delta):
    mask[:,:,0]+=delta
    mask[:,:,1]+=delta
    mask[:,:,2]+=delta
    newimg=np.multiply(img,mask/65535.0)
    return newimg,mask

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
game_scene='gamescene/sc.jpg'
game_hands='gamescene/hands.jpg'
game_pose_roi=np.zeros((7,4),dtype=np.int32)
game_pose_box=(125,153)
crow=0
scol=7
ccol=0
box_startAt=(175,15)
box_startAt=(25,15)
weight=[]
gui=None
game_thread=None
for i in range(7):
    game_pose_roi[i,:]=(crow*game_pose_box[0],ccol*game_pose_box[1],(crow+1)*game_pose_box[0],(ccol+1)*game_pose_box[1])
    ccol+=1
    if (i+1)%scol==0:
        crow+=1
        ccol=0
        
cv2.namedWindow('hand')
cv2.moveWindow('hand',256,256)

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
    nsample=1
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
                if cv2.waitKey(1000)==27:
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
    print '%d samples are captured for calibration'%totalSample
    lr=0.01
    while tried<maxTried:
        sumErr=0
        for i in idx:
            y=fx(weights,all_x[i])
            err=cost(y,labels[i])
            weights=updateWeight(weights, y, labels[i], 0.01,True)
            sumErr+=err*err
        sumErr=np.sqrt(sumErr)
        print 'total loss=%f at %d iteration'%(sumErr,tried+1)
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
        pimg=cv2.cvtColor(pimg,cv2.COLOR_GRAY2BGR)
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

class RunTask(QtCore.QThread):
    
    pose_updated= QtCore.pyqtSignal(object)
    
    def __init__(self,w,func):
        QtCore.QThread.__init__(self)
        self.w=w
        self.func={'func1':func}
        
    def run(self):
        self.func['func1'](self.w)
        
    def updateGUI(self):
        self.pose_updated.emit('')

class GameGUI(QtGui.QTableWidget):
    def __init__(self,parent=None):
        super(GameGUI,self).__init__(parent)
        self.img_list={}
        self.setWindowTitle('Recognition for Hand Washing Steps')
        self.w=[]
        self.task=[]
        self.title='Game'
        self.updatePose=-1
        self.trueCount=[]
        
    def setTask(self,w,func):
        global game_thread
        self.task=RunTask(w,func)
        self.task.pose_updated.connect(self.updatePoseFunc)
        game_thread=self.task
        
    def runTask(self):
        self.task.start()
        
    def updatePoseFunc(self):
        p=self.updatePose
        self.updatePose=-1
        if p>-1:
            self.cellWidget(p,2).setStyleSheet('QLabel {background-color:green;font-size:24px}')
            self.cellWidget(p,2).setText('%d'%self.trueCount[p])
            #self.cellWidget(p,2).setText('Done')
        
    def load_IndexMap(self,path):
        self.img_list={}
        i=0
        self.insertColumn(0)
        self.insertColumn(1)
        self.insertColumn(2)
        title=QtCore.QStringList()
        title.append('Pose')
        title.append('Example')
        title.append('How Many Times')
        self.setHorizontalHeaderLabels(title)
        for fname in os.listdir(path):
            class_id,fext=os.path.splitext(fname)
            fpath=os.path.join(path,fname)
            self.img_list[class_id]=cv2.imread(os.path.join(path,fname))[0:256,0:256]
            qim=toQImage(self.img_list[class_id])
            #QPixmap.fromImage(qim)
            self.insertRow(i)
            lbl=QtGui.QLabel('Pose %s'%class_id,self)
            lblStatus=QtGui.QLabel('None',self)
            self.setCellWidget(i,0,lbl)
            self.setCellWidget(i,2,lblStatus)
            self.setCellWidget(i,1,ImageWidget(fpath,self))
            self.setRowHeight(i,96)
            #print self.cellWidget(i,2).setStyleSheet('QLabel {background-color: red}')
            self.cellWidget(i,2).setStyleSheet('QLabel{background-color: red;font-size:24px}')
            self.cellWidget(i,2).setAlignment(Qt.AlignCenter)
            
            i+=1
        #self.setCellWidget(len(self.img_list.keys())-1,3)
        self.removeRow(i-1)
        self.setColumnWidth(2,96)
        self.resize(QtCore.QSize(800,700))
        self.trueCount=np.zeros(len(self.img_list.keys()))
        return self.img_list

    
def getMostPose(counts,threshold):
    poses=set(counts)
    mostPose=-1
    mostN=0
    for pose in poses:
        n=counts.count(pose)
        if n>=threshold:
            mostN=n
            mostPose=pose
            
    return mostPose
    
def realtime(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui,game_thread
    game_scene_img=cv2.imread(game_scene)
    cam=cv2.VideoCapture(1)
    expected_pose=(0,2,1,3,6,4,5)
    # cv format: [H,W,C]
    #cam.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cv2.namedWindow('real time classifier')
    #cv2.namedWindow('roi')
    cv2.namedWindow('Class Image')
    cv2.namedWindow('hand')
    cv2.moveWindow('hand',256,256)
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
    engine = pyttsx.init()
    img_count=0
    modes=('RGB Mode','Grayscale Mode','Edge Mode')
    for fname in os.listdir(index_map_path):
        class_id,fext=os.path.splitext(fname)
        img_list[class_id]=cv2.imread(os.path.join(index_map_path,fname))[0:256,0:256]
    cv2.setMouseCallback('real time classifier',draw_roi)
    countImg=0
    for i in range(bufferSize):
        bufferImg[i]=[]
        
    countThreshold=(5,5,5,5,5,5,5)
    truecount=np.zeros((len(expected_pose)),dtype=np.int32)
    currentPose=0
    
    # ready for game
    # ready end
    hist=[]
    max_hist=1000
    criteria=5
    trueCount=np.zeros(len(expected_pose),dtype=np.int32)
    threshold=criteria*0.7
    while True:
        ret,img=cam.read()
        if img is None:
            continue
        frame=img.copy()
        rawimg=img.copy()
        cv2.putText(img,modes[mode],(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
        frame=preprocess(frame,mode)
        cv2.rectangle(img,(ix,iy),(ex,ey),(0,255,0),3)
        cv2.imshow('real time classifier',img)
        #scene=game_scene_img.copy()
        pbbox=game_pose_roi[currentPose]
        sroi_color=(0,0,255)
        if compute_area((ix,iy), (ex,ey))>1:
            roi=frame[iy:ey,ix:ex]
            rawroi=rawimg[iy:ey,ix:ex]
            roi=cv2.resize(roi,(256,256))
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
                            if class_scs==[]:
                                class_scs=predicts
                            else:
                                class_scs=np.add(class_scs,predicts)
                        print class_scs
                        max_idx=np.argmax(class_scs)
                        #print len(expected_pose)
                        if max_idx==len(expected_pose):
                            del hist[:]
                            hist=[]
                        else:
                            hist.append(max_idx)
                            if len(hist)>=criteria:
                                seq=hist[len(hist)-criteria:]
                                # check 1
                                #if len(set(seq))==1:
                                    # UI Callback for update
                                    #print 'Passed for %d'%(max_idx+1)
                                    #gui.updatePose=max_idx
                                    #gui.trueCount[max_idx]+=1
                                    #game_thread.updateGUI()
                                    
                                # check 2
                                ppose=getMostPose(seq,threshold)
                                if ppose>-1:
                                    print 'Passed for %d'%(ppose+1)
                                    gui.updatePose=ppose
                                    gui.trueCount[ppose]+=1
                                    del hist[:]
                                    game_thread.updateGUI()
                                    
                            if len(hist)>=max_hist:
                                del hist[:len(hist)-criteria]
                        
                        #print max_idx
                        
                        #engine.say('Well Done! Your are clean now')
                        #engine.runAndWait()
                        
                        class_idx=max_idx+1
                        class_str='%d'%class_idx
                        if img_list.has_key(class_str):
                            cv2.imshow('Class Image',img_list[class_str])
        #cv2.rectangle(scene,(box_startAt[1]+pbbox[1],box_startAt[0]+pbbox[0]),(box_startAt[1]+pbbox[3],box_startAt[0]+pbbox[2]),sroi_color,3)
        key=cv2.waitKey(1)
        if key==27:
            break
        elif key==103:
            mode+=1
            if mode>2:
                mode=0
    cv2.destroyAllWindows()

def realtime2(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui
    app=QApplication(sys.argv)
    gui=GameGUI()
    gui.load_IndexMap('indexMap')
    gui.setTask(w, realtime)
    gui.show()
    realtime(w)
    #gui.runTask()
    #app.exec_()
    

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
#weight[7,7]=0.000065
#weight[5,5]=0.1
#weight[4,4]=0.4
#weight[6,6]=0.1
#weight[3,3]=1.5
#weight[1,1]=1.2
#weight=reinforcement((0,1,2,3,4,5,6,7))
#print weight
#exit()


#dirty_hand=cv2.bitwise_and(dirty,game_hand_img,mask=hand_mask)
#print hand_mask.shape
#display_scene=[]
#for bbox in game_pose_roi:
#    display_scene=game_scene_img.copy()
#    cv2.rectangle(display_scene,(box_startAt[1]+bbox[1],box_startAt[0]+bbox[0]),(box_startAt[1]+bbox[3],box_startAt[0]+bbox[2]),(0,255,0),3)
#    cv2.imshow('scene',display_scene)
#    cv2.waitKey(100)

#for h in range(35):
#    dirty_hand,dirty=updateHand(game_hand_img, dirty, delta)
#    cv2.imshow('hand',dirty_hand)
#    cv2.waitKey(100)
realtime2(weight)
