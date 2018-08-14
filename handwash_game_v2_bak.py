import numpy as np
import os,sys
import caffe
import cv2
import pyttsx
import time
import Queue
from PySide.QtGui import QApplication, QPixmap, QMovie, QVBoxLayout
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import imageio
from PySide.QtCore import QByteArray, QCoreApplication, QSize

QtCore.QCoreApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
QtGui.QImageReader.supportedImageFormats()

gray_color_table = [qRgb(i, i, i) for i in range(256)]

screen_width=1366
screen_height=768
camID=1
cool_down_time=3
cool_down_start=0
threadStarted=True
replayOnly=False
isManualNext=True
current_pose_startend=[0,0]
current_pose_duration=0
current_pose_duration_limit=1

transitions=['','Step 1\nPalms','Step 2\nBetween Fingers','Step 3\nBack of Hands','Step 4\nBack of Fingers','Step 5\nFinger Tips','Step 6\nThumbs','Step 7\nWrists','']
infos=['','Step 1: Rub your palms together.','Step 2: Rub both your hands while interlocking your fingers.','Step 3: Rub the back of each hand.','Step 4: Rub the back of your fingers.','Step 5: Rub the tips of your fingers.','Step 6: Rub your thumbs and the ends of your wrists.','Step 7: Rub both wrists in a rotating manner.','']

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

class OverlayLabel(QtGui.QLabel):
    def __init__(self,parent=None):
        super(OverlayLabel,self).__init__(parent)
        self.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.setText('hello')
        self.setGeometry(200,200,200,200)

class ImageWidget(QtGui.QWidget):
    def __init__(self,path,parent):
        super(ImageWidget,self).__init__(parent)
        #self.picture=QtGui.QPixmap(path)
        roi=QRect(0,0,256,256)
        self.picture=QtGui.QPixmap(path)
        self.picture=self.picture.scaled(256,256)
        self.label=''
        self.lightOn=False
        self.sWidth=256
        self.sHeight=256
        print 'ok'
        
    def setImage(self,cvimg):
        
        roi=QRect(0,0,256,256)
        tmp=cvimg[:,:,(2,1,0)]
        tmp=np.require(tmp, np.uint8, 'C')
        qimg=toQImage(tmp,False)
        self.picture=QtGui.QPixmap(qimg)
        
    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0,0, self.picture.scaled(self.sWidth,self.sHeight))
        painter.setFont(QFont("Comic Sans MS",20))
        painter.drawText(QPoint(100,250),self.label)
        if self.lightOn:
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(0,255,0),15))
            painter.drawRect(0,0,256,256)

class CVImageWidget(QtGui.QWidget):
    def __init__(self,cvimg,parent):
        super(CVImageWidget,self).__init__(parent)
        #self.picture=QtGui.QPixmap(path)
        roi=QRect(0,0,256,256)
        tmp=cvimg[:,:,(2,1,0)]
        tmp=np.require(tmp, np.uint8, 'C')
        qimg=toQImage(tmp,False)
        self.picture=QtGui.QPixmap(qimg).copy(roi)
        self.picture=self.picture.scaled(256,256)
        self.label=''
        print 'ok'
        
    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0,0, self.picture)
        #painter.setFont(QFont("Arial"))
        #painter.drawText(QPoint(100,100),self.label)

class CVImageRTWidget(QtGui.QWidget):
    def __init__(self,parent):
        super(CVImageRTWidget,self).__init__(parent)
        #self.picture=QtGui.QPixmap(path)
        self.picture=None
        self.label=''
        print 'ok'
        
    def setImage(self,cvimg):
        global screen_width,screen_height
        roi=QRect(0,0,256,256)
        tmp=cvimg[:,:,(2,1,0)]
        tmp=np.require(tmp, np.uint8, 'C')
        qimg=toQImage(tmp,False)
        self.picture=QtGui.QPixmap(qimg)
        #self.picture=QtGui.QPixmap(qimg).copy(roi)
        self.picture=self.picture.scaled(screen_width,screen_height)
        
        
    def paintEvent(self,event):
        global screen_width,screen_height
        painter = QtGui.QPainter(self)
        if not self.picture is None:
            painter.drawPixmap(0,0, self.picture)
        painter.setFont(QFont("Comic Sans MS",40))
        painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255),15))
        painter.drawText(QPoint(screen_width/2,screen_height-100),self.label)

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
tex,tey=-1,-1
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
container=None
rawroi=None
all_pose_startend_time={}
previousPredict=-1
pmode=0
realtimeView=None
gui2=None
gui3=None
rawframe=None
resultGUI=None
passed_poses=[]
all_poses=[1,2,3,4,5,6,7]
train_duration=2
all_pose_durations=[train_duration,train_duration,train_duration,train_duration,train_duration,train_duration,train_duration]
modeChanged=False
train_currentPose=-1
uiUpdated=True
mainMenu=None

readyDetect=False
resultBack=False
predict_idx=-1
mainTask=None

# for global timer
global_start=0
global_end=0
global_duration=60 # in second
global_count=0
#

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
    global ix,iy,ex,ey,drawing,frame,roi,lock,tex,tey
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
        ex,ey=x,y
        tex,tey=x,y
        lock=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            lock=True
            tex,tey=x,iy+x-ix
            #ex,ey=x,iy+x-ix
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        tex,tey=x,iy+x-ix
        ex,ey=x,iy+x-ix
        lock=False

class RunTask2(QtCore.QThread):
    
    
    def __init__(self,w,func):
        QtCore.QThread.__init__(self)
        self.w=w
        self.func={'func1':func}
        
    def run(self):
        while True:
            self.func['func1'](self.w)
            self.sleep(1)
    

class RunTask(QtCore.QThread):
    
    pose_updated= QtCore.pyqtSignal(object)
    frame_updated= QtCore.pyqtSignal(object)
    frame_updated2= QtCore.pyqtSignal(object)
    result_pop= QtCore.pyqtSignal(object)
    mainWindowFront=QtCore.pyqtSignal(object)
    gotoNextPose=QtCore.pyqtSignal(object)
    triggerTransition=QtCore.pyqtSignal(object)
    updateNextButton=QtCore.pyqtSignal(object)
    
    def __init__(self,w,func):
        QtCore.QThread.__init__(self)
        self.w=w
        self.func={'func1':func}
        
    def run(self):
        self.func['func1'](self.w)
        
    def updateGUI(self):
        self.pose_updated.emit('')
    
    def updateRealtime(self):
        self.frame_updated.emit('')
        
    def updateRealtime2(self):
        self.frame_updated2.emit('')
        
    def updateResult(self):
        self.result_pop.emit('')
        
    def mainWindowPop(self):
        self.mainWindowFront.emit('')
        
    def gotoNextPoseEvent(self):
        self.gotoNextPose.emit('')
        
    def updateTransition(self):
        self.triggerTransition.emit('')
        
    def doUpdateNextButton(self):
        self.updateNextButton.emit('')

class GameGUI(QtGui.QTableWidget):
    def __init__(self,parent=None):
        super(GameGUI,self).__init__(parent)
        self.img_list={}
        self.img_wdg={}
        self.setWindowTitle('Recognition for Hand Washing Steps')
        self.w=[]
        self.task=[]
        self.pose_time={}
        self.placement={}
        self.title='Game'
        self.updatePose=-1
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        
    def setTask(self,w,func):
        global game_thread
        self.task=RunTask(w,func)
        self.task.pose_updated.connect(self.updatePoseFunc)
        game_thread=self.task
        
    def resetMap(self):
        global all_pose_startend_time,countOnThis
        for i in range(7):
            imgpath=os.path.join('indexMapd','%d.jpg'%(i+1))
            print imgpath
            self.cellWidget(self.placement[str(i+1)][0],self.placement[str(i+1)][1]).setImage(cv2.imread(imgpath))
            all_pose_startend_time[str(i+1)]=[0,0]
        countOnThis=False
        
    def runTask(self):
        self.task.start()
    
    
    def updatePoseFunc(self):
        global rawroi,global_duration
        p=self.updatePose
        self.updatePose=-1
        if p>-1:
            #self.cellWidget(p,2)
            #self.setCellWidget(1,3,CVImageWidget(rawroi,self))
            #self.cellWidget(self.placement[str(p+1)][0],self.placement[str(p+1)][1]).lightOn=True
            imgpath=os.path.join('indexMap','%d.jpg'%(p+1))
            self.cellWidget(self.placement[str(p+1)][0],self.placement[str(p+1)][1]).setImage(cv2.imread(imgpath))
            print 'frame update'
            #self.cellWidget(p,2).setStyleSheet('QLabel {background-color:green}')
            #self.cellWidget(p,2).setText('Done')
        for cls in all_pose_startend_time.keys():
            #itime=global_duration-self.pose_time[cls]
            #t_min=int(np.floor(itime/60))
            #t_sec=itime-t_min*60
            t_min=int(np.floor(self.pose_time[cls]/60))
            t_sec=self.pose_time[cls]-t_min*60
            #self.cellWidget(self.placement[cls][0],self.placement[cls][1]).label='%02d:%02d'%(t_min,t_sec)
            self.cellWidget(self.placement[cls][0],self.placement[cls][1]).update()
            #tmp=self.img_wdg[cls]
        
    def load_IndexMap(self,path):
        
        global all_pose_startend_time
        all_pose_startend_time={}
        self.img_list={}
        self.img_wdg={}
        self.pose_time={}
        self.placement={}
        i=0
        self.insertColumn(0)
        self.insertColumn(1)
        self.insertColumn(2)
        self.insertColumn(3)
        title=QtCore.QStringList()
        title.append('Pose')
        title.append('Example')
        title.append('is Done')
        #sself.setHorizontalHeaderLabels(title)
        maxC=4
        r=0
        c=0
        init_time=time.time()
        for fname in os.listdir(path):
            
            class_id,fext=os.path.splitext(fname)
            fpath=os.path.join(path,fname)
            print fpath
            self.img_list[class_id]=cv2.imread(fpath)[0:256,0:256]
            qim=toQImage(self.img_list[class_id])
            self.img_wdg[class_id]=ImageWidget(fpath,self)
            self.pose_time[class_id]=0
            t_min=round(self.pose_time[class_id]/60)
            t_sec=self.pose_time[class_id]-t_min*60
            #self.img_wdg[class_id].label='%02d:%02d'%(t_min,t_sec)
            all_pose_startend_time[class_id]=[0,0]
            #QPixmap.fromImage(qim)
            #lbl=QtGui.QLabel('Pose %s'%class_id,self)
            #lblStatus=QtGui.QLabel('False',self)
            #self.setCellWidget(i,0,lbl)
            #self.setCellWidget(i,2,lblStatus)
            #print self.cellWidget(i,2).setStyleSheet('QLabel {background-color: red}')
            
            if i%maxC==0:
                
                self.insertRow(r)
                r+=1
                c=0
                self.setCellWidget(r-1,c,self.img_wdg[class_id])
                self.setRowHeight(r-1,256)
            else:
                self.setCellWidget(r-1,c,self.img_wdg[class_id])
            self.placement[class_id]=[r-1,c]
            i+=1
            c+=1
        
        #self.setCellWidget(len(self.img_list.keys())-1,3)
        #self.removeRow(i-1)
        self.setColumnWidth(0,256)
        self.setColumnWidth(1,256)
        self.setColumnWidth(2,256)
        self.setColumnWidth(3,256)
        self.resize(QtCore.QSize(1024,700))
        #self.showFullScreen()
        fg = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry()
        #fg.moveCenter(cp)
        print cp
        self.move(200,200)
        
        print all_pose_startend_time
        print self.pose_time
        return self.img_list
         
class ResultWidget(QtGui.QTableWidget):
    def __init__(self,parent=None):
        super(ResultWidget,self).__init__(parent)
        self.pose_time={}
        self.correctFrames={}
        self.img_list={}
        self.img_wdg={}
        self.wdg_pos={}
        self.model_videos={}
        self.my_rank=0
        
    def prepare(self,path):
        self.my_rank=0
        self.insertColumn(0) # no
        self.insertColumn(1) # animation
        self.insertColumn(2) # time
        self.insertColumn(3) # no
        self.insertColumn(4) # animation
        self.insertColumn(5) # time
        
        title=QtCore.QStringList()
        title.append('Pose #')
        title.append('Example')
        title.append('Duration')
        title.append('Pose #')
        title.append('Example')
        title.append('Duration')
        self.setHorizontalHeaderLabels(title)
        maxC=4
        r=0
        c=0
        rn=0
        icid=0
        init_time=time.time()
        for fname in os.listdir(path):
            class_id,fext=os.path.splitext(fname)
            if class_id!='8':
                cn=0
                if int(class_id)%2==0:
                    cn=1
                if icid%2==0:
                    rn+=1
                    self.insertRow(rn-1)
                    print rn
                self.wdg_pos[class_id]=[int(np.floor((float(class_id)-1)/2)),cn]
                self.correctFrames[class_id]=[]
                print self.wdg_pos[class_id]
            icid+=1         
                
        for fname in os.listdir(path):
            
            class_id,fext=os.path.splitext(fname)
            fpath=os.path.join(path,fname)
            #print fpath
            self.img_list[class_id]=cv2.imread(fpath)[0:256,0:256]
            qim=toQImage(self.img_list[class_id])
            self.pose_time[class_id]=0
            t_min=round(self.pose_time[class_id]/60)
            t_sec=self.pose_time[class_id]-t_min*60
            all_pose_startend_time[class_id]=[0,0]
            
            #QPixmap.fromImage(qim)
            #lbl=QtGui.QLabel('Pose %s'%class_id,self)
            #lblStatus=QtGui.QLabel('False',self)
            #self.setCellWidget(i,0,lbl)
            #self.setCellWidget(i,2,lblStatus)
            #print self.cellWidget(i,2).setStyleSheet('QLabel {background-color: red}')
            if class_id!='8':
                self.img_wdg[class_id]=ImageWidget(fpath,self)
                self.img_wdg[class_id].sWidth=150
                self.img_wdg[class_id].sHeight=150
                r=self.wdg_pos[class_id][0]
                c=self.wdg_pos[class_id][1]
                #self.insertRow(r)
                #self.setItem(r,0,QtGui.QTableWidgetItem(class_id))
                label=QtGui.QLabel(class_id,self)
                labelTime=QtGui.QLabel('00:00',self)
                label.setGeometry(0,0,100,150)
                labelTime.setGeometry(0,0,100,150)
                newfont = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
                label.setFont(newfont)
                label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                labelTime.setFont(newfont)
                labelTime.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.setCellWidget(r,3*c+0,label)
                self.setCellWidget(r,3*c+1,self.img_wdg[class_id])
                #self.setCellWidget(r,1,QtGui.QLabel(class_id,self))
                self.setCellWidget(r,3*c+2,labelTime)
                self.setRowHeight(r,150)
                #print fname
                #r+=1
        self.setColumnWidth(0,100)
        self.setColumnWidth(1,150)
        self.setColumnWidth(2,100)
        self.setColumnWidth(3,100)
        self.setColumnWidth(4,150)
        self.setColumnWidth(5,100)
        
        self.setStyleSheet("QTableWidget {background-color: transparent;}")
        self.setFrameStyle(QtGui.QFrame.NoFrame)
        self.verticalHeader().hide()
        self.resize(QtCore.QSize(750,650))
        fg = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry()
        self.move((cp.width()-fg.width())/2,100)
    def update(self):
        global gui3,passed_poses,all_poses,pmode
        self.my_rank=0
        for class_id in self.img_wdg.keys():
            if class_id!='8':
                r=self.wdg_pos[class_id][0]
                c=self.wdg_pos[class_id][1]
                t_min=round(self.pose_time[class_id]/60)
                t_sec=self.pose_time[class_id]-t_min*60
                self.cellWidget(r,c*3+2).setText('%02d:%02d'%(t_min,t_sec))
                self.cellWidget(r,c*3+2).update()
                if t_sec>0 or t_min>0:
                    if not os.path.isdir('tmp'):
                        os.mkdir('tmp')
                    gif_output=imageio.get_writer('tmp/%s.gif'%class_id, mode='I',fps=2)
                    for img in self.correctFrames[class_id]:
                        pimg=cv2.resize(img,(150,150))
                        #pimg=cv2.rectangle(img,(0,0),(256,256),(0,255,0),3)
                        gif_output.append_data(pimg[:,:,(2,1,0)])
                    gif_output.close()
                    print 'load gif'
                    lblMov=QtGui.QLabel('',self)
                    gif_anim=QMovie('tmp/%s.gif'%class_id,parent=self)
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setSpeed(500)
                    lblMov.setMovie(gif_anim)
                    gif_anim.start()
                    self.setCellWidget(r,c*3+1,lblMov)
                    self.cellWidget(r,c*3).setStyleSheet('background-color: green;')
                    self.my_rank+=1   
                    if class_id not in passed_poses:
                        passed_poses.append(class_id)
                    #self.cellWidget(r,c*3+1).setImage(self.correctFrames[class_id][0])
                    #self.cellWidget(r,c*3+1).update()
                else:
                    lblMov=QtGui.QLabel('',self)
                    gif_anim=QMovie('examples/%s.gif'%class_id,parent=self)
                    print 'examples/%s.gif'%class_id
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setSpeed(300)
                    gif_anim.setScaledSize(QtCore.QSize(150,150))
                    lblMov.setMovie(gif_anim)
                    gif_anim.start()
                    self.setCellWidget(r,c*3+1,lblMov)
                print '%02d:%02d'%(t_min,t_sec)
        if len(passed_poses)<len(all_poses):
            gui3.reinit()

class ResultWidget2(QtGui.QWidget):
    def __init__(self,parent=None):
        super(ResultWidget2,self).__init__(parent)
        self.pose_time={}
        self.correctFrames={}
        self.img_list={}
        self.img_wdg={}
        self.wdg_pos={}
        self.model_videos={}
        self.my_rank=0
        self.topList=QtGui.QTableWidget(self)
        self.bottomList=QtGui.QTableWidget(self)
        self.setStyleSheet('QTableWidget {border:0px;}')
        self.topList.setGeometry(50,100,1240,200)
        self.bottomList.setGeometry(50,350,1240,200)
        self.topList.setStyleSheet('QTableView::item {border:0px;}')
        self.bottomList.setStyleSheet('QTableView::item {border:0px;}')
        self.topList.setShowGrid(False)
        self.bottomList.setShowGrid(False)
        
        newfont = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
        #self.lblTop=QtGui.QLabel('Done',self)
        #self.lblTop.setFont(newfont)
        #self.lblTop.setGeometry(25,0,200,40)
        
        #self.lblBottom=QtGui.QLabel('Missing',self)
        #self.lblBottom.setFont(newfont)
        #self.lblBottom.setStyleSheet('color: red')
        #self.lblBottom.setGeometry(25,300,200,45)
        
    def resetTime(self):
        global gui3,passed_poses,all_pose_startend_time
        init_time=time.time()
        for class_id in self.img_wdg.keys():
            if class_id!='8':
                if int(class_id) in passed_poses:
                    continue
                all_pose_startend_time[class_id]=[0,0]
        
    def prepare(self,path):
        global screen_width,screen_height
        self.my_rank=0
        
        maxC=4
        r=0
        c=0
        rn=0
        icid=0
        init_time=time.time()
        for fname in os.listdir(path):
            class_id,fext=os.path.splitext(fname)
            if class_id!='8':
                cn=0
                if int(class_id)%2==0:
                    cn=1
                if icid%2==0:
                    rn+=1
                    #self.insertRow(rn-1)
                    print rn
                self.wdg_pos[class_id]=[int(np.floor((float(class_id)-1)/2)),cn]
                self.correctFrames[class_id]=[]
                print self.wdg_pos[class_id]
            icid+=1         
                
        for fname in os.listdir(path):
            
            class_id,fext=os.path.splitext(fname)
            fpath=os.path.join(path,fname)
            #print fpath
            self.img_list[class_id]=cv2.imread(fpath)[0:256,0:256]
            qim=toQImage(self.img_list[class_id])
            self.pose_time[class_id]=0
            t_min=np.floor(self.pose_time[class_id]/60)
            t_sec=self.pose_time[class_id]-t_min*60
            all_pose_startend_time[class_id]=[0,0]
            if class_id!='8':
                self.img_wdg[class_id]=fpath
                #self.img_wdg[class_id]=ImageWidget(fpath,self)
                #self.img_wdg[class_id].sWidth=150
                #self.img_wdg[class_id].sHeight=150
                #r=self.wdg_pos[class_id][0]
                #c=self.wdg_pos[class_id][1]
                #label=QtGui.QLabel(class_id,self)
                #labelTime=QtGui.QLabel('00:00',self)
                #label.setGeometry(0,0,100,150)
                #labelTime.setGeometry(0,0,100,150)
                #newfont = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
                #label.setFont(newfont)
                #label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                #labelTime.setFont(newfont)
                #labelTime.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                #self.setCellWidget(r,3*c+0,label)
                #self.setCellWidget(r,3*c+1,self.img_wdg[class_id])
                #self.setCellWidget(r,1,QtGui.QLabel(class_id,self))
                #self.setCellWidget(r,3*c+2,labelTime)
                #self.setRowHeight(r,150)
                #print fname
                #r+=1
        
        self.setStyleSheet("QTableWidget {background-color: transparent;}")
        self.topList.setFrameStyle(QtGui.QFrame.NoFrame)
        self.bottomList.setFrameStyle(QtGui.QFrame.NoFrame)
        self.topList.verticalHeader().hide()
        self.topList.horizontalHeader().hide()
        self.bottomList.verticalHeader().hide()
        self.bottomList.horizontalHeader().hide()
        self.resize(QtCore.QSize(screen_width,int(np.floor(screen_height*0.8))))
        self.topList.resize(QtCore.QSize(screen_width,int(np.floor(screen_height*0.8))))
        #fg = self.frameGeometry()
        #cp = QtGui.QDesktopWidget().availableGeometry()
        #self.move((cp.width()-fg.width())/2,100)
    def update(self):
        global gui3,passed_poses,all_poses,pmode
        self.topList.setRowCount(2)
        self.topList.setRowHeight(0,150)
        self.topList.setRowHeight(1,25)
        self.topList.setColumnCount(0)
        
        self.bottomList.setRowCount(1)
        self.bottomList.setRowHeight(0,150)
        self.bottomList.setColumnCount(0)
        self.my_rank=0
        c=0
        wc=0
        newfont = QtGui.QFont("Comic Sans MS",16, QtGui.QFont.Bold)
        mask_img=cv2.imread('resources/done_square.png',cv2.IMREAD_UNCHANGED)
        mask_img3=cv2.cvtColor(mask_img,cv2.COLOR_BGRA2BGR)
        ret,bmask=cv2.threshold(mask_img[:,:,3],254,255,cv2.THRESH_BINARY)
        masked_mask=cv2.bitwise_and(mask_img3,mask_img3,mask=bmask)
        masked_mask=cv2.resize(masked_mask,(150,150))
        for class_id in self.img_wdg.keys():
            if class_id!='8':
                #r=self.wdg_pos[class_id][0]
                #c=self.wdg_pos[class_id][1]
                t_min=np.floor(self.pose_time[class_id]/60)
                t_sec=self.pose_time[class_id]-t_min*60
                print '%s: %f,%f,%f'%(class_id,self.pose_time[class_id],t_min,t_sec)
                #self.cellWidget(r,c*3+2).setText('%02d:%02d'%(t_min,t_sec))
                #self.cellWidget(r,c*3+2).update()
                
                if t_sec>0 or t_min>0:
                    if not os.path.isdir('tmp'):
                        os.mkdir('tmp')
                    gif_output=imageio.get_writer('tmp/%s.gif'%class_id, mode='I',fps=2)
                    for img in self.correctFrames[class_id]:
                        pimg=cv2.resize(img,(150,150))
                        pcvframe=cv2.addWeighted(pimg,0.3,masked_mask,0.7,20)
                        #pimg=cv2.rectangle(img,(0,0),(256,256),(0,255,0),3)
                        #gif_output.append_data(pimg[:,:,(2,1,0)])
                        gif_output.append_data(pcvframe[:,:,(2,1,0)])
                    gif_output.close()
                    print 'load gif'
                    lblMov=QtGui.QLabel('',self)
                    lblTime=QtGui.QLabel('%02d:%02d'%(t_min,t_sec),self)
                    lblTime.setFont(newfont)
                    lblTime.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    gif_anim=QMovie('tmp/%s.gif'%class_id,parent=self)
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setScaledSize(QtCore.QSize(125,125))
                    gif_anim.setSpeed(500)
                    lblMov.setMovie(gif_anim)
                    lblMov.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    gif_anim.start()
                    self.topList.insertColumn(c)
                    self.topList.setCellWidget(0,c,lblMov)
                    self.topList.setCellWidget(1,c,lblTime)
                    self.topList.setColumnWidth(c,130)
                    #self.setCellWidget(r,c*3+1,lblMov)
                    #self.cellWidget(r,c*3).setStyleSheet('background-color: green;')
                    self.my_rank+=1   
                    if class_id not in passed_poses:
                        passed_poses.append(int(class_id))
                    c+=1
                    #self.cellWidget(r,c*3+1).setImage(self.correctFrames[class_id][0])
                    #self.cellWidget(r,c*3+1).update()
                else:
                    lblMov=QtGui.QLabel('',self)
                    gif_anim=QMovie('missing_imgs/%s.gif'%class_id,parent=self)
                    print 'missing_imgs/%s.gif'%class_id
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setSpeed(300)
                    gif_anim.setScaledSize(QtCore.QSize(125,125))
                    lblMov.setMovie(gif_anim)
                    lblMov.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    gif_anim.start()
                    self.bottomList.insertColumn(wc)
                    self.bottomList.setCellWidget(0,wc,lblMov)
                    self.bottomList.setColumnWidth(wc,130)
                    
                    wc+=1
                print '%02d:%02d'%(t_min,t_sec)
        if len(passed_poses)<len(all_poses):
            gui3.reinit()

            
class ResultContainer(QtGui.QWidget):
    def __init__(self,parent=None):
        global screen_height,screen_width
        super(ResultContainer,self).__init__(parent)
        self.setQAction=QtGui.QAction('quit',self,shortcut=Qt.Key_Q, triggered=self.setQ)
        self.addAction(self.setQAction)
        self.rank=0 # 0-F, 1-C, 2-B, 3-A
        self.ranks=['F','C','B','A']
        self.lblRank=QtGui.QLabel(self)
        self.lblStaticRank=QtGui.QLabel('Rank',self)
        self.lblStaticRank.setGeometry(screen_width*0.8,600,140,100)
        self.lblRank.setGeometry(self.lblStaticRank.geometry().x()+140,600,100,100)
        #self.poseList=ResultWidget(self)
        self.poseList=ResultWidget2(self)
        self.poseList.setGeometry(0,20,640,480)
        newfont = QtGui.QFont("Comic Sans MS",68, QtGui.QFont.Bold)
        newfont2 = QtGui.QFont("Comic Sans MS",46, QtGui.QFont.Bold)
        newfont3 = QtGui.QFont("Comic Sans MS",12, QtGui.QFont.Bold)
        newfont4 = QtGui.QFont("Comic Sans MS",36, QtGui.QFont.Bold)
        self.lblRank.setText('%s'%self.ranks[self.rank])
        self.lblStaticRank.setFont(newfont2)
        self.lblRank.setFont(newfont)
        self.lblRank.setStyleSheet('color: red')
        self.lblRank.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.picture=QtGui.QPixmap('resources/WASHING-HANDS.png')
        
        self.btnTrain=QtGui.QPushButton('Train missing poses',self)
        self.btnTrain.setGeometry(50,screen_height-100,180,50)
        self.btnTrain.setFont(newfont3)
        self.btnTrain.clicked.connect(self.retrain)
        
        self.btnReturn=QtGui.QPushButton('Back to Main Menu',self)
        self.btnReturn.setGeometry(300,screen_height-100,180,50)
        self.btnReturn.setFont(newfont3)
        self.btnReturn.clicked.connect(self.backToMenu)
        
        self.lblHeading=QtGui.QLabel('Result',self)
        self.lblHeading.setAlignment(Qt.AlignCenter)
        self.lblHeading.setGeometry(0,0,screen_width,150)
        self.lblHeading.setFont(newfont4)
        
    def setQ(self):
        print 'Quit'
        QCoreApplication.quit()
        
    def reset(self):
        self.rank=0
        self.lblRank.setText('%s'%self.ranks[self.rank])
        
    def retrain(self):
        global gui3,pmode,cool_down_start,cool_down_time,uiUpdated,game_thread,threadStarted
        gui3.nextPoseDummy()
        gui3.showFullScreen()
        pmode=2
        gui3.nextGoBack=True
        if not threadStarted:
            game_thread.start()
        
    def backToMenu(self):
        global pmode,gui,gui2,gui3,mainMenu
        pmode=-1
        mainMenu.showFullScreen()
        
    def updateRank(self):
        self.rank=int(float(self.poseList.my_rank)*len(self.ranks))/7
        self.lblRank.setText('%s'%self.ranks[self.rank])

    def paintEvent(self,event):
        #gamescene/WASHING-HANDS.jpg
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0,0, self.picture)

class Game2GUI(QtGui.QWidget):
    def __init__(self,parent=None):
        super(Game2GUI,self).__init__(parent)
        global realtimeView
        self.resultView=ResultWidget(self)
        realtimeView=CVImageRTWidget(self)
        
        
        newfont = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
        newfont2 = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
        self.lblHeader=QtGui.QLabel('Wash Your Hands Freely',self)
        self.lblHeader.setStyleSheet('background-color: rgba(255, 255, 255, 0);color: rgba(255, 255, 255, 255);')
        self.lblHeader.setGeometry((screen_width-800)/2,0,800,100)
        self.lblHeader.setAlignment(Qt.AlignCenter)
        self.lblHeader.setFont(newfont)
        
        self.lblTime=QtGui.QLabel('',self)
        self.lblTime.setStyleSheet('background-color: rgba(255, 255, 255, 0);color: rgba(255, 255, 255, 255);')
        self.lblTime.setGeometry((screen_width-300)/2,680,300,100)
        self.lblTime.setAlignment(Qt.AlignCenter)
        self.lblTime.setFont(newfont)
    
    def frameUpdate(self):
        global rawroi,realtimeView,rawframe,global_start,global_end,global_count,global_duration,screen_width,screen_height
        diff=global_count-global_start
        itime=global_duration-diff
        t_min=int(np.floor(itime/60))
        t_sec=itime-t_min*60
        #t_min=int(np.floor(diff/60))
        #t_sec=diff-t_min*60
        #realtimeView.label='%02d:%02d'%(t_min,t_sec)
        self.lblTime.setText('%02d:%02d'%(t_min,t_sec))
        realtimeView.setImage(rawframe)
        realtimeView.update()
        
    def resultUpdate(self):
        global resultGUI
        resultGUI.poseList.update()
        resultGUI.updateRank()
        resultGUI.poseList.resetTime()
        
    def setSharedTask(self):
        global game_thread 
        game_thread.frame_updated.connect(self.frameUpdate)
        game_thread.result_pop.connect(self.resultUpdate)
        
    def bringMeFront(self):
        self.activateWindow()

class MainContainer(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MainContainer,self).__init__(parent)
        self.setQAction=QtGui.QAction('quit',self,shortcut=Qt.Key_Q, triggered=self.setQ)
        self.addAction(self.setQAction)
        self.setMAction=QtGui.QAction('back',self,shortcut=Qt.Key_M, triggered=self.setM)
        self.addAction(self.setMAction)
        
    def bringMeFront(self):
        self.activateWindow()
    
    def setQ(self):
        print 'Quit'
        QCoreApplication.quit()
        
    def setM(self):
        print 'back to main menu'
        global pmode,gui,gui2,gui3,mainMenu
        pmode=-1
        mainMenu.showFullScreen()

class Game3GUI(QtGui.QWidget):
    def __init__(self,parent=None):
        global screen_width,screen_height,infos,transitions
        super(Game3GUI,self).__init__(parent)
        #self.layout=QVBoxLayout()
        self.setQAction=QtGui.QAction('quit',self,shortcut=Qt.Key_Q,triggered=self.setQ)
        self.addAction(self.setQAction)
        self.todo_poses=[]
        self.nextGoBack=False
        
        
        
        pwidth=screen_width/2.5
        pheight=pwidth
        ptop=(screen_height-pheight)/2
        
        print 'pos=[%d,%d,%d,%d]'%(0,ptop,pwidth,pheight)
        self.setAutoFillBackground(True)
        p=self.palette()
        p.setColor(self.backgroundRole(),Qt.black)
        self.setPalette(p)
        
        left=((screen_width-pwidth*1.2)/2)-pwidth/2
        
        self.leftPanel=QtGui.QLabel('',self)
        self.leftPanel.setGeometry(left,ptop,pwidth,pheight)
        self.rightPanel=ImageWidget('',self)
        self.rightPanel.sWidth=pwidth
        self.rightPanel.sHeight=pwidth
        self.rightPanel.setGeometry(left+pwidth*1.2,ptop,pwidth,pheight)
        
        p2=self.rightPanel.palette()
        p2.setColor(self.rightPanel.backgroundRole(),Qt.white)
        self.rightPanel.setPalette(p2)
        #self.layout().addWidget(self.leftPanel)
        #self.layout().addWidget(self.rightPanel)
        
        self.lblHeading=QtGui.QLabel('Please follow the instruction',self)
        self.lblInfo=QtGui.QLabel(infos[1],self)
        
        newfont2 = QtGui.QFont("Comic Sans MS",24, QtGui.QFont.Bold)
        
        self.lblHeading.setAlignment(Qt.AlignCenter)
        self.lblHeading.setGeometry((screen_width-800)/2,20,800,50)
        #self.lblHeading.setAutoFillBackground(True)
        self.lblHeading.setStyleSheet('color:#ffffff')
        self.lblInfo.setGeometry(50,screen_height-100,screen_width*0.8,50)
        #self.lblInfo.setAutoFillBackground(True)
        self.lblInfo.setStyleSheet('color:#ffffff')
        
        self.lblHeading.setFont(newfont2)
        self.lblInfo.setFont(newfont2)
        
        newfont3 = QtGui.QFont("Comic Sans MS",36, QtGui.QFont.Bold)
        self.lblTransition=QtGui.QLabel(transitions[1],self)
        self.lblTransition.setGeometry(0,0,screen_width,screen_height)
        self.lblTransition.setAlignment(Qt.AlignCenter)
        self.lblTransition.setFont(newfont3)
        self.lblTransition.setStyleSheet('color:#ffffff')
        self.triggerTransition()
        
        self.lblHeading.hide()
        self.lblInfo.hide()
        self.leftPanel.hide()
        self.rightPanel.hide()
        
        self.btnNext=QtGui.QPushButton('>>',self)
        self.btnNext.setFont(newfont3)
        self.btnNext.setGeometry((screen_width-120),screen_height*0.88,120,100)
        self.btnNext.clicked.connect(self.clickNextPose)
        self.btnNext.hide()
        
    def setQ(self):
        print 'Quit'
        QCoreApplication.quit()
        
    def trigNextButton(self):
        if len(self.todo_poses)==1:
            self.btnNext.setText('Done')
        else:
            self.btnNext.setText('>>')
        if current_pose_duration>=current_pose_duration_limit and not self.btnNext.isVisible():
            self.btnNext.show()
            
        
    def clickNextPose(self):
        global current_pose_duration,current_pose_duration_limit,current_pose_startend,mainMenu,cool_down_start,cool_down_time
        current_pose_duration=0
        current_pose_startend=[0,0]
        if len(self.todo_poses)==1:
            pmode=-1
            self.reinit()
            mainMenu.showFullScreen()
        else:
            self.todo_poses.pop(0)
            train_currentPose=self.todo_poses[0]
            self.lblInfo.setText(infos[train_currentPose])
            self.lblTransition.setText(transitions[train_currentPose])
            self.triggerTransition()
            self.lblTransition.show()
            self.lblHeading.hide()
            self.lblInfo.hide()
            self.leftPanel.hide()
            self.rightPanel.hide()
            cool_down_time=3
            cool_down_start=0
            gif_anim=QMovie('examples/%s.gif'%self.todo_poses[0],parent=self)
            gif_anim.setCacheMode(QMovie.CacheAll)
            gif_anim.setSpeed(500)
            box=self.leftPanel.geometry()
            gif_anim.setScaledSize(QtCore.QSize(box.width(),box.height()))
            self.leftPanel.setMovie(gif_anim)
            gif_anim.start()
            uiUpdated=True
            
        self.btnNext.hide()
        
    def setTestMode(self):
        global modeChanged,pmode,mainMenu
        if self.nextGoBack:
            pmode=-1
            mainMenu.showFullScreen()
            self.nextGoBack=False
        else:
            print 'swap to validation mode'
            pmode=1
            modeChanged=True
        
    def reinit(self):
        global all_poses,passed_poses,train_currentPose,all_pose_durations,train_duration
        self.todo_poses=[]
        for pose in range(len(all_poses)):
            if all_poses[pose] in passed_poses:
                continue
            print pose
            self.todo_poses.append(all_poses[pose])
            all_pose_durations[all_poses[pose]-1]=train_duration
        train_currentPose=self.todo_poses[0]
        gif_anim=QMovie('examples/%s.gif'%train_currentPose,parent=self)
        gif_anim.setCacheMode(QMovie.CacheAll)
        gif_anim.setSpeed(500)
        box=self.leftPanel.geometry()
        gif_anim.setScaledSize(QtCore.QSize(box.width(),box.height()))
        self.leftPanel.setMovie(gif_anim)
        gif_anim.start()
        self.leftPanel.update()
    
    def nextPoseDummy(self):
        global all_poses,passed_poses,train_currentPose,uiUpdated,infos,transitions,cool_down_start,cool_down_time
        train_currentPose=self.todo_poses[0]
        self.lblInfo.setText(infos[train_currentPose])
        self.lblTransition.setText(transitions[train_currentPose])
        self.triggerTransition()
        self.lblTransition.show()
        self.lblHeading.hide()
        self.lblInfo.hide()
        self.leftPanel.hide()
        self.rightPanel.hide()
        cool_down_time=3
        cool_down_start=0
        gif_anim=QMovie('examples/%s.gif'%self.todo_poses[0],parent=self)
        gif_anim.setCacheMode(QMovie.CacheAll)
        gif_anim.setSpeed(500)
        box=self.leftPanel.geometry()
        gif_anim.setScaledSize(QtCore.QSize(box.width(),box.height()))
        self.leftPanel.setMovie(gif_anim)
        gif_anim.start()
        uiUpdated=True
        
    def nextPose(self):
        global all_poses,passed_poses,train_currentPose,uiUpdated,infos,transitions,cool_down_start,cool_down_time
        self.todo_poses.pop(0)
        train_currentPose=self.todo_poses[0]
        self.lblInfo.setText(infos[train_currentPose])
        self.lblTransition.setText(transitions[train_currentPose])
        self.triggerTransition()
        self.lblTransition.show()
        self.lblHeading.hide()
        self.lblInfo.hide()
        self.leftPanel.hide()
        self.rightPanel.hide()
        cool_down_time=3
        cool_down_start=0
        gif_anim=QMovie('examples/%s.gif'%self.todo_poses[0],parent=self)
        gif_anim.setCacheMode(QMovie.CacheAll)
        gif_anim.setSpeed(500)
        box=self.leftPanel.geometry()
        gif_anim.setScaledSize(QtCore.QSize(box.width(),box.height()))
        self.leftPanel.setMovie(gif_anim)
        gif_anim.start()
        uiUpdated=True
        
    def triggerTransition(self):
        global cool_down_time,cool_down_start
        if cool_down_time<=0:
            self.lblTransition.hide()
            self.lblHeading.show()
            self.lblInfo.show()
            self.leftPanel.show()
            self.rightPanel.show()
            print 'triggerTransition Off'
        else:
            self.lblTransition.show()
            self.lblHeading.hide()
            self.lblInfo.hide()
            self.leftPanel.hide()
            self.rightPanel.hide()
            print 'triggerTransition On'
        
    def updateFrame(self):
        global rawframe
        self.rightPanel.setImage(rawframe)
        self.rightPanel.update()
        
    def wireUpEvent(self):
        global game_thread
        game_thread.gotoNextPose.connect(self.nextPose)
        game_thread.frame_updated2.connect(self.updateFrame)
        game_thread.triggerTransition.connect(self.triggerTransition)
        game_thread.updateNextButton.connect(self.trigNextButton)
        
    def bringMeFront(self):
        self.activateWindow()
        
    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        #painter.drawPixmap(0,0, self.picture)
        
class MainMenuUI(QtGui.QWidget):
    def __init__(self,parent=None):
        global pmode,screen_height,screen_width
        super(MainMenuUI,self).__init__(parent)    
        self.picture=QtGui.QPixmap('resources/WASHING-HANDS.png')
        self.Heading=QtGui.QLabel('Hand-Washing Training System',self)
        self.trainingMode=QtGui.QPushButton('Quiz Mode',self)
        self.testMode=QtGui.QPushButton('Learning Mode',self)
        self.freeMode=QtGui.QPushButton('Free Mode',self)
        self.exitGame=QtGui.QPushButton('Exit',self)
        
        self.Heading.setGeometry((screen_width-800)/2,10,900,100)
        self.Heading.setAlignment(Qt.AlignCenter)
        self.trainingMode.setGeometry((screen_width-300)/2,270,300,50)
        self.testMode.setGeometry((screen_width-300)/2,200,300,50)
        self.freeMode.setGeometry((screen_width-300)/2,350,300,50)
        self.exitGame.setGeometry((screen_width-300)/2,700,300,50)
        
        newfont1 = QtGui.QFont("Comic Sans MS",42, QtGui.QFont.Bold)
        newfont2 = QtGui.QFont("Comic Sans MS",18, QtGui.QFont.Bold)
        self.Heading.setFont(newfont1)
        self.trainingMode.setFont(newfont2)
        self.testMode.setFont(newfont2)
        self.freeMode.setFont(newfont2)
        self.exitGame.setFont(newfont2)
        
        self.trainingMode.clicked.connect(self.gotoTrainingMode)
        self.testMode.clicked.connect(self.gotoTestMode)
        self.freeMode.clicked.connect(self.gotoFreeMode)
        self.exitGame.clicked.connect(self.exitGameEvent)
        
    def gotoTrainingMode(self):
        global pmode,gui,gui2,gui3,resultGUI,realtimeView,game_thread,threadStarted,replayOnly
        if not threadStarted:
            game_thread.start()
        self.resetAll()
        pmode=1
        resultGUI.showFullScreen()
        gui2.showFullScreen()
        realtimeView.setGeometry(QtCore.QRect(0,0,screen_width,screen_height))
        gui.task.mainWindowFront.connect(gui2.bringMeFront)
        #cv2.namedWindow('real time classifier',cv2.WINDOW_NORMAL)
        #cv2.moveWindow('real time classifier',50,50)
    
    def gotoTestMode(self):
        global pmode,gui,gui2,gui3,resultGUI,realtimeView,game_thread,threadStarted,replayOnly
        replayOnly=False
        if not threadStarted:
            game_thread.start()
        self.resetAll()
        pmode=2
        resultGUI.showFullScreen()
        gui2.showFullScreen()
        realtimeView.setGeometry(QtCore.QRect(0,0,screen_width,screen_height))
        gui3.showFullScreen()
        gui3.leftPanel.update()
        gui.task.mainWindowFront.connect(gui3.bringMeFront)
        #cv2.namedWindow('real time classifier',cv2.WINDOW_NORMAL)
        #cv2.moveWindow('real time classifier',50,50)
    
    def gotoFreeMode(self):
        global pmode,gui,gui2,gui3,resultGUI,realtimeView,game_thread,threadStarted
        if not threadStarted:
            game_thread.start()
        self.resetAll()
        pmode=0
        gui.resetMap()
        container.showFullScreen()
        gui.task.mainWindowFront.connect(container.bringMeFront)
        #cv2.namedWindow('real time classifier',cv2.WINDOW_NORMAL)
        #cv2.moveWindow('real time classifier',50,50)
    
    def exitGameEvent(self):
        cv2.destroyAllWindows()
        QCoreApplication.quit()
        
    def resetAll(self):
        global all_pose_durations,all_pose_startend_time,gui,resultGUI
        for cls in all_pose_startend_time.keys():
            all_pose_startend_time[cls]=[0,0]
            gui.pose_time[cls]=0
            resultGUI.poseList.pose_time[cls]=0
            
        for i in range(len(all_pose_durations)):
            all_pose_durations[i]=train_duration
            
    def paintEvent(self,event):
        #gamescene/WASHING-HANDS.jpg
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0,0, self.picture)
        
def detectionLoop(w=[]):
    global readyDetect,resultBack,predict_idx,classifier,bufferSize,bufferImg
    if not readyDetect:
        return
    else:
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
        predict_idx=np.argmax(class_scs)
        readyDetect=False
        resultBack=True
    #while True:
    #    if classifier !=[]:
    #        continue
    #    if not readyDetect:
    #        continue
    #    else:
    #        class_scs=[]
    #        for ii in range(bufferSize):
    #            proi=bufferImg[ii]
    #            cimg=proi
    #            if len(cimg.shape)>2:
    #                cimg=cimg[:,:,(2,1,0)]
    #            else:
    #                cimg=cimg.reshape(cimg.shape[0], cimg.shape[1],1 )
    #            predicts=classifier.predict([cimg])
    #            if w!=[]:
    #                predicts=np.dot(predicts,w)
    #            if class_scs==[]:
    #                class_scs=predicts
    #            else:
    #                class_scs=np.add(class_scs,predicts)
    #        max_idx=np.argmax(class_scs)
    #        readyDetect=False
    #        resultBack=True
        
def realtime(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui,game_thread,rawroi,rawframe,previousPredict,pmode,global_start,global_end,global_count,resultGUI,tex,tey,camID,gui2,gui3,all_pose_durations,uiUpdated,modeChanged,predict_idx,resultBack,readyDetect,mainTask,mainMenu,cool_down_time,cool_down_start,threadStarted,current_pose_startend,current_pose_duration,current_pose_duration_limit,isManualNext
    #game_scene_img=cv2.imread(game_scene)
    threadStarted=True
    cam=cv2.VideoCapture(camID)
    expected_pose=(0,2,1,3,6,4,5)
    isExit=False
    maxTmpFrame=10
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
    criteria=2
    
    behide=True
    restart=True
    train_pose_startTime=-1
    train_pose_endTime=-1
    countOnThis=False
    max_idx=7
    while restart:
        restart=False
        if pmode==99 or pmode==-1:
            ret,img=cam.read()
            cv2.rectangle(img,(ix,iy),(tex,tey),(0,255,0),3)
            cv2.imshow('real time classifier',img)
            key=cv2.waitKey(10)
            if key==27:
                isExit=True
                cv2.destroyAllWindows()
                break
            elif key==103:
                mode+=1
                if mode>2:
                    mode=0
            elif key==109:
                if pmode==0:
                    pmode=1
                    gui2.showFullScreen()
                    gui.hide()
                    behide=True
                else:
                    pmode=0
                    gui.showFullScreen()
                    gui2.hide()
                    behide=True
            restart=True
        else:
            while True:
                ret,img=cam.read()
                frame=img.copy()
                rawimg=img.copy()
                
                if pmode==1 and compute_area((ix,iy), (ex,ey))>1:
                    global_count=time.time()
                    if global_start==0:
                        global_start=global_count
                        global_end=global_start+global_duration
                    elif global_count>=global_end:
                        print 'time\'s up'
                        global_start=0
                        pmode=99
                        break
                    
                cv2.putText(img,modes[mode],(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
                frame=preprocess(frame,mode)
                cv2.rectangle(img,(ix,iy),(tex,tey),(0,255,0),3)
                cv2.imshow('real time classifier',img)
                #scene=game_scene_img.copy()
                pbbox=game_pose_roi[currentPose]
                sroi_color=(0,0,255)
                
                if compute_area((ix,iy), (ex,ey))>1:
                    if behide:
                        game_thread.mainWindowPop()
                        behide=False
                    roi=frame[iy:ey,ix:ex]
                    rawroi=rawimg[iy:ey,ix:ex]
                    center=((ex-ix)/2,(ey-iy)/2)
                    M=cv2.getRotationMatrix2D(center,-90,1.0)
                    rawroi=cv2.warpAffine(rawroi,M,(rawroi.shape[1],rawroi.shape[0]))
                    roi=cv2.warpAffine(roi,M,(roi.shape[1],roi.shape[0]))
                    if pmode==1:
                        gwidth=800
                        if gwidth>=screen_height:
                            gwidth=600
                        rroi=cv2.resize(rawroi.copy(),(gwidth,gwidth))
                        idim=rroi.shape
                        rawframe= cv2.copyMakeBorder(rroi,(screen_height-idim[0])/2,(screen_height-idim[0])/2,(screen_width-idim[1])/2,(screen_width-idim[1])/2,cv2.BORDER_CONSTANT,value=[0,0,0])
                        #cv2.putText(rawframe,'Wash Your Hand Freely',(int(np.floor((screen_width-idim[1])/1.5)),100),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255))
                        game_thread.updateRealtime()
                    elif pmode==2:
                        rroi=cv2.resize(rawroi.copy(),(800,800))
                        idim=rroi.shape
                        rawframe=rroi
                        game_thread.updateRealtime2()
                        
                    roi=cv2.resize(roi,(256,256))
                    bufferImg[countImg]=roi
                    countImg+=1
                    if countImg>=bufferSize:
                        countImg=0
                        if lock==False:
                            if classifier !=[]:
                                
                                if resultBack:
                                    max_idx=predict_idx
                                    resultBack=False
                                    countOnThis=True
                                    print max_idx
                                if not readyDetect:
                                    readyDetect=True
                                    mainTask.start()
                                class_idx=max_idx+1
                                
                                #print len(expected_pose)
                                if max_idx>-1 and max_idx<8:
                                    class_sidx=str(class_idx)
                                    if pmode==1:
                                        if class_sidx in resultGUI.poseList.correctFrames.keys():
                                            if len(resultGUI.poseList.correctFrames[class_sidx])<maxTmpFrame:
                                                resultGUI.poseList.correctFrames[class_sidx].append(rawroi.copy())
                                                print 'catched frame'
                                if pmode!=2:
                                    if countOnThis:
                                        if class_idx>-1 and class_idx<8 and class_idx==previousPredict and previousPredict>-1:
                                            cutime=time.time()
                                            if all_pose_startend_time[str(class_idx)][0]==0:
                                                all_pose_startend_time[str(class_idx)][0]=cutime
                                            all_pose_startend_time[str(class_idx)][1]=cutime
                                            gui.pose_time[str(class_idx)]+=round(all_pose_startend_time[str(class_idx)][1]-all_pose_startend_time[str(class_idx)][0])
                                            if pmode==1:
                                                resultGUI.poseList.pose_time[str(class_idx)]+=round(all_pose_startend_time[str(class_idx)][1]-all_pose_startend_time[str(class_idx)][0])
                                            all_pose_startend_time[str(class_idx)][0]=all_pose_startend_time[str(class_idx)][1]
                                            print gui.pose_time
                                        elif class_idx>-1 and class_idx<8 and class_idx!=previousPredict:
                                            all_pose_startend_time[str(class_idx)][0]=time.time()
                                            all_pose_startend_time[str(class_idx)][1]=all_pose_startend_time[str(class_idx)][0]
                                else:
                                    currentPose=gui3.todo_poses[0]-1
                                    if uiUpdated:
                                        if cool_down_time>0:
                                            game_thread.updateTransition()
                                            ltime=time.time()
                                            if cool_down_start==0:
                                                cool_down_start=ltime
                                            cool_down_time-=ltime-cool_down_start
                                            cool_down_start=ltime
                                            print cool_down_time
                                        else:
                                            game_thread.updateTransition()
                                            if isManualNext:
                                                #if currentPose==class_idx:
                                                if class_idx>0 and class_idx<8: # cheat for debug
                                                    if current_pose_duration<current_pose_duration_limit:
                                                        ltime=time.time()
                                                        if current_pose_startend[0]==0:
                                                            current_pose_startend[0]=ltime
                                                        current_pose_startend[1]=ltime
                                                        current_pose_duration+=current_pose_startend[1]-current_pose_startend[0]
                                                        current_pose_startend[0]=ltime
                                                    else:
                                                        game_thread.doUpdateNextButton()
                                            else:
                                                if all_pose_durations[currentPose]>0:
                                                    ltime=time.time()
                                                    if train_pose_startTime==-1:
                                                        train_pose_startTime=ltime
                                                    train_pose_endTime=ltime
                                                    durationSec=train_pose_endTime-train_pose_startTime
                                                    train_pose_startTime=ltime
                                                    all_pose_durations[currentPose]-=durationSec
                                                else:
                                                    train_pose_startTime=-1
                                                    if len(gui3.todo_poses)==1 and not gui3.nextGoBack:
                                                        modeChanged=True
                                                        pmode=1
                                                        gui2.showFullScreen()
                                                        break
                                                    elif len(gui3.todo_poses)==1 and gui3.nextGoBack:
                                                        modeChanged=True
                                                        pmode=-1
                                                        mainMenu.showFullScreen()
                                                        gui3.nextGoBack=False
                                                        break
                                                    else:
                                                        uiUpdated=False
                                                        game_thread.gotoNextPoseEvent()
                                                    #if currentPose==class_idx:
                                                    #    if train_pose_startTime==-1:
                                                    #        train_pose_startTime=time.time()
                                                    #    train_pose_endTime=time.time()
                                                    #    durationSec=train_pose_endTime-train_pose_startTime
                                                    #    all_pose_durations[currentPose]-=durationSec
                                                    #    if all_pose_durations[currentPose]<=0:
                                                    #        train_pose_startTime=-1
                                                    #        if len(gui3.todo_poses)==1:
                                                    #            modeChanged=True
                                                    #            pmode=1
                                                    #            break
                                                    #        else:
                                                    #            uiUpdated=False
                                                    #            game_thread.gotoNextPoseEvent()
                                                    #else:
                                                    #    train_pose_startTime=-1                                        
                                
                                
                                previousPredict=class_idx
                                if max_idx==len(expected_pose):
                                    del hist[:]
                                    hist=[]
                                else:
                                    hist.append(max_idx)
                                    if len(hist)>=criteria:
                                        seq=hist[len(hist)-criteria:]
                                        if len(set(seq))==1:
                                            # UI Callback for update
                                            print 'Passed for %d'%(max_idx+1)
                                            gui.updatePose=max_idx
                                            if pmode==0:
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
                                
                                #gui.pose_time[]
                                
                                countOnThis=False
                #cv2.rectangle(scene,(box_startAt[1]+pbbox[1],box_startAt[0]+pbbox[0]),(box_startAt[1]+pbbox[3],box_startAt[0]+pbbox[2]),sroi_color,3)
                key=cv2.waitKey(1)
                if key==27:
                    isExit=True
                    cv2.destroyAllWindows()
                    break
                elif key==103:
                    mode+=1
                    if mode>2:
                        mode=0
                elif key==109:
                    if pmode==0:
                        pmode=1
                        gui2.showFullScreen()
                        gui.hide()
                        behide=True
                    else:
                        pmode=0
                        gui.showFullScreen()
                        gui2.hide()
                        behide=True
            
            if pmode==99:
                cv2.destroyAllWindows()
                gui2.hide()
                pmode=99
                restart=True
                game_thread.updateResult()
            if modeChanged:
                restart=True
                if pmode==1:
                    gui.task.mainWindowFront.connect(gui2.bringMeFront)
                    gui3.hide()
                    gui2.showFullScreen()
                    print 'replay'
                    behide=True
                elif pmode==2:
                    gui.task.mainWindowFront.connect(gui3.bringMeFront)
                    gui2.hide()
                    gui3.showFullScreen()
                else:
                    restart=False
    if isExit:
        cv2.destroyAllWindows()
        QCoreApplication.quit()
    threadStarted=False

def realtime2(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui,container,gui2,pmode,realtimeView,screen_width,screen_height,resultGUI,gui3,mainTask,mainMenu
    app=QApplication(sys.argv)
    container=MainContainer()
    mainTask=RunTask2(w,detectionLoop)
    gui=GameGUI(container)
    gui.load_IndexMap('indexMapd')
    gui.setTask(w, realtime)
    #gui.show()
    
    gui2=Game2GUI()
    gui2.setSharedTask()
    resultGUI=ResultContainer()
    #resultGUI.setStyleSheet('background-image: url(gamescene/WASHING-HANDS.jpg) no-repeat center center fixed;')
    resultGUI.poseList.pose_time=gui.pose_time
    resultGUI.poseList.prepare('indexMap')
    
    gui3=Game3GUI()
    gui3.reinit()
    gui3.wireUpEvent()
    
    mainMenu=MainMenuUI()
    
    
    if pmode==1:
        resultGUI.showFullScreen()
        gui2.showFullScreen()
        realtimeView.setGeometry(QtCore.QRect(0,0,screen_width,screen_height))
        gui.task.mainWindowFront.connect(gui2.bringMeFront)
    elif pmode==0:
        container.showFullScreen()
        gui.task.mainWindowFront.connect(container.bringMeFront)
    elif pmode==2:
        resultGUI.showFullScreen()
        gui2.showFullScreen()
        realtimeView.setGeometry(QtCore.QRect(0,0,screen_width,screen_height))
        gui3.showFullScreen()
        gui3.leftPanel.update()
        gui.task.mainWindowFront.connect(gui3.bringMeFront)
    elif pmode==-1:
        mainMenu.showFullScreen()
        mainMenu.activateWindow()
        
    mainTask.start()
    gui.runTask()
    app.exec_()
    
    

caffe.set_mode_gpu()
model_path=''
prototxt_path=''
if len(sys.argv)>2:
    model_path=sys.argv[1]
    prototxt_path=sys.argv[2]
    load_model(model_path, prototxt_path)
    if len(sys.argv)>3:
        pmode=int(sys.argv[3])

#if len(sys.argv)>3:
#    bufferSize=int(sys.argv[3])
#    for i in range(bufferSize):
#        bufferImg[i]=[]
    
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
