import numpy as np
import os,sys
import caffe
import cv2
#import pyttsx
import time
import Queue
from PySide.QtGui import QApplication, QPixmap, QMovie
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import imageio
from PySide.QtCore import QByteArray

gray_color_table = [qRgb(i, i, i) for i in range(256)]

screen_width=1920
screen_height=1080

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
        self.picture=QtGui.QPixmap(path).copy(roi)
        self.picture=self.picture.scaled(256,256)
        self.label='Hello'
        self.lightOn=False
        self.sWidth=256
        self.sHeight=256
        print 'ok'
        
    def setImage(self,cvimg):
        
        roi=QRect(0,0,256,256)
        tmp=cvimg[:,:,(2,1,0)]
        tmp=np.require(tmp, np.uint8, 'C')
        qimg=toQImage(tmp,False)
        self.picture=QtGui.QPixmap(qimg).copy(roi)
        
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
        painter.setFont(QFont("Comic Sans MS",20))
        painter.drawText(QPoint(screen_width/2,screen_height-50),self.label)

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
rawframe=None
resultGUI=None

# for global timer
global_start=0
global_end=0
global_duration=10 # in second
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

class RunTask(QtCore.QThread):
    
    pose_updated= QtCore.pyqtSignal(object)
    frame_updated= QtCore.pyqtSignal(object)
    result_pop= QtCore.pyqtSignal(object)
    mainWindowFront=QtCore.pyqtSignal(object)
    
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
        
    def updateResult(self):
        self.result_pop.emit('')
        
    def mainWindowPop(self):
        self.mainWindowFront.emit('')

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
        
    def runTask(self):
        self.task.start()
    
    
    def updatePoseFunc(self):
        global rawroi
        p=self.updatePose
        self.updatePose=-1
        if p>-1:
            #self.cellWidget(p,2)
            #self.setCellWidget(1,3,CVImageWidget(rawroi,self))
            self.cellWidget(self.placement[str(p+1)][0],self.placement[str(p+1)][1]).lightOn=True
            print 'frame update'
            #self.cellWidget(p,2).setStyleSheet('QLabel {background-color:green}')
            #self.cellWidget(p,2).setText('Done')
        for cls in all_pose_startend_time.keys():
            t_min=int(np.floor(self.pose_time[cls]/60))
            t_sec=self.pose_time[cls]-t_min*60
            self.cellWidget(self.placement[cls][0],self.placement[cls][1]).label='%02d:%02d'%(t_min,t_sec)
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
            self.img_wdg[class_id].label='%02d:%02d'%(t_min,t_sec)
            all_pose_startend_time[class_id]=[init_time,init_time]
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
            all_pose_startend_time[class_id]=[init_time,init_time]
            
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
                print fname
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
                        #pimg=cv2.rectangle(img,(0,0),(256,256),(0,255,0),3)
                        gif_output.append_data(img[:,:,(2,1,0)])
                    gif_output.close()
                    print 'load gif'
                    lblMov=QtGui.QLabel('',self)
                    gif_anim=QMovie('tmp/%s.gif'%class_id,parent=self)
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setSpeed(200)
                    lblMov.setMovie(gif_anim)
                    gif_anim.start()
                    self.setCellWidget(r,c*3+1,lblMov)
                    self.cellWidget(r,c*3).setStyleSheet('background-color: green;')
                    self.my_rank+=1    
                    #self.cellWidget(r,c*3+1).setImage(self.correctFrames[class_id][0])
                    #self.cellWidget(r,c*3+1).update()
                else:
                    lblMov=QtGui.QLabel('',self)
                    gif_anim=QMovie('examples/%s.gif'%class_id,parent=self)
                    gif_anim.setCacheMode(QMovie.CacheAll)
                    gif_anim.setSpeed(300)
                    gif_anim.setScaledSize(QtCore.QSize(150,150))
                    lblMov.setMovie(gif_anim)
                    gif_anim.start()
                    self.setCellWidget(r,c*3+1,lblMov)
                print '%02d:%02d'%(t_min,t_sec)
        
            
class ResultContainer(QtGui.QWidget):
    def __init__(self,parent=None):
        global screen_height,screen_width
        super(ResultContainer,self).__init__(parent)
        self.setQAction=QtGui.QAction('quit',self,shortcut=Qt.Key_Q, triggered=self.setQ)
        self.addAction(self.setQAction)
        self.rank=0 # 0-F, 1-C, 2-B, 3-A
        self.ranks=['F','C','B','A']
        self.lblRank=QtGui.QLabel(self)
        self.lblRank.setGeometry(0,0,screen_width,100)
        self.poseList=ResultWidget(self)
        self.poseList.setGeometry(0,100,640,480)
        newfont = QtGui.QFont("Comic Sans MS",48, QtGui.QFont.Bold)
        self.lblRank.setText('Rank:%s'%self.ranks[self.rank])
        self.lblRank.setFont(newfont)
        self.lblRank.setStyleSheet('color: orange')
        self.lblRank.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        
    def setQ(self):
        print 'Quit'
        QCoreApplication.quit()
        
    def reset(self):
        self.rank=0
        self.lblRank.setText('Rank:%s'%self.ranks[self.rank])
        
    def updateRank(self):
        self.rank=int(float(self.poseList.my_rank)*len(self.ranks))/7
        self.lblRank.setText('Rank:%s'%self.ranks[self.rank])

class Game2GUI(QtGui.QWidget):
    def __init__(self,parent=None):
        super(Game2GUI,self).__init__(parent)
        global realtimeView
        self.resultView=ResultWidget(self)
        realtimeView=CVImageRTWidget(self)
    
    def frameUpdate(self):
        global rawroi,realtimeView,rawframe,global_start,global_end,global_count
        diff=global_count-global_start
        t_min=int(np.floor(diff/60))
        t_sec=diff-t_min*60
        realtimeView.label='%02d:%02d'%(t_min,t_sec)
        realtimeView.setImage(rawframe)
        realtimeView.update()
        
    def resultUpdate(self):
        global resultGUI
        resultGUI.poseList.update()
        resultGUI.updateRank()
        
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
        
    def bringMeFront(self):
        self.activateWindow()
    
    def setQ(self):
        print 'Quit'
        QCoreApplication.quit()
    
def realtime(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui,game_thread,rawroi,rawframe,previousPredict,pmode,global_start,global_end,global_count,resultGUI,tex,tey
    game_scene_img=cv2.imread(game_scene)
    cam=cv2.VideoCapture(1)
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
    #engine = pyttsx.init()
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
    while True:
        ret,img=cam.read()
        frame=img.copy()
        rawimg=img.copy()
        
        if pmode==1:
            
            global_count=time.time()
            if global_start==0:
                global_start=global_count
                global_end=global_start+global_duration
            elif global_count>=global_end:
                print 'time\'s up'
                break
            
        cv2.putText(img,modes[mode],(10,25),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
        frame=preprocess(frame,mode)
        cv2.rectangle(img,(ix,iy),(tex,tey),(0,255,0),3)
        cv2.imshow('real time classifier',img)
        scene=game_scene_img.copy()
        pbbox=game_pose_roi[currentPose]
        sroi_color=(0,0,255)
        if pmode==1:
            rawframe=img.copy()
            game_thread.updateRealtime()
        if compute_area((ix,iy), (ex,ey))>1:
            if behide:
                game_thread.mainWindowPop()
                behide=False
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
                        #print class_scs
                        max_idx=np.argmax(class_scs)
                        class_idx=max_idx+1
                        
                        #print len(expected_pose)
                        print max_idx
                        if max_idx>-1 and max_idx<8:
                            class_sidx=str(class_idx)
                            if class_sidx in resultGUI.poseList.correctFrames.keys():
                                if len(resultGUI.poseList.correctFrames[class_sidx])<maxTmpFrame:
                                    resultGUI.poseList.correctFrames[class_sidx].append(rawroi.copy())
                                    print 'catched frame'
                        if class_idx>-1 and class_idx<8 and class_idx==previousPredict and previousPredict>-1:
                            all_pose_startend_time[str(class_idx)][1]=time.time()
                            gui.pose_time[str(class_idx)]+=round(all_pose_startend_time[str(class_idx)][1]-all_pose_startend_time[str(class_idx)][0])
                            if pmode==1:
                                resultGUI.poseList.pose_time[str(class_idx)]+=round(all_pose_startend_time[str(class_idx)][1]-all_pose_startend_time[str(class_idx)][0])
                            all_pose_startend_time[str(class_idx)][0]=all_pose_startend_time[str(class_idx)][1]
                            print gui.pose_time
                        elif class_idx>-1 and class_idx<8 and class_idx!=previousPredict:
                            all_pose_startend_time[str(class_idx)][0]=time.time()
                            all_pose_startend_time[str(class_idx)][1]=all_pose_startend_time[str(class_idx)][0]
                        
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
                        
        cv2.rectangle(scene,(box_startAt[1]+pbbox[1],box_startAt[0]+pbbox[0]),(box_startAt[1]+pbbox[3],box_startAt[0]+pbbox[2]),sroi_color,3)
        key=cv2.waitKey(1)
        if key==27:
            isExit=True
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
            else:
                pmode=0
                gui.showFullScreen()
                gui2.hide()
    
    cv2.destroyAllWindows()
    if pmode==1:
        gui2.hide()
        game_thread.updateResult()
    if isExit:
        QCoreApplication.quit()

def realtime2(w=[]):
    global classifier,sub_classifier,ix,iy,ex,ey,frame,roi,lock,bufferImg,bufferSize,game_pose_roi,box_startAt,gui,container,gui2,pmode,realtimeView,screen_width,screen_height,resultGUI
    app=QApplication(sys.argv)
    container=MainContainer()
    gui=GameGUI(container)
    gui.load_IndexMap('indexMap')
    gui.setTask(w, realtime)
    #gui.show()
    
    gui2=Game2GUI()
    gui2.setSharedTask()
    resultGUI=ResultContainer()
    resultGUI.poseList.pose_time=gui.pose_time
    resultGUI.poseList.prepare('indexMap')
    if pmode==1:
        resultGUI.showFullScreen()
        gui2.showFullScreen()
        realtimeView.setGeometry(QtCore.QRect(0,0,screen_width,screen_height))
        gui.task.mainWindowFront.connect(gui2.bringMeFront)
    elif pmode==0:
        container.showFullScreen()
        gui.task.mainWindowFront.connect(container.bringMeFront)
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
