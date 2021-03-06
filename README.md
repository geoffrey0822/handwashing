# Hand Washing Program

## 1. Descriptions
The hand-washing model is based on pretrained GoogLeNet's model and finetune with our dataset. Moreover, the input data is greyscaled image instead of RGB colored image.
___
## 2. Step to execute the program
1. Clone this repository
2. Download pretrained model [https://drive.google.com/open?id=1tZ9zOvFWg0WHIYgoZiS9RFRGMoTyw794] and copy to current directory
3. Execute run_game3_mode_tools.bat or "python handwash_game_v2.py googlenet/snapshot_iter_750.caffemodel googlenet/deploy.prototxt -1"

*Supported Platform: Windows, Linux*

*Supported Python Version: 2.7, 3.5*

*How to enable/disable reforcement: Uncomment/comment line #1834 in handwash_game_v2.py*
*How to enable capturing sample in background: add corresponded codes to line #1612 as shown as below*

```python
tsstr=str(int(time.time()))
file_path=os.path.join('cached','%s/%s.jpg'%(class_sidx,tsstr))
```

___
## 4. Publications
___