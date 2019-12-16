from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import model_from_json
import json
import copy
from PIL import Image
import matplotlib.pyplot as plt

model_file_name='weights'
with open('./save/'+'model.json','r') as f:
    model_json = json.load(f)
model = model_from_json(model_json)
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
model.load_weights('./save/' + model_file_name +'.h5')

a=0



dxy = ((1,0),(0,1),(-1,0),(0,-1))
rensa=0
rensa_max=0
same_count=0
#消す関数
def erace():
    flag = True
    for y in range(0,12):
        for x in range(0,6):
            if field[x+y*6]==0:
                continue
            same = same_puyo((x,y),[])
            if len(same) < 4:
                continue
            for p in same:
                field[p[0]+p[1]*6]=0
                flag = True
            flag = False
    return flag
def same_puyo(cur,same):
    same.append(cur)
    for di in dxy:
        next = (cur[0]+di[0],cur[1]+di[1],)
        if 0<=next[0]<6 and 0 <= next[1]<12 \
            and next not in same \
            and field[next[0]+next[1]*6] == field[cur[0]+cur[1]*6]:
            same_puyo(next, same)
    return same

def drop():
    for x in range(0,6):
        for y in range(0,12)[::-1]:
            if field[x+y*6]==0:
                for y2 in range(y+1)[::-1]:
                    if field[x+y2*6]!=0:
                        field[x+y*6]=field[x+y2*6]
                        field[x+y2*6]=0
                        break

def solve():
    global rensa
    global rensa_max
    while 1:
        drop()
        show()
        if erace():
            break
        show()
        rensa_max=max(rensa_max,rensa)



#表示する関数
#表示
def show():
    global a
    #2段はツモ表示に使用する
    img = np.zeros((14,6,3))
    for i in range(0,12):
        for j in range(0,6):
            for k in range(0,5):
                if field[i*6+j]!=0:
                    #なにもなし
                    if field[i*6+j]==1 and k == 0:
                        #print('toumei')
                        img[i+2,j,0]=0
                        img[i+2,j,1]=0
                        img[i+2,j,2]=0
                    #赤
                    elif field[i*6+j]==1 and k == 1:
                        #print('aka')
                        img[i+2,j,0]=1
                        img[i+2,j,1]=0
                        img[i+2,j,2]=0
                    #紫                        
                    elif field[i*6+j]==2 and k == 2:
                        #print('murasaki')                      
                        img[i+2,j,0]=1
                        img[i+2,j,1]=0
                        img[i+2,j,2]=1
                    #黄色
                    elif field[i*6+j]==3 and k == 3:
                        #print('kiiro') 
                        img[i+2,j,0]=0
                        img[i+2,j,1]=1
                        img[i+2,j,2]=1
                    #緑
                    elif field[i*6+j]==4 and k == 4:
                        #print('midori') 
                        img[i+2,j,0]=0
                        img[i+2,j,1]=1
                        img[i+2,j,2]=0
    #ツモ1
    if field[len(field)-6]==1:
        img[0,1,0]=1
        img[0,1,1]=0
        img[0,1,2]=0
    elif field[len(field)-6]==2:
        img[0,1,0]=1
        img[0,1,1]=0
        img[0,1,2]=1
    elif field[len(field)-6]==3:
        img[0,1,0]=0
        img[0,1,1]=1
        img[0,1,2]=1
    elif field[len(field)-6]==4:
        img[0,1,0]=0
        img[0,1,1]=1
        img[0,1,2]=0
    #ツモ2
    if field[len(field)-5]==1:
        img[1,1,0]=1
        img[1,1,1]=0
        img[1,1,2]=0
    elif field[len(field)-5]==2:
        img[1,1,0]=1
        img[1,1,1]=0
        img[1,1,2]=1
    elif field[len(field)-5]==3:
        img[1,1,0]=0
        img[1,1,1]=1
        img[1,1,2]=1
    elif field[len(field)-5]==4:
        img[1,1,0]=0
        img[1,1,1]=1
        img[1,1,2]=0
    #ツモ3
    if field[len(field)-4]==1:
        img[0,3,0]=1
        img[0,3,1]=0
        img[0,3,2]=0
    elif field[len(field)-4]==2:
        img[0,3,0]=1
        img[0,3,1]=0
        img[0,3,2]=1
    elif field[len(field)-4]==3:
        img[0,3,0]=0
        img[0,3,1]=1
        img[0,3,2]=1
    elif field[len(field)-4]==4:
        img[0,3,0]=0
        img[0,3,1]=1
        img[0,3,2]=0
    if field[len(field)-3]==1:
        img[1,3,0]=1
        img[1,3,1]=0
        img[1,3,2]=0
    elif field[len(field)-3]==2:
        img[1,3,0]=1
        img[1,3,1]=0
        img[1,3,2]=1
    elif field[len(field)-3]==3:
        img[1,3,0]=0
        img[1,3,1]=1
        img[1,3,2]=1
    elif field[len(field)-3]==4:
        img[1,3,0]=0
        img[1,3,1]=1
        img[1,3,2]=0
    if field[len(field)-2]==1:
        img[0,5,0]=1
        img[0,5,1]=0
        img[0,5,2]=0
    elif field[len(field)-2]==2:
        img[0,5,0]=1
        img[0,5,1]=0
        img[0,5,2]=1
    elif field[len(field)-2]==3:
        img[0,5,0]=0
        img[0,5,1]=1
        img[0,5,2]=1
    elif field[len(field)-2]==4:
        img[0,5,0]=0
        img[0,5,1]=1
        img[0,5,2]=0
    if field[len(field)-1]==1:
        img[1,5,0]=1
        img[1,5,1]=0
        img[1,5,2]=0
    elif field[len(field)-1]==2:
        img[1,5,0]=1
        img[1,5,1]=0
        img[1,5,2]=1
    elif field[len(field)-1]==3:
        img[1,5,0]=0
        img[1,5,1]=1
        img[1,5,2]=1
    elif field[len(field)-1]==4:
        img[1,5,0]=0
        img[1,5,1]=1
        img[1,5,2]=0
                        
    #print (img)
    plt.imshow(img)
    #plt.savefig("pred100/"+pred_label+"/"+str(a)+".png")
    #a+=1
    plt.pause(0.3)
    
#おいた回数
num_set = 0
#プレイ画面
field = np.zeros((12*6+6))
#色をパターン化した盤面
val_field = np.zeros((12*6+6))
#val_fieldをmodelに対応した形に
data = np.zeros((12,6,39))

#盤面をresetする
def reset():
    global field
    global num_set
    field = np.zeros(12*6+6)
    num_set = 0
    #4色うちからのツモを生成
    field[len(field)-1]=int(np.random.randint(1,5))
    field[len(field)-2]=int(np.random.randint(1,5))
    flag = False
    while(flag == False):

        field[len(field)-3]=int(np.random.randint(1,5))
        field[len(field)-4]=int(np.random.randint(1,5))
        field[len(field)-5]=int(np.random.randint(1,5))
        field[len(field)-6]=int(np.random.randint(1,5))
        if field[len(field)-3] == field[len(field)-4] or field[len(field)-4] == field[len(field)-5] or field[len(field)-5] == field[len(field)-6]:
            flag = True

#n列目（0～5）にn色（１～４）を設置する関数
def set(retu,color):
    i = 0
    for i in range(0,12):
        if field[(11-i)*6+retu]==0:
            field[(11-i)*6+retu]=color
            break    

#ぷよの連結数
def len_puyo(x, y, c,npy):
    global same_count
    npy[x+y*6]=0
    #print("x:"+str(x)+"y:"+str(y))
    same_count+=1
    if(x>0 and npy[(x-1)+y*6]==c):
        #print("hidari")
        len_puyo(x-1,y,c,npy)
    if(x<5 and npy[(x+1)+y*6]==c):
        #print("migi")
        len_puyo(x+1,y,c,npy)
    if(y>0 and npy[x+(y-1)*6]==c):
        #print("ue")
        len_puyo(x,y-1,c,npy)
    if(y<11 and npy[x+(y+1)*6]==c):
        #print("sita")
        len_puyo(x,y+1,c,npy)
    return same_count 
        
    
reset()
while(True):
    print("--------------------------------------------------")
    #盤面の一番上の段にぷよがあったらreset()
    for i in range(0,6):
        if field[i] != 0:
            reset()
    #10手でリセット
    if num_set == 10:
        reset()
    
    val_R=0
    val_P=0
    val_Y=0
    val_G=0
    
  
    #色をパターン化した盤面
    val_field = np.zeros((12*6+6))
    #val_fieldをmodelに対応した形に
    data = np.zeros((12,6,39))
    
    #盤面のパターン割り振り
    color = 1
    for i in range(0,12)[::-1]:
        for j in range(0,6):
            if field[i*6+j] == 1:
                if val_R ==0:
                    val_R =color
                    color+=1
            elif field[i*6+j] == 2:
                if val_P ==0:
                    val_P =color
                    color+=1
            elif field[i*6+j] == 3:
                if val_Y ==0:
                    val_Y =color
                    color+=1
            elif field[i*6+j] == 4:
                if val_G ==0:
                    val_G =color
                    color+=1
    #フィールドで４色ない場合ツモでパターン割り振り
    for i in range(0,6):
            if field[12*6+i] == 1:
                if val_R ==0:
                    val_R =color
                    color+=1
            elif field[12*6+i] == 2:
                if val_P ==0:
                    val_P =color
                    color+=1
            elif field[12*6+i] == 3:
                if val_Y ==0:
                    val_Y =color
                    color+=1
            elif field[12*6+i] == 4:
                if val_G ==0:
                    val_G =color
                    color+=1
        
    #val_fieldを対応色に入れ替え
    for i in range(0,12*6+6):
       if field[i] == 1:
           val_field[i] = val_R
       elif field[i] == 2:
           val_field[i] = val_P
       elif field[i] == 3:
           val_field[i] = val_Y
       elif field[i] == 4:
           val_field[i] = val_G
    for j in range (0,12):
                for i in range(0,6):
                    for k in range (0,4):
                        if val_field[len(field)-6]==k+1:
                            data[j,i,k+4]=1
                        if val_field[len(field)-5]==k+1:
                            data[j,i,k+8]=1
                        if val_field[len(field)-4]==k+1:
                            data[j,i,k+12]=1
                        if val_field[len(field)-3]==k+1:
                            data[j,i,k+16]=1
                        if val_field[len(field)-2]==k+1:
                            data[j,i,k+20]=1
                        if val_field[len(field)-1]==k+1:
                            data[j,i,k+24]=1
                        if val_field[j*6+i]==k+1:
                            data[j,i,k]=1
    #for j in range (0,12):
            #for i in range(0,6):
                #if val_field[j*6+i]==0:
                        #data[j,i,28]=1
    
    #for j in range (0,12):
            #for i in range(0,6):
                #print(data[j,i,28])
    for i in range (0,6):
                flag0=False
                for j in range(1,12):
                        if val_field[j*6+i] != 0 and flag0 == False:
                                data[j-1,i,28]=1
                                flag0=True

    for i in range(0,6):
            for j in range(0,12):
                val_field2= copy.deepcopy(val_field)
                same_count=0
                #print(npy[j*6+i])
                if(val_field[j*6+i]!=0 and val_field[j*6+i]!=5):
                    num_same=len_puyo(i,j,val_field[j*6+i],val_field2)
                    if(num_same)==2:
                        data[j,i,int(28+val_field[j*6+i])]=1
                    if(num_same)==3:
                        data[j,i,int(32+val_field[j*6+i])]=1
                            
    #print (field)
    #print("val_field")
    #print (val_field)
    #print(data)
    #37全部1
    #38全部0
    for i in range(0,6):
        for j in range(0,12):
            data[j,i,37]=1
    
    print(data[:,:,28])
    
    
    data = data[None, ...]
    print(data.shape)
    label=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
    pred = model.predict(data, batch_size=1, verbose=0)
    score = np.max(pred)
    #print("pred")
    #print(pred)

    #１，モデルどおりの行動
    pred_label = label[np.argmax(pred[0])]
    #２，ランダムな行動
    #pred_label = label[int(np.random.randint(0,22))]
    print('action:',pred_label)
    #print('score:',score)

    if  max(pred[0]) == 1.:
        print (val_field)
        print(pred)
        #show()

    t1 = field[len(field)-6]
    t2 = field[len(field)-5]
    t3 = field[len(field)-4]
    t4 = field[len(field)-3]
    t5 = field[len(field)-2]
    t6 = field[len(field)-1]

    #actionによってぷよを置きツモを更新
    if pred_label =='1':
        set(0,t2)
        set(0,t1)
    elif pred_label == '2':
        set(1,t2)
        set(1,t1)
    elif pred_label == '3':
        set(2,t2)
        set(2,t1)
    elif pred_label == '4':
        set(3,t2)
        set(3,t1)
    elif pred_label == '5':
        set(4,t2)
        set(4,t1)
    elif pred_label == '6':
        set(5,t2)
        set(5,t1)
    elif pred_label == '7':
        set(0,t1)
        set(0,t2)
    elif pred_label == '8':
        set(1,t1)
        set(1,t2)
    elif pred_label == '9':
        set(2,t1)
        set(2,t2)
    elif pred_label == '10':
        set(3,t1)
        set(3,t2)
    elif pred_label == '11':
        set(4,t1)
        set(4,t2)
    elif pred_label == '12':
        set(5,t1)
        set(5,t2)
    elif pred_label == '13':
        set(0,t1)
        set(1,t2)
    elif pred_label == '14':
        set(1,t1)
        set(2,t2)
    elif pred_label == '15':
        set(2,t1)
        set(3,t2)
    elif pred_label == '16':
        set(3,t1)
        set(4,t2)
    elif pred_label == '17':
        set(4,t1)
        set(5,t2)
    elif pred_label == '18':
        set(0,t2)
        set(1,t1)
    elif pred_label == '19':
        set(1,t2)
        set(2,t1)
    elif pred_label == '20':
        set(2,t2)
        set(3,t1)
    elif pred_label == '21':
        set(3,t2)
        set(4,t1)
    elif pred_label == '22':
        set(4,t2)
        set(5,t1)

    #ツモを繰り上げ、新しいツモを生成
    field[len(field)-6]=field[len(field)-4]
    field[len(field)-5]=field[len(field)-3]
    field[len(field)-4]=field[len(field)-2]
    field[len(field)-3]=field[len(field)-1]
    field[len(field)-2]=int(np.random.randint(1,5))
    field[len(field)-1]=int(np.random.randint(1,5))

    #４つ以上繋がったぷよを消す、浮いているぷよを落とすを繰り返す
    solve()
    #print(field)
    num_set+=1
    
    #show()
    plt.clf()
    