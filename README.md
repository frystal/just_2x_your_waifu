# just_2x_your_waifu
能实现较为完美的将anime插画放大两倍  
这玩意只是我人工智障导论的大作业，现在也只是勉强能用，等以后会继续修补一下  
<p align="center">
原始图片
</p>  
<p align="center">
<img src="https://raw.githubusercontent.com/frystal/just_2x_your_waifu/master/demo/original.png"/><br/>
</p>  
<p align="center">
放大图片
</p>  
<p align="center">
<img src="https://raw.githubusercontent.com/frystal/just_2x_your_waifu/master/demo/2x.png"/><br/>
</p>  


图片已获授权 
画师：墨鱼 p站(https://www.pixiv.net/member.php?id=17432287)  

## Usage
需求tensorflow2.1.0  
需要放大的图片放置于src目录下的test_image，并命名为original.png
```
cd ./src
pip3 -r install requirements.txt
python3 start.py
```
## How to train
将尽量高清的图片放置于./data/original目录下，要求为png  
```
cd ./src
python3 generate.py
python3 train.py
```
保存的模型参数在./checkpoints目录下  
模型导出在./model目录下

## Live demo
增加了简易的web端，仅供娱乐用途  
需求flask  
```
cd ./webserver
python3 app.py
```
端口为5000
## 即将而来的更新