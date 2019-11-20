function cpsb3=main()
  [filename, pathname] = uigetfile( ...
       {'*.png;*.jpg', 'All MATLAB Files (*.png, *.jpg)'; ...
        '*.*', 'All Files (*.*)'},  'Pick a file'); 
I2=imread(filename);%图象的采集与转换
I = imcrop(I2,[19,0,800,320])
figure(1),subplot(2,2,1),imshow(I);title('原图');

I1=rgb2gray(I);
subplot(2,2,2),imshow(I1);title('灰度图');
subplot(2,2,3),imhist(I1);title('灰度图直方图');
I2=edge(I1,'canny',0.01,'both');%边缘算子 平滑处理 减小噪声 提取边缘
subplot(2,2,4),imshow(I2);title('canny算子边缘检测')

 se=[1;1;1];%线型结构元素
 I3=imerode(I2,se);

figure(2),subplot(2,2,1),imshow(I3);title('腐蚀后图像');
se=strel('rectangle',[60,60]);%生成一个矩阵 线型结构元素 
I4=imclose(I3,se);  %闭运算
subplot(2,2,2),imshow(I4);title('平滑图像的轮廓');
I5=bwareaopen(I4,2000,8);

subplot(2,2,3),imshow(I5);title('从图象中移除小对象');
subplot(2,2,4),imhist(I5);title('灰度图直方图');


[y,x]=size(I5);%【行 列 默认1 】
myI=double(I5);

white_y=zeros(y,1);
for i=1:y
    for j=1:x
        if(myI(i,j,1)==1)
           white_y(i,1)= white_y(i,1)+1;%蓝色像素点统计
        end
    end
end
[temp MaxY]=max(white_y);%temp为向量white_y的元素中的最大值，MaxY为该值的索引（ 在向量中的位置）
PY1=MaxY;
while ((white_y(PY1,1)>=120)&&(PY1>1))
    PY1=PY1-1; %找车牌最上端
end
PY2=MaxY;
while ((white_y(PY2,1)>=40)&&(PY2<y))
    PY2=PY2+1;  %找车牌最下端
end
%begin纵向扫描
white_x=zeros(1,x);%进一步确定x方向的车牌区域
for j=1:x
    for i=PY1:PY2
        if(myI(i,j,1)==1)
            white_x(1,j)= white_x(1,j)+1;
        end
    end
end


PX1=1;  % 以下为找车牌X方向最小值？？？
while ((white_x(1,PX1)<3)&&(PX1<x))
    PX1=PX1+1;
end
PX2=x;  % 以下为找车牌X方向最大值？？
while ((white_x(1,PX2)<3)&&(PX2>PX1))
    PX2=PX2-1;
end
PX2=PX2+2;
dw1=I(PY1:PY2,PX1:PX2,:); % 
figure(3),subplot(2,2,1),imshow(dw1),title('定位剪切后的彩色车牌图像')
se2=strel('disk',2)
dw=imerode(dw1,se2)

imwrite(dw,'dw.jpg');   %dw
a=imread('dw.jpg');  %读取图片文件中的数据
b=rgb2gray(a);  %彩色图像转换为灰度图像
figure(3);subplot(2,2,2),imshow(b),title('1.车牌灰度图像')

%对车牌二值化  设定一个阈值T，用T将图像的数据分成两部分：大于T的像素群和小于T的像素群，即对图像二值化
g_max=double(max(max(b)));%换成双精度数值
g_min=double(min(min(b)));%换成双精度数值
T=round(g_max-(g_max-g_min)/3); % T 为二值化的阈值  最佳阈值二值化，灰度的最大值减去 最大值与最下值的1/3梯度
[m,n]=size(b);  %返回矩阵b的尺寸信息 m为行 n为列
d=(double(b)>=T);  % d:二值图像
figure(3);subplot(2,2,3),imshow(d),title('2.车牌二值图像')
% % 滤波

h=fspecial('average',10);  %均值滤波器 average指定算子的类型，3为相应的参数
d=im2bw(round(filter2(h,d)));  %BW = im2bw(I, level) 将灰度图像 I 转换为二进制图像。输出图像 %BW 将输入图像中亮度值大于 level 的像素替换为值1 (白色)，其他替换为值0(黑色
imwrite(d,'4.均值滤波后.jpg');  % 保存
figure(3),subplot(2,2,4),imshow(d),title('4.均值滤波后')


% 某些图像进行操作，膨胀或腐蚀
se=eye(3); % eye(n) 返回n乘n单一矩阵 单位矩阵
[m,n]=size(d);
temp=bwarea(d);%matlab里函数bwarea 计算目标物的面积
if bwarea(d)/m/n>=0.365 %计算二值图像中白的对象占总面积
    d=imerode(d,se);   %imerode 实现图像腐蚀 d为待处理图像，se是结构元素对象
elseif bwarea(d)/m/n<=0.235
    d=imdilate(d,se); %imdilate 图像膨胀
end
imwrite(d,'5.膨胀或腐蚀处理后.jpg');
figure(4),subplot(2,2,1),imshow(d),title('5.膨胀或腐蚀处理后')

d=qiege(d);


I1 = padarray(d,[20,20],1,'both');
figure;imshow(I1)
%连通区域分开
CC = bwconncomp(I1);
%列出每个区域的面积大小并排序
numPixels = cellfun(@numel,CC.PixelIdxList);
I2 = I1;
%填充
I2(CC.PixelIdxList{8}) = 0;
% [biggest,idx] = min(numPixels);
% I2(CC.PixelIdxList{idx}) = 0;
figure;imshow(I2)








figure(4),subplot(2,2,2),imshow(d),title('6.切割后图片')
[word1,d]=getword(d);% 分割出第二个字符
[word2,d]=getword(d);% 分割出第二个字符
[word3,d]=getword(d);% 分割出第三个字符
[word4,d]=getword(d);% 分割出第四个字符
[word5,d]=getword(d);% 分割出第五个字符
[word6,d]=getword(d);% 分割出第六个字符
[word7,d]=getword(d);% 分割出第七个字符

figure(5),subplot(1,7,1),imshow(word1)
subplot(1,7,2),imshow(word2)
subplot(1,7,3),imshow(word3)
subplot(1,7,4),imshow(word4)
subplot(1,7,5),imshow(word5)
subplot(1,7,6),imshow(word6)
subplot(1,7,7),imshow(word7)
end

function [word,result]=getword(d)
word=[];
flag=0;
y1=8;
y2=0.5;
while flag==0
    [m,n]=size(d); % 求行列
    wide=0;
    while sum(d(:,wide+1))~=0 && wide<=n-2  %有白色加1知道没有白色，也就是找出一个白色区域
        wide=wide+1;
    end
    temp=qiege(imcrop(d,[0 0 wide m])); %切出第一个字符
    figure;imshow(d)
    [m1,n1]=size(temp);
    if wide<y1 && n1/m1>y2  %像素小于  或 切割字体大小 列除以行
        d(:,[1:wide])=0;  % 第一个涂黑
        if sum(sum(d))~=0
            d=qiege(d);  % 切割出最小范围
        else word=[];flag=1;
        end
    else
        word=qiege(imcrop(d,[0 0 wide m]));
        d(:,[1:wide])=0;
        if sum(sum(d))~=0;
            d=qiege(d);flag=1;
        else d=[];
        end
    end
end
result=d;
end
%切割
    function e=qiege(d)
        [m,n]=size(d);
        top=1;bottom=m;left=1;right=n;   % init
        while sum(d(top,:))==0 && top<=m     %切割出白色区域（横切）
            top=top+1;
        end
        while sum(d(bottom,:))==0 && bottom>1   %同上
            bottom=bottom-1;
        end
        while sum(d(:,left))==0 && left<n        %切割出白区域（纵切）
            left=left+1;
        end
        while sum(d(:,right))==0 && right>=1
            right=right-1;
        end
        dd=right-left+1;
        hh=bottom-top+1;
        e=imcrop(d,[left top dd hh]);
    end