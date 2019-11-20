function cpsb3=main()
  [filename, pathname] = uigetfile( ...
       {'*.png;*.jpg', 'All MATLAB Files (*.png, *.jpg)'; ...
        '*.*', 'All Files (*.*)'},  'Pick a file'); 
I2=imread(filename);%ͼ��Ĳɼ���ת��
I = imcrop(I2,[19,0,800,320])
figure(1),subplot(2,2,1),imshow(I);title('ԭͼ');

I1=rgb2gray(I);
subplot(2,2,2),imshow(I1);title('�Ҷ�ͼ');
subplot(2,2,3),imhist(I1);title('�Ҷ�ͼֱ��ͼ');
I2=edge(I1,'canny',0.01,'both');%��Ե���� ƽ������ ��С���� ��ȡ��Ե
subplot(2,2,4),imshow(I2);title('canny���ӱ�Ե���')

 se=[1;1;1];%���ͽṹԪ��
 I3=imerode(I2,se);

figure(2),subplot(2,2,1),imshow(I3);title('��ʴ��ͼ��');
se=strel('rectangle',[60,60]);%����һ������ ���ͽṹԪ�� 
I4=imclose(I3,se);  %������
subplot(2,2,2),imshow(I4);title('ƽ��ͼ�������');
I5=bwareaopen(I4,2000,8);

subplot(2,2,3),imshow(I5);title('��ͼ�����Ƴ�С����');
subplot(2,2,4),imhist(I5);title('�Ҷ�ͼֱ��ͼ');


[y,x]=size(I5);%���� �� Ĭ��1 ��
myI=double(I5);

white_y=zeros(y,1);
for i=1:y
    for j=1:x
        if(myI(i,j,1)==1)
           white_y(i,1)= white_y(i,1)+1;%��ɫ���ص�ͳ��
        end
    end
end
[temp MaxY]=max(white_y);%tempΪ����white_y��Ԫ���е����ֵ��MaxYΪ��ֵ�������� �������е�λ�ã�
PY1=MaxY;
while ((white_y(PY1,1)>=120)&&(PY1>1))
    PY1=PY1-1; %�ҳ������϶�
end
PY2=MaxY;
while ((white_y(PY2,1)>=40)&&(PY2<y))
    PY2=PY2+1;  %�ҳ������¶�
end
%begin����ɨ��
white_x=zeros(1,x);%��һ��ȷ��x����ĳ�������
for j=1:x
    for i=PY1:PY2
        if(myI(i,j,1)==1)
            white_x(1,j)= white_x(1,j)+1;
        end
    end
end


PX1=1;  % ����Ϊ�ҳ���X������Сֵ������
while ((white_x(1,PX1)<3)&&(PX1<x))
    PX1=PX1+1;
end
PX2=x;  % ����Ϊ�ҳ���X�������ֵ����
while ((white_x(1,PX2)<3)&&(PX2>PX1))
    PX2=PX2-1;
end
PX2=PX2+2;
dw1=I(PY1:PY2,PX1:PX2,:); % 
figure(3),subplot(2,2,1),imshow(dw1),title('��λ���к�Ĳ�ɫ����ͼ��')
se2=strel('disk',2)
dw=imerode(dw1,se2)

imwrite(dw,'dw.jpg');   %dw
a=imread('dw.jpg');  %��ȡͼƬ�ļ��е�����
b=rgb2gray(a);  %��ɫͼ��ת��Ϊ�Ҷ�ͼ��
figure(3);subplot(2,2,2),imshow(b),title('1.���ƻҶ�ͼ��')

%�Գ��ƶ�ֵ��  �趨һ����ֵT����T��ͼ������ݷֳ������֣�����T������Ⱥ��С��T������Ⱥ������ͼ���ֵ��
g_max=double(max(max(b)));%����˫������ֵ
g_min=double(min(min(b)));%����˫������ֵ
T=round(g_max-(g_max-g_min)/3); % T Ϊ��ֵ������ֵ  �����ֵ��ֵ�����Ҷȵ����ֵ��ȥ ���ֵ������ֵ��1/3�ݶ�
[m,n]=size(b);  %���ؾ���b�ĳߴ���Ϣ mΪ�� nΪ��
d=(double(b)>=T);  % d:��ֵͼ��
figure(3);subplot(2,2,3),imshow(d),title('2.���ƶ�ֵͼ��')
% % �˲�

h=fspecial('average',10);  %��ֵ�˲��� averageָ�����ӵ����ͣ�3Ϊ��Ӧ�Ĳ���
d=im2bw(round(filter2(h,d)));  %BW = im2bw(I, level) ���Ҷ�ͼ�� I ת��Ϊ������ͼ�����ͼ�� %BW ������ͼ��������ֵ���� level �������滻Ϊֵ1 (��ɫ)�������滻Ϊֵ0(��ɫ
imwrite(d,'4.��ֵ�˲���.jpg');  % ����
figure(3),subplot(2,2,4),imshow(d),title('4.��ֵ�˲���')


% ĳЩͼ����в��������ͻ�ʴ
se=eye(3); % eye(n) ����n��n��һ���� ��λ����
[m,n]=size(d);
temp=bwarea(d);%matlab�ﺯ��bwarea ����Ŀ��������
if bwarea(d)/m/n>=0.365 %�����ֵͼ���а׵Ķ���ռ�����
    d=imerode(d,se);   %imerode ʵ��ͼ��ʴ dΪ������ͼ��se�ǽṹԪ�ض���
elseif bwarea(d)/m/n<=0.235
    d=imdilate(d,se); %imdilate ͼ������
end
imwrite(d,'5.���ͻ�ʴ�����.jpg');
figure(4),subplot(2,2,1),imshow(d),title('5.���ͻ�ʴ�����')

d=qiege(d);


I1 = padarray(d,[20,20],1,'both');
figure;imshow(I1)
%��ͨ����ֿ�
CC = bwconncomp(I1);
%�г�ÿ������������С������
numPixels = cellfun(@numel,CC.PixelIdxList);
I2 = I1;
%���
I2(CC.PixelIdxList{8}) = 0;
% [biggest,idx] = min(numPixels);
% I2(CC.PixelIdxList{idx}) = 0;
figure;imshow(I2)








figure(4),subplot(2,2,2),imshow(d),title('6.�и��ͼƬ')
[word1,d]=getword(d);% �ָ���ڶ����ַ�
[word2,d]=getword(d);% �ָ���ڶ����ַ�
[word3,d]=getword(d);% �ָ���������ַ�
[word4,d]=getword(d);% �ָ�����ĸ��ַ�
[word5,d]=getword(d);% �ָ��������ַ�
[word6,d]=getword(d);% �ָ���������ַ�
[word7,d]=getword(d);% �ָ�����߸��ַ�

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
    [m,n]=size(d); % ������
    wide=0;
    while sum(d(:,wide+1))~=0 && wide<=n-2  %�а�ɫ��1֪��û�а�ɫ��Ҳ�����ҳ�һ����ɫ����
        wide=wide+1;
    end
    temp=qiege(imcrop(d,[0 0 wide m])); %�г���һ���ַ�
    figure;imshow(d)
    [m1,n1]=size(temp);
    if wide<y1 && n1/m1>y2  %����С��  �� �и������С �г�����
        d(:,[1:wide])=0;  % ��һ��Ϳ��
        if sum(sum(d))~=0
            d=qiege(d);  % �и����С��Χ
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
%�и�
    function e=qiege(d)
        [m,n]=size(d);
        top=1;bottom=m;left=1;right=n;   % init
        while sum(d(top,:))==0 && top<=m     %�и����ɫ���򣨺��У�
            top=top+1;
        end
        while sum(d(bottom,:))==0 && bottom>1   %ͬ��
            bottom=bottom-1;
        end
        while sum(d(:,left))==0 && left<n        %�и�����������У�
            left=left+1;
        end
        while sum(d(:,right))==0 && right>=1
            right=right-1;
        end
        dd=right-left+1;
        hh=bottom-top+1;
        e=imcrop(d,[left top dd hh]);
    end