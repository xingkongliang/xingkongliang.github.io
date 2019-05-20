---
title: Pillow Tutorial
date: 2019-04-11 15:56:55
description: Pillow 教程
categories:
tags: Python
---


# Pillow 教程

PIL(Python Image Library)是python的第三方图像处理库。其官方主页为:PIL。 PIL历史悠久，原来是只支持python2.x的版本的，后来出现了移植到python3的库pillow, pillow号称是friendly fork for PIL。

Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.

for Python2
[Python Imaging Library (PIL)](http://pythonware.com/products/pil/)

for Python3
[The Python Imaging Library Handbook](http://www.effbot.org/imagingbook/)

[python-pillow](https://python-pillow.org/)

[Pillow Github](https://github.com/python-pillow/Pillow)

[Pillow Docs](https://pillow.readthedocs.io/en/stable/)



```python
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
```

从一个文件加载图片，使用在`Image`模块下的`open()`函数。


```python
im = Image.open("lena.jpg")
```

## 1. 使用Image类

如果成功，这个函数会返回一个`Image`目标。你可以使用实例属性去测试这个文件的内容：


```python
from __future__ import print_function
print(im.format, im.size, im.mode)
```

    JPEG (512, 512) RGB


`format`属性判断图片的来源。如果一个图片不是从一个文件读入，它会被设置成None。`size`属性是一个2元组，包含宽和高。`mode`属性定义图片通道的数量和名字，以及像素类型和深度。通常的模式包含：“L”（luminance）表示灰度图像，”RGB“表示真彩色图像，”CMYK“表示印前（pre-press）图像

如果文件没有被打开，会给出`IOError`。

一旦你有一个`Image`类的实例，你可以使用这个类定义的方法取处理这个图片。例如，显示我们刚加载的图片：


```python
im.show()
```

使用matplotlib进行可视化


```python
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f1173b85518>




![png](output_12_1.png)


## 2. 读取和写入图像

## 3. 剪切、复制和合并图像

`Image`类包含允许您操作图像中的区域的方法。要从图像中提取矩形区域，请使用`crop()`方法。

### 从一个图片中复制一个矩形区域


```python
box = (100, 100, 400, 400)
region = im.crop(box)
print(region.size)
```

    (300, 300)


这个区域使用一个4元组定义，它的坐标是(left, upper, right, lower)。Python Imaging Library在左上角使用(0, 0)的坐标系统。坐标表示像素之间的位置，因此上边例子中的区域是300x300像素。

### 处理一个矩形区域，并且把它粘贴回原来位置


```python
region = region.transpose(Image.ROTATE_180)
im.paste(region, box)
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f11721e2518>




![png](output_20_1.png)


当把区域粘贴回去，区域的尺寸必须相同。除此之外，这个区域不能超越图像的边界。


```python
def roll(image, delta):
    """Roll an image sideways."""
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0: return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize-delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize-delta, ysize))

    return image
```

### 拆分和合并通道


```python
r, g, b = im.split()
im = Image.merge("RGB", (b, g, r))
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f11721c9b38>




![png](output_24_1.png)


### 保存图像


```python
im.save(r'out.jpg')
```

### 新建图像


```python
newIm= Image.new('RGB', (50, 50), 'red')
plt.imshow(newIm)
```




    <matplotlib.image.AxesImage at 0x7f1175c2d630>




![png](output_28_1.png)



```python
# 十六进制颜色
newIm = Image.new('RGBA',(100, 50), '#FF0000')
plt.imshow(newIm)
```




    <matplotlib.image.AxesImage at 0x7f1175bfd8d0>




![png](output_29_1.png)



```python
# 传入元组形式的RGBA值或者RGB值
# 在RGB模式下，第四个参数失效，默认255，在RGBA模式下，也可只传入前三个值，A值默认255
newIm = Image.new('RGB',(200, 100), (255, 255, 0, 120))
plt.imshow(newIm)
```




    <matplotlib.image.AxesImage at 0x7f1175bd1be0>




![png](output_30_1.png)


### 复制图片


```python
copyIm = im.copy()
copyIm.size
```




    (512, 512)



### 调整图片大小


```python
width, height = copyIm.size
resizedIm = im.resize((width, int(0.5* height)))
resizedIm.size
```




    (512, 256)



## 4. 几何变换


```python
im = Image.open("lena.jpg")
out = im.resize((128, 128))
out = im.rotate(45) # degrees counter-clockwise
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f11721370b8>




![png](output_36_1.png)


### 转置图像


```python
out = im.transpose(Image.FLIP_LEFT_RIGHT)
out = im.transpose(Image.FLIP_TOP_BOTTOM)
out = im.transpose(Image.ROTATE_90)
out = im.transpose(Image.ROTATE_180)
out = im.transpose(Image.ROTATE_270)
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f1172111080>




![png](output_38_1.png)


## 5. 颜色变换

### 在不同models之间转换


```python
from PIL import Image
im = Image.open("lena.jpg").convert("L")
print(im.mode)
```

    L


## 6. 图片增强

### 滤波器


```python
from PIL import ImageFilter
out = im.filter(ImageFilter.DETAIL)
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f1172073860>




![png](output_44_1.png)



```python
# 高斯模糊
out = im.filter(ImageFilter.GaussianBlur)
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f117204f470>




![png](output_45_1.png)



```python
# 边缘增强
im.filter(ImageFilter.EDGE_ENHANCE)
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f1171faf0b8>




![png](output_46_1.png)



```python
# 普通模糊
im.filter(ImageFilter.BLUR)
# 找到边缘
im.filter(ImageFilter.FIND_EDGES)
# 浮雕
im.filter(ImageFilter.EMBOSS)
# 轮廓
im.filter(ImageFilter.CONTOUR)
# 锐化
im.filter(ImageFilter.SHARPEN)
# 平滑
im.filter(ImageFilter.SMOOTH)
# 细节
im.filter(ImageFilter.DETAIL)
```




![png](output_47_0.png)



### 应用点变换


```python
# multiply each pixel by 1.2
out = im.point(lambda i: i * 1.2)
```

### 增强图像


```python
from PIL import ImageEnhance

enh = ImageEnhance.Contrast(im)
out = enh.enhance(1.3)
plt.imshow(out)
```




    <matplotlib.image.AxesImage at 0x7f1171f906a0>




![png](output_51_1.png)


## 7. 图片序列


```python
from PIL import Image

im = Image.open("chi.gif")
im.seek(1) # skip to the second frame

try:
    while 1:
        im.seek(im.tell()+1)
        # do something to im
except EOFError:
    pass # end of sequence
```


```python
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f1171eec208>




![png](output_54_1.png)


### 使用ImageSequence Iterator类


```python
from PIL import ImageSequence
count = 1
for frame in ImageSequence.Iterator(im):
    # ...do something to frame...
    count += 1
    if count % 10 == 0:
        plt.imshow(frame)
        plt.show()
```


![png](output_56_0.png)



![png](output_56_1.png)



![png](output_56_2.png)


## 8. Postscript printing
### Drawing Postscript


```python
from PIL import Image
from PIL import PSDraw

im = Image.open("hopper.ppm")
title = "hopper"
box = (1*72, 2*72, 7*72, 10*72) # in points

ps = PSDraw.PSDraw() # default is sys.stdout
ps.begin_document(title)

# draw the image (75 dpi)
ps.image(box, im, 75)
ps.rectangle(box)

# draw title
ps.setfont("HelveticaNarrow-Bold", 36)
ps.text((3*72, 4*72), title)

ps.end_document()
```

    %!PS-Adobe-3.0
    save
    /showpage { } def
    %%EndComments
    %%BeginDocument
    /S { show } bind def
    /P { moveto show } bind def
    /M { moveto } bind def
    /X { 0 rmoveto } bind def
    /Y { 0 exch rmoveto } bind def
    /E {    findfont
            dup maxlength dict begin
            {
                    1 index /FID ne { def } { pop pop } ifelse
            } forall
            /Encoding exch def
            dup /FontName exch def
            currentdict end definefont pop
    } bind def
    /F {    findfont exch scalefont dup setfont
            [ exch /setfont cvx ] cvx bind def
    } bind def
    /Vm { moveto } bind def
    /Va { newpath arcn stroke } bind def
    /Vl { moveto lineto stroke } bind def
    /Vc { newpath 0 360 arc closepath } bind def
    /Vr {   exch dup 0 rlineto
            exch dup neg 0 exch rlineto
            exch neg 0 rlineto
            0 exch rlineto
            100 div setgray fill 0 setgray } bind def
    /Tm matrix def
    /Ve {   Tm currentmatrix pop
            translate scale newpath 0 0 .5 0 360 arc closepath
            Tm setmatrix
    } bind def
    /Vf { currentgray exch setgray fill setgray } bind def
    %%EndProlog
    gsave
    226.560000 370.560000 translate
    0.960000 0.960000 scale
    gsave
    10 dict begin
    /buf 384 string def
    128 128 scale
    128 128 8
    [128 0 0 -128 0 128]
    { currentfile buf readhexstring pop } bind
    false 3 colorimage

    %%%%EndBinary
    grestore end

    grestore
    72 144 M 504 720 0 Vr
    /PSDraw-HelveticaNarrow-Bold ISOLatin1Encoding /HelveticaNarrow-Bold E
    /F0 36 /PSDraw-HelveticaNarrow-Bold F
    216 288 M (hopper) S
    %%EndDocument
    restore showpage
    %%End


## 9. 更多关于图像读取


```python
from PIL import Image
im = Image.open("hopper.ppm")
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f1171d5a400>




![png](output_60_1.png)



```python
from PIL import Image
with open("hopper.ppm", "rb") as fp:
    im = Image.open(fp)
    plt.imshow(im)
```


![png](output_61_0.png)


## 10. 控制解码器

### 使用草稿（draft）模式读取


```python
im = Image.open("lena.jpg")
print("original =", im.mode, im.size)

im.draft("L", (100, 100))
print("draft =", im.mode, im.size)
```

    original = RGB (512, 512)
    draft = L (128, 128)



```python

```
