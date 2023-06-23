Euclidean Distance

Aytaylik bizda `X` vectorlar fazosi bor va u hayvonlar haqidagi ma'lumotlarni o'z ichiga oladi. Bular `length` va `weight` bilan o'lchanadigan hayvonlarning uzunligi va bo'yidir. Qolaversa, hayvonlar yoshlarining darajasiga qarab ham belgilangan (`young = 0`, `mid = 1`, `adult = 2`). 

Bizda quyidagicha ma'lumotlarimiz bor:

```python
import numpy as np
X = np.array([[6.6, 6.2, 1],
              [9.7, 9.9, 2],
              [8.0, 8.3, 2],
              [6.3, 5.4, 1],
              [1.3, 2.7, 0],
              [2.3, 3.1, 0],
              [6.6, 6.0, 1],
              [6.5, 6.4, 1],
              [6.3, 5.8, 1],
              [9.5, 9.9, 2],
              [8.9, 8.9, 2],
              [8.7, 9.5, 2],
              [2.5, 3.8, 0],
              [2.0, 3.1, 0],
              [1.3, 1.3, 0]])
```

# Ma'lumotlarni tartiblab olamiz
Birinchi ma'lumotlarni DataFrame formati shaklida o'zgartirib olamiz va mos ravishda ularning ko'rsatgichlarini belgilaymiz:

```python
import pandas as pd
df = pd.DataFrame(X, colums = ['weight', 'length', 'label'])
df
```
Endi ma'lumotni uch guruhga ajratilgan tasvirini ekranga chiqaramiz. Ular ko'rsatgichlari bilan belgilangan va turli ranglarda ifodalangan.

```
matplotlib inline
ax = df[df['label'] == 0].plot.scatter(x='weight',y='length', c='blue',label='young')
ax = df[df['label'] == 1].plot.scatter(x='weight',y='length', c='orange',label='young')
ax = df[df['label'] == 2].plot.scatter(x='weight',y='length', c='red',label='young')
```

Yuqorida ko'rib turganimizdek bu uchta guruh bu ikkita ko'rsatgichlari bo'yicha aniq bir biridan farqlanib turibdi. Aytaylik biz k-NN metodini qo'lash orqali bu ma'lumotlarimizni klasterlarga ajratib olaylik. Algoritm bu ma'lumotlarning bir biridan qanchalik uzoq yoki yaqin masofada joylashganiga qarab ularni klasterlarga ajratishi kerak.

Euclidean Distance funksiyasi quyidagicha ifodalanishi mumkin:

![Book logo](./TUTORIALS/SECTIONS/euclidean_distance/euclidean_distance_fuction.png)

Bu yerda x va y lar vectorlar hisoblanadi.