## K-Nearest Neighbors Algorithm

K-Nearest Neighbors Algoritmi (bazan KNN yoki k-NN ham deb yuritiladi) parameterli bo'lmagan (non-parametric), supervised learning classifier hisoblanadi va mustaqil malumotlar nuqtalarining o'zaro bir biriga nisbatan bo'lgan yaqinligini yoki o'xshashligini hisobga olgan holda klassifikatsiyalash yoki bashorat masalalarini yechishda qo'llaniladi. Bu algoritmni regressiya yoki klassifikatsiyalash mauammolarini yechishda qo'llash mumkin bo'lsada, asosan klassifikatsiyalash masalalarida ko'proq ishlatiladi va bir biriga yaqin bo'lgan nuqtalarni bir toifaga mansub nuqtalar deb xulosa beradi.

Klassifikatsiyalash masalalarida, biror bir klasni baholash ko'pchilik ovozlar hisobiga amalga oshadi. Misol uchun berilgan nuqtaning atrofida eng ko'p uchraydigan klasning yorlig'(label)iga  qarab bu nuqta ham shu klasga tegishli deb hulosa qilinadi. Va bu inglizcha adabiyotlarda `majority voting` deb yuritiladi. Ikki kategoriyali ma'lumotlar bilan ishlaganda ko'pchilik ovozlarning soni 50% dan baland bo'lishi talab etiladi. Lekin ikkidan ortiq kategoriyaga ega ma'lumotlar bilan ishlaganda ovozlar soni 50% dan baland qilib belgilash talab etilmaydi. Misol uchun berilgan nuqtaning qaysi klasga tegishli ekanligini 25%dan baland ovoz olgan klasning yorlig'i (label) bilan belgilash mumkin.

![Book logo](./machine-learning/TUTORIALS/SECTIONS/knn/image.png)

Wisconsin-Madison Universiteti bu haqda juda yaxshi misol keltirib o'tgan ([link](https://sebastianraschka.com/pdf/lecture-notes/stat479fs18/02_knn_notes.pdf)).

Regressiya masalalari ham klasifikatsiya masalalari kabi bir xil tushinchadan foydalanadi, lekin bu holatda, klassifikatsiya haqida bashoratni amalga oshirish uchun o'rtacha `k nearest neighbors` tanlanadi. Bu yerda asosiy farq klasifikatsiya tavsiflovchi qiymatlar uchun ishlatiladi, regressiya esa uzluksiz qiymatlar uchun ishlatiladi. Biroq klasifikatsiya qilinishidan oldin, masofa hisoblab chiqilishi kerak. Bu holatda masofani hisoblashda eng ko'p hollarda `Euclidean distance` ishlatiladi, va keyinchalik bu haqda to'liqroq o'rganamiz.

Shu o'rinda yana shuni aytib o'tish kerakki KKN algoritmi `lazy learning` modellar oilasiga mansub bo'lib, bu shuni anglatadiki u faqat o'qitish jarayoni davom etayotgan bosqichga qarshi *training dataset*ni saqlaydi. Bu yana shuni anglatadiki hamma hisoblashlar klassifikatsiya yoki bashorat qilish jaroyoni bo'layotganda amalga oshadi. U hamma o'qitilgan ma'lumotlarni saqlash uchun xotiraga juda chuqur bog'langanligi uchun `instance-based` yoki `memory based` o'qitish usuli ham deb yuritiladi. 

U hozir avvalgidek mashxur bo'lmasada, soddaligi va aniqligi sababli *data science*da hali ham eng birinchi o'rganiladigan algoritmlardan bittasi hisoblanadi. Biroq, ma'lumotlar to'plami o'sib borgan sari, KNN tobora samarasiz bo'lib boradi va modelning umumiy ishlashiga salbiy tasir etadi. U odatda sodda tavsiya qilish tizimlari, ketma-ketliklarni aniqlash, *data mining*, moliya bozorini bashorat qilish, raqamli (electron) hujumlarni aniqlash va shu kabi tizimlarni yaratishda ishlatiladi.

KNNni hisoblash: Masofa o'lchovchilari (*distance metrics*)

Eslatib o'tamiz, `k-nearest neighbor` algorithmning maqsadi izlanayotgan nuqtaning eng yaqin qo'shnilarini aniqlashdan iboratdir va shundagina biz izlanayotgan nuqtaning qaysi klasga tegishli ekanligini belgilay olamiz. Buni amalga oshirish uchun KNNning bir nechta talablari bor:

### Masofa ko'rsatkichlarini aniqlash
Berilgan nuqtalarga eng yaqin ma'lumotlar nuqtalari qaysilari ekanligini aniqlash uchun, berilgan nuqta va qolgan ma'lumotlar nuqtalari orasidagi masofalarni aniqlash kerak bo'ladi. Shu masofalar ko'rsatkichlari so'ralayotgan nuqtalarni turli guruhlarga ajratadigan chegaralarni chizishga yordam beradi.

Masofa ko'rsatkichlarini aniqlaydigan bir qancha usullari bo'lsada bularning orasidan bazilarini ko'rib chiqamiz

***`Euclidean distance (p=2)`***: Bu eng keng qo'llaniladigan masofa o'lchovidir va u haqiqiy qiymatli vektorlar bilan cheklanadi holos. U quydagi formuladan foydalanib, so'ralayotgan nuqta va qolgan boshqa nuqta orasidagi to'g'ri chiziqli masofani hisoblaydi.

![Book logo](./machine-learning/TUTORIALS/SECTIONS//euclidean_distance/euclidean_distance_fuction.png)

***`Manhattan distance (p=1)`***: Bu yana bir mashhur masofa ko'rsatgichi bo'lib, ikki nuqta orasidagi mutlaq qiymatni o'lchaydi. Bu yana  `taxicab distance` yoki `city block distance` ham deb ataladi. Chunki u odatda shahar ko'chalari orqali bir manzildan boshqasiga qanday o'tish mumkinligini ko'rsatadigan panjara bilan tasvirlaydi. 

<img src="/machine-learning/TUTORIALS/SECTIONS/knn/manhattan_distance_formula.png" width="100">
<img src="/machine-learning/TUTORIALS/SECTIONS/knn/manhattan_distance2.png" width="100">




