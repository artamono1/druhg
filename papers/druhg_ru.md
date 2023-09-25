https://docs.google.com/document/d/1x__iCjU89ZQRa_pJ3f2PIsYJkGokj4Yy7u6UlJVptp8

# Друг: Диалектический Рефлексивный Универсальный Группировщик
Pavel Artamonov

*Last of the Soviets*

Август 27, 2023

## Аннотация

В статье вы познакомитесь с методом кластеризации основанным на философии диалектического материализма. Данные саморазовьются во вложенные структуры разных плотностей, что также позволяет ловить глобальные выбросы. Друг не требует параметров, что делает его основным инструментом первичного анализа данных.

Работа погрузит читателя в мир диалектики и предложит формализацию диалектического метода. В отличии от идеялистических гегелевских триад отражающихся от Абсолютного Духа, за основу берутся материялистические противоположности отражений Бытия от другого Бытия.

*Ключевые слова*: Кластеризация, Диалектика, Квантовая плотность, Мао, Гегель

## Вступление

Кластеризацией обычно называют попытку группировки данных с точки зрения человеческой интуиции. К сожалению, наша интуиция плохо определена и очень контекстно обусловлена[15]. Некоторые считают, что кластеры это человеческий конструкт – Я буду кластеризовать, как мне хочется. Другие могут возразить - кластера объективны, давай я выберу тебе k кластеров по этому правилу k-means[9]. С виду различные точки зрения на деле оказываются идентичными, в обоих случаях приоритизируется наблюдатель, приоритизируется Идея, а не Материя. Существовали ли кластеры, до того как на них взглянул специалист по данным?

Многие научные сферы приоритизируют наблюдателся. Методы кластеризации не являются исключением в этом. Параметрические модели применяются к данным. Модель двигает выводы. И в какой то момент, Модель начинает жить своей жизнью. Изменившиеся данные не соответствующие модели, становятся проблемами данных. Модель-надстройка правит балом.

Чтобы вылечиться от этой детской болезни, нужно найти независимую от наблюдателя реальность. Реальность и должна стать моделью. Данные должны быть двигателем кластеров. Метод позволяющий такое был продемонстрирован в работах Гегеля[2], Бытие постепенно стало Сущностью. В настоящей работе, точки данных разовьются в кластеры, а Гегель опять будет поставлен с головы на ноги.

Друга можно отнести к семье плотностных алгоритмов кластеризации, но без прямого указания плотности. Стандартный плотностный алгоритм DBSCAN[13] сканирует точки на наличие заданной плотности (ϵ-радиус и k-количество точек) и определят кластера, которые плотнее, чем ввод. Второе поколение плотностных алгоритмов HDBSCAN[14][16] выбрасывает параметр ϵ-радиус, и оставляет параметр k-точек, что позволяет проще манипулировать данными. Работа учёного-наблюдателя состоит в правильном подборе параметров. Нужно ещё раз подчеркнуть - Кластеры уже существуют, наблюдатель пытается их найти, а не представить в голове!

Можно сказать, что Друг это третье поколение избавления DBSCAN от параметров. Друг практически детерминистичен, не зависит от масштаба, и не требует параметров. На выходе получаются вложенные кластера разных “плотностей”. Дальнейшее повествование окунёт в философские глубины диалектики, подкрепляя математическими выкладками с иллюстрациями и формулами. Всё это можно пропустить и посмотреть код в Приложении.

## Диалектическое расстояние

> Материя – это философская категория для обозначения объективной реальности, которая дана человеку в ощущениях его, которая копируется, отображается, фотографируется нашими ощущениями, существуя независимо от них.
>
> В.И. Ленин, Материализм и Эмпириокритицизм[5]

Материя движется и меняет свои формы, она не пропадёт из-за открытия новых веществ. Более того, материя может быть не физической. Социальная, биологическая, механическая, химическая, физическая - такие типы материи бывают по Энгельсу[4]. Мы добавим ещё и информационный тип в этот список.

Современный мир насыщен таблицами с данными. Одна таблица обычно содержит строки с однопорядковыми данными. Такие данные и метрика для сравнения, могут быть рассмотрены как бытия в пространстве. Бытие и пространство это философские категории, такие понятия, которые отражают самые основные свойства и отношения в мире. Развитие этих категорий приведёт к универсальным законам, которые можно будет применять в том числе и к данным.

Чтобы дать вам почувствовать универсальность и сгладить погружение в диалектический материализм, мы представим вам понятие диалектического расстояния на примерах двух типов материи: социальной на знакомствах-отношениях и физической на квантовых экспериментах.

### Взаимность

> Во всём поступайте с людьми так, как хотите, чтобы они поступали с вами.
>
> Библия, Матфея 7:12

Испытывали ли вы безответные чувства к кому-то? Как могло быть такое, что самый близкий для вас человек, не отвечал вам взаимностью? Пытались ли вы разобраться, что чувствует другая сторона? В голову залезть нам не удастся, но  восстановить чужое отношение мы попробуем (Рис.1).

Для этого, мы поднимемся на категориальное мышление и заполним таблички данными. Для вашей таблицы отношений всё просто: Вы, она, Вася, Ваня и попрошайка, с которым вы перекидываетесь словами по пути на работу.
Её лист ярок: Она, собачка, мама, Клава и Вы. Вы под номером пять.

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_reciprocity.png" alt="reciprocity" width="200"/> |
:--------:|
> Рис.1: В синем списке красное сердце под номером 2. Напротив собака. В розовом списке синий ромб под номер 5, напротив пирата. |

И тут приходит осознание.

Вы пятый в её списке, а в вашем списке пятый это попрошайка! Вам нужно относиться соответствующим образом! (не сочтите это за совет)

Она в свою очередь могла бы восстановить ваше к ней отношения, если бы чувства к вам умножила на 5/2.

### Квантовая “плотность”

> Что первично сознание или материя?
>
> Основной вопрос философии

Знаменитый эксперимент[1] с пушкой, стреляющей частицами через две щели, рисует не две полосы, а целую зебру интерференции. Как будто пушка заряжена не частицей, а жидкостью, проходящей одновременно через обе щели. Частичка ведёт себя как волна(корпускулярно-волновой дуализм). Мы предложим разрешение этого противоречия, переведя фокус с наблюдателя, выбирающего наблюдаемый объект, на пространство.

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_double_slit1a.png" width="150"/>            |  <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_double_slit1b.png" alt="reciprocity" width="150"/> |
:-------------------------:|:-------------------------:|
| В сфере А одна щель        |  В сфере A’ две щели.     |

> Рис 2: Стена с двумя щелями и зелёный шарик. Сферы A и A’ отличаются по “плотности”. Сферы B и B’ совпадают.

На рисунке 2 представлены два случая, когда частичка (зелёный шарик) находится в разных положениях относительно щелей, но на одинаковом расстоянии от стены. Взяв пространство за центр, мы получим две пересекающиеся сферы. Рассмотрим "плотности" сфер в разных случаях. Правые сферы B и B’ идентичны. Но сфера A содержит больше вещества, чем сфера А’. Центричность на пространстве будет учитывать эту разницу, и породит возможность сравнения этих случаев.

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_double_slit2a.png" width="150"/>            |  <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_double_slit2b.png" alt="reciprocity" width="150"/> |
:----------------------------:|:-------------------------:|
| Рост D попал в нижнюю щель. |  Рост D’ за счёт верхней и нижней стенки вне сферы A’.      |

> Рис.3. Зеленая площадь равна синей, т.е. Площади вещества A=D и A’=D’. При этом радиус D > D’.

Расстояние d не подходит для измерения пространства между сторонами, так как оно не отражает разницы в количестве вещества. Чтобы найти диалектическое расстояние, учитывающее вещество обеих сторон, нужно увеличить радиус сферы B до уровня вещества в сфере А. То же самое проделать для сферы B` (Рис 3). Диалектические расстояния и количеств вещества различаются, а значит и “силы” воздействующие на стороны тоже различны.
Предложенное взаимодействие не является целью настоящего исследования, и является всего лишь демонстрацией рассмотрения приоритета материи-пространства над сознанием-наблюдателем.

| △ |
:----:

Эти примеры показали как диалектически разрешать зависимость от наблюдателя. Примеры разбирались аналитически, обнаруживались две противоположные стороны, производилось решение включающее обе стороны. Но откуда взять противоположности для анализа, если решается абстрактная задача в общем? Саморазвитие категорий породит свои противоположности. Частичка “не” позволит синтезировать следующий шаг процесса. Будет получаться, что всё новое, это основанное на старом, и для наблюдателя не останется места.

## Единичное и противоречия

> Есть только один субъект – мир в его целом
>
> Основа диалектического материализма

Пространство существует. Пространство это всё. Пространство одно.

Не пространство это ничто. Не всё это ничто. Ничтов много.

Не всё-ничто это дыры в пространстве, называемые точками. Точка сама по себе ничто. Только отношение к другим точкам вкладывает в неё смысл.

Отношение точки к другой точке есть направление, захватывающее и ограничивающее часть пространства. Такое отношение субъект-объект можно представить в виде пространства-сферы с центр-субъектом и радиус-направлением в объект. В пространство-сферу попадёт  точек, это количественное свойство отношения. Не количественное, есть качественное свойство – это “радиус” расстояние от субъекта к объекту. Количество и качество неразрывны и вместе образуют количественно-качественные (далее к-к) свойства отношения субъекта s к объекту:

![r d(r)](https://latex.codecogs.com/svg.image?\overrightarrow{r_{s}}\cdot&space;d_{s}(\overrightarrow{r_{s}}))

Не субъект-объектное отношение это обратное отношение, т.е. объект-субъектное. С обратным направлением из объекта, но с сохранением центральности субъекта. Сфера-пространство количественно смещается в объект, но качественно остаётся в субъекте (Рис. 4). Обратное количество образуется внешним подсчётом, а качество функцией от этого количества. Субъект отражается в объект, получает искажённую картину действительности. Отношение объекта к субъекту s:

![r’ d(r’)](https://latex.codecogs.com/svg.image?\overleftarrow{r_{s}}\cdot&space;d_{s}(\overleftarrow{r_{s}}))

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_reverse_vector.png" alt="reciprocity" width="200"/> |
:----------------------------:|
> Рис 4: Субъект под номером 1. Объект под номером 3. В (левой)сфере субъекта 3 точки, в (правой)сфере объекта 5 точек. Новый обратный розовый 5-радиус больше, чем начальный синий 3-радиус. |  

**Противоположности**. Для субъекта s одно отношение к объекту составляет два противоположных выражения опосредованных пространством:

![r d(r) vs r’ d(r’)](https://latex.codecogs.com/svg.image?\overrightarrow{r_{s}}\cdot&space;d_{s}(\overrightarrow{r_{s}})\quad&space;vs\quad\overleftarrow{r_{s}}\cdot&space;d_{s}(\overleftarrow{r_{s}}))

Таких отношений много для каждого субъекта и самих субъектов также много. Для дальнейшего движения множественность должна быть снята. Не множественность отношений есть особенное отношение, самое близкое. Для того чтобы найти такое отношение, нужно снять три момента: единичность, всеобщность и особенность.

**Мера**. Мера очищает качество от количества и позволяет сравнивать противоположности. Математически это означает, что пары чисел будут приведены к одиночным значениям. Парность препятствует сравнению (нельзя ответить, что больше пара (2; 1,5) или пара (3;1)). Пары к-к нужно привести к одинарным качествам, для этого разделим обе стороны на количество точек из (правой) объектной части. Не количество-качество это меры отношений субъект-объекта

![r d(r) /r’ vs r’ d(r’) /r’](https://latex.codecogs.com/svg.image?\frac{\overrightarrow{r_{s}}\cdot&space;d_{s}(\overrightarrow{r_{s}})}{\overleftarrow{r_{s}}}\quad&space;vs\quad\frac{\overleftarrow{r_{s}}\cdot&space;d_{s}(\overleftarrow{r_{s}})}{\overleftarrow{r_{s}}})

Или

![r/r’ d(r) vs d(r’)](https://latex.codecogs.com/svg.image?\frac{\overrightarrow{r_{s}}}{\overleftarrow{r_{s}}}\cdot&space;d_{s}(\overrightarrow{r_{s}})\quad&space;vs\quad&space;d_{s}(\overleftarrow{r_{s}}))

**Тотальность**. Наибольшее(всеобщее) из мер будет удовлетворять “плотностям” обеих сторон. 

Противоположности внутри отношения разрешены, но противостояния множественности остались.

**Суждение**. Второе отрицание множественности заключается в нахождении наименьшего(особенного) из всех отношений. Что производит абстрактное всеобщее

![max(r/r’ d(r); d(r’))](https://latex.codecogs.com/svg.image?\max(\frac{\overrightarrow{r_{s}}}{\overleftarrow{r_{s}}}\cdot&space;d_{s}(\overrightarrow{r_{s}});\quad&space;d_{s}(\overleftarrow{r_{s}})))

Первая фаза завершилась единством противоположностей. Точки ограничили пространство и произвели кусочек пространства. Пространство провернулось и сняло с себя точки, их субъектность и направленность. Наступает следующая фаза в которой мы познакомимся с общностями и становлением кластеров.

**Перезапуск процесса**. Как мы узнаем в дальнейшем, не точки отрезали пространство, а общности посредством точек. Поэтому при перезапуске процесса нахождения наименьшего отношения нужно учитывать принадлежность точек к общностям. В конечном итоге все точки соберутся в одну общность, которая может быть представлена остовным деревом[18].

**Упорядоченность соединений**. Процесс сортирует соединения, дружит одинаковые плотности, отдаляя присоединения выбросов. На примере Рис. 5, все точки равноудалены друг от друга, но тем не менее, вначале соединятся тело и рёбра, а вершины в самом конце. Для одинаковой “плотности”  расстояния примерно равны . Для областей с разными “плотностями” и , что снижает очередность. Для выбросов , приоритетность по остаточному принципу.

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_square_order.png" alt="reciprocity" width="300"/> |
:----------------------------:|
> Рис. 5: Ранги и расстояния в порядке возрастания меры: равноплотные ядро и рёбра, потом ядро-рёбра, последними выбросы вершины.

## Диалектика отрицания

> Борьба и единство противоположностей
>
> Закон диалектики

Частичка “не” позволила развернуть внутренние противоречия пространства. Такое отрицание раскрывает определённость моментов и движет процесс вперёд к своему само-возврату. Такое отрицание лежит в сердце диалектического развития.

Отрицания бывают разные, например бывает разрушительное отрицание. Отрицание жука, если его раздавить. Или бывает отрицание полного не, это когда из математического множества вычитают элемент. Такие виды отрицаний мертвы, они не произведут развития и дальнейшего отрицания. В отличие от них, диалектическое отрицание остаётся в контексте отрицаемого, но в то же время производит на свет сущность высшего порядка. Например, пофиксить багу в коде.

Движение на более высокий уровень не должно приносить ничего из вне. Система логики Гегеля начинается из Ничто-Бытие-Становление, развивается в Количество-Качество-Меру, а далее возвышается до Абсолюта. Всё развитие происходило изнутри и Абсолют уже скрывался в Ничто. В настоящей работе, заражение внешними идеями тоже было предотвращено, но развитие происходило за счёт отражения от реальных Бытия, а не от идеялистического Абсолюта.

Повествование Гегеля синтезируют конкретные триады отрицаний, а потом они рассматриваются через анализ моментов. Такой триадо-центричный подход не ложится в формулы, поэтому пришлось привлечь подход другого мыслителя. Мао Цзедун в своей работе “Относительно противоречия” подчеркнул роль противоположностей многочисленными примерами, что привело к противоположность-центричному результату, что прекрасным образом легло в математику. Как мы увидим в дальнейшем, цитата Мао описывает появление кластера “внешние причины являются условием изменений, а внутренние причины — основой изменений, причём внешние причины действуют через внутренние”[7].

Вкратце, - как писал Ленин, - диалектику можно определить как учение о единстве противоположностей[6]. Для самостоятельного изучения диалектики рекомендуем начать с основных законов из эпиграфов настоящей статьи, их можно найти в работе Сталина “Диалектический и исторический материализм”[8].

## Особенное и субстанция

> Переход количественных изменений в качественные
>
> Закон диалектики

В первой фазе единичные отрезали кусочек пространства RD. Получилось абстрактное всеобщее. А для каждой из двух примыкающих точек это абсолютное всеобщее, определяющая их граница. Таков конец второй фазы. Далее последует слияние в общность и перезапуск процесса, где уже будет присутствовать общность с двумя одноточечными кластерами определённых границей RD.

**Противоположности**. В последующих перезапусках соединятся будут общности, а не точки. И внешняя абстрактная всеобщность RD столкнётся с уже принятыми внутренними абсолютными всеобщими. Возникнет множество противоположностей внешнего и внутреннего:

![RD vs kg](https://latex.codecogs.com/svg.image?RD\quad&space;vs\quad&space;k_i\cdot&space;g_i) , где *kg* это количество-качество кластера i из общности.

У каждого кластера есть такое отношение. Кластера состоят в общности. Отсутствие единства решается снятием трёх моментов: особенности, единичности и всеобщности.

**Мера**. Приводим внешнее к внутреннему количеству кластера, снимаем количественную особенность.

![RD/k vs kg/k](https://latex.codecogs.com/svg.image?\frac{RD}{k_i}\quad&space;vs\quad\frac{k_i\cdot&space;g_i}{k_i})

или

![RD 1/k vs g](https://latex.codecogs.com/svg.image?RD\frac{1}{k_i}\quad&space;vs\quad&space;g_i\notag)

**Тотальность**. Сумма (единичность) собирает кластера и их отражения в единые:

![RD sum 1/k vs sum g](https://latex.codecogs.com/svg.image?RD\sum_{K}{\frac{1}{k_i}}\quad&space;vs\quad\sum_{K}{g_i})

**Суждение**. Получилось противостояние целого и частей. Наибольшее(всеобщее) из сторон проявляет видимость общности и её к-к свойства:

![KG = max (RD sum 1/k; sum g)](https://latex.codecogs.com/svg.image?KG=\max(RD\sum_{K}{\frac{1}{k_i}}\quad;\quad\sum_{K}{g_i}))

Если побеждает новое, то граница переходит в общность и рождается кластер¹(¹ – в терминах Гегеля, кластер это субстанция, общность это субстрат) со свойствами ![KG = RD sum 1/k](https://latex.codecogs.com/svg.image?KG=RD\sum_{K}{\frac{1}{k_i}}). Если же старое устоит, то граница игнорируется.

*По сути, формула кластеризации проверяет может ли новое расстояние побороть старую структуру кластеров минус разбавление выбросами.*

**Перезапуск процесса**. Для дальнейших противостояний новообразованный кластер понесёт ![G = RD avg 1/k](https://latex.codecogs.com/svg.image?G~=~RD\cdot&space;avg{\frac{1}{k_i}}) (усреднённую внутреннюю структуру), а с другой стороны 1/K. Чем сложнее структура кластера, тем сложнее будет его переформировать.

**Выбросы**. Кластера с k=1 значительным образом усиливают ![RD sum 1/k)](https://latex.codecogs.com/svg.image?RD\sum_{K}{\frac{1}{k_i}}), то есть шанс на кластеризацию.

**Изолированность сторон соединения**. Граница RD независимо влияет на обе стороны соединения. Возможны три исхода: образуются 2, 1, или 0 кластеров за одно соединение.

## Всеобщее и слияние

> Отрицание отрицания
>
> Закон диалектики

В первой фазе пространство обосновало себя посредством общностей. Во второй фазе это внешнее обосновало две общности. Две части утвердили себя и свою видимость. Теперь они доступны друг другу для слияния. Их слияние порождает новое состояние всей системы.

Перезапуск всего алгоритма с учётом новых общностей произведёт новое слияние. И так раз за разом отдельные будут расти, пока не останется одна единая общность. Развитие завершится концом кластеризации Друга.

Диалектика отрицания не знает конца. Придя к невозможности развития, произойдёт отрицание всей системы. Тик времени перезапустит саморазвитие пространства с перемещёнными точками. Фаза перемещения выводится из циклической симметрии фигур умозаключения(см Таблицу 1). Умозаключения это самые обычные отношения вещей[6]. Роза есть растение; растение нуждается во влаге; следовательно, роза нуждается во влаге, т.е. роза, как единичное, смыкается через особенное, со всеобщим[3].

Подобные смыкания моментов мы наблюдали в первой фазе: субъект-объект сошлись в одном отношении, и через эти отношения нашлось одно. То есть В-всеобщность объединила в Е-единичном (В-Е), а из Е-единичных выделилось О-особенное (Е-О). Результат опять имеет все три момента Е-В-О, но уже как особенное. Поэтому следующее умозаключение сдвигается в Е-О и О-В. Для третьей фазы останется О-В и В-Е, вернётся снятая направленность и точки сместятся в центры. Реализацию этого прогноза оставим на другой раз.

|  Фаза |  Среда отражения    |  Субъект |  Противо- положности          | Умозаключение |             |          |  Результат                |                    |
|-------|---------------------|----------|-------------------------------|:-------------:|:-----------:|:--------:|---------------------------|--------------------|
|       |                     |          |                               |      Мера     | Тотальность | Суждение |                           |                    |
|   1   |     Пространство    |   Точки  |   rd vs rd Субъект и объект   |      1/r      |     max     |    min   |    Соединение общностей   |     Граница RD     |
|       |          В          |     Е    |                               |       Е       |      В      |     О    |             О             |                    |
|   2   | Общность соединения | Кластеры | RD vs kg Внешнее и внутреннее |      1/k      |     sum     |    max   |       Кластеризация       |    Видимость KG    |
|       |          Е          |     О    |                               |       О       |      Е      |     В    |             В             |                    |
|   3   |       Кластер       | Общности |            KG vs nv           |      1/n      |     min     |    sum   | Притяжение и отталкивание | Центр NV (коорд-ы) |
|       |          О          |     В    |                               |       В       |      О      |     Е    |             Е             |                    |
> Проект схемы полного автомата. В - Всеобщность, О - Особенность, Е - Единичность(отдельность).

## Эксперименты по кластеризации

> Практика критерий истины
>
> Карл Маркс, Тезисы о Фейербахе, 1845

**Наборы данных**. Мы проводили запуски на синтетических данных sklearn, Хамелеона, и геометрических наборах. Для всех них используется Евклидова метрика.

**Алгоритм**. Код доступен публично (https://github.com/artamono1/druhg) и может быть запущен с помощью пары команд. Псевдокод в Приложении. Запускали на Питоне с версией druhg 1.5.0.

В отличии от других алгоритмов, Другу не нужны параметры. Текущая имплементация использует KD-дерево с ограничением в k-соседей для повышения производительности. Пробел или выброс может кластеризовать весь набор данных, пользователь может выбирать нужный масштаб кластеров без перезапуска алгоритма.

### Производительность и точность

(Рис. 6) Внутреннее KD-дерево использует k-близких-соседей (k-nn), как параметр. Чем больше k, тем точнее результат, но скорость ниже. После некоторого k результат сходится.

| <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_diffNNs.png" alt="run_diffNNs" width="300"/> |
|----|
>Рис 6. Малый k-nn не включает соединений между лунами. Больший k-nn дольше выполняется.

### Иерархия и фильтр по размеру

Единственный запуск строит иерархию вложенных кластеров.

Обычно доступ к слоям кластеров производят за счёт ограничения расстояния кластеризации. Мы предлагаем новый метод ограничения по размеру (Рис. 7).

| <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_chameleon05.png" alt="run_chameleon05" width="100"/> 5%| <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_chameleon25.png" alt="run_chameleon25" width="100"/> 25%| <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_chameleon45.png" alt="run_chameleon45" width="100"/> 45%|
|---|---|---|
| <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_chameleon75.png" alt="run_chameleon75" width="100"/> 75%|<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_chameleon100.png" alt="run_chameleon100" width="100"/> 100%| |

> Рис 7: Хамелеон. Один запуск Друга. Функция Relabel ограничивает размер кластера сверху X% от всего кол-ва точек.

### Геометрические фигуры

(Рис. 8). Вначале соединяются равноплотные области, образуя тела фигур (синий цвет). Соединения внутри тела могут менять последовательность при перезапусках, но кластеры при этом останутся те же. Результаты не зависят от изменения масштаба.

Подобно человеку Друг определил грани, рёбра, вершины.

|<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_line.png" alt="pdn_line" height="100"/> | <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_square.png" alt="pdn_square" height="100"/> | <img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_cube.png" alt="run_cube" height="100"/> |
|---|---|---|
|Линия|Квадрат|Куб|

> Рис 8: Порядок формирования и соединений геометрических фигур образованных сеткой равноудалённых точек.

### Сравнения с другими

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/run_comparison.png" alt="run_comparison" height="300"/>

> Рис 9: Сравнение стандартных алгоритмов на наборе данных sklearn. Последней колонкой Друг.

(Рис. 9) Друг смог поймать отдельные глобальные выбросы и малые кластеры с единого запуска. Некоторые алгоритмы получили лучший результат, заранее имея подобранные параметры.
Хаос в правом нижнем углу подчёркивает философское ядро алгоритма — невозможно задать Бытие без Иного.

### Критическая точка

<img src="https://raw.githubusercontent.com/artamono1/druhg/paper_druhg/papers/druhg/pdn_sandpiles.png" alt="pdn_sandpiles" width="100"/>

> Рис 10. Сложение куч песка и песчинок на разных расстояниях.

(Рис. 10) Когда куча становится кучей? Сможет ли Друг дать ответ на этот тысячилетний вопрос?

## Заключение

> О сколько нам открытий чудных
> Готовят просвещенья дух
> И Опыт, сын ошибок трудных,
> И Гений, парадоксов друг,
> И Случай, бог изобретатель
>
> А.С. Пушкин, 1929

В работе был представлен новый подход к кластеризации, который: (i) предоставляет полную плотностную иерархию кластеров; (ii) не требует параметров; (iii) выбор метрики задаёт ближайшие расстояния, от которых зависит всё остальное. Философское повествование провело через диалектику и определило кластер. Эксперименты на различных датасетах показало SOTA результаты. Эта работа открывает широкие возможности для будущих исследований: оценка и улучшение производительности; открытия свойств диалектического расстояния и остового дерева; рождение перемещения точек в третьей фазе и построение квантово-качественной модели.

## Список литературы

- [1] Thomas Young. Double-slit experiment. https://en.wikipedia.org/wiki/Double-slit_experiment. 1801.
- [2] Фридрих Гегль. Наука Логики. 1812.
- [3] Фридрих Гегель. Наука Логики, Том второй : Субъективная логика, Книга третья: Учение о Понятии. 1812. 
- [4] Фридрих Энгельс. Диалектика Природы. 1873. 
- [5] Владимир Ильич Ленин. Материализм и Эмпириокритицизм. 1909. 
- [6] Владимир Ильич Ленин. Философские тетради. 1933. 
- [7] Мао Цзедун. Относительно противоречия. 1937.
- [8] Иосиф Сталин. Диалектический и Исторический Материализм. 1938. 
- [9] S. Lloyd and E. Forgy. k-means clustering. https://en.wikipedia.org/wiki/K-means_clustering. 1957. 
- [10] B.A. Galler and M.J. Fischer. Union-find structure (An improved equivalence algorithm). https://en.wikipedia.org/wiki/Disjoint-set_data_structure. 1964. doi: 10.1145/364099.364331. 
- [11] J. W. J. Williams. Heap data structure (Algorithm 232 - Heapsort). https://en.wikipedia.org/wiki/Heap_(data_structure). 1964. doi: 10.1145/512274.512284. 
- [12] J. L. Bentley. KD tree (Multidimensional binary search trees used for associative searching). https://en.wikipedia.org/wiki/K-d_tree. 1975. doi: 10.1145/361002.361007. 
- [13] M Ester et al. DBSCAN: A density-based algorithm for discovering clusters in large spatial databases with noise. https://en.wikipedia.org/wiki/DBSCAN. 1996. 
- [14] R. J. Campello, D. Moulavi, and J. Sander. “HDBSCAN: Density-based clustering based on hierarchical density estimates”. In: Pacific-Asia Conference on Knowledge Discovery and Data Mining (2013), pp. 160–172. 
- [15] C. Hennig. “What are the true clusters?” In: Pattern Recognition Letters 64 (2015), pp. 53–62.
- [16] Leland McInnes and John Healy. “Accelerated Hierarchical Density Based Clustering”. In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW) (Nov. 2017). doi: 10.1109/icdmw.2017.12. Url: http://dx.doi.org/10.1109/ICDMW.2017.12.
- [17] Павел Артамонов, перевод настоящей статьи на английский https://github.com/artamono1/druhg/blob/master/papers/druhg_en.pdf. 2023.
- [18] Minimum spanning tree structure. https://en.wikipedia.org/wiki/Minimum_spanning_tree.

## Приложение Псевдокод

Друг (DRUHG): Диалектический Рефлексивный Универсальный Группировщик. Алгоритм состоит из двух частей: формирование дерева и разметки кластеров.
Дерево хранится в структуре непересекающихся множеств (union-find)[10]. Соседи находятся через KD-дерево[12], получая k-ближайших-соседей (knn). Заметим, что повышая k не обязательно приводит к улучшению точности, но обязательно замедляет запуск.

```python
1: procedure Чистая взаимность(Points): ◃ Добавляет 1-к-1 отношения
2: 	для всех p ∈ Points:
3: 		если для самого ближнего соседа p самый ближний тогда:
4: 			Добавляем ребро 
5: 			Вес ребра = расстояние между точками
```

Хоть мы и ищем глобальный оптимум, в реальности поддеревья независимо растут из 1-к-1 чистых соединений. 

```python
6: procedure Строим Дерево(Points): ◃ Находим и добавляем глобальный min
7: 	Цикл
8: 		GlobalMinimum = INF
9: 		GM-Edge = Null
10: 		для всех p ∈ Points:
11:	 		для всех nn ∈  соседей p:			
12: 				если p и nn уже в дереве тогда:
13: 					далее
14: 				d = расстояние до nn из глаз p
15: 				если d > GlobalMinimum тогда:
16: 					выход
18: 				r = ранг nn из глаз p
19: 				R = ранг p из глаз nn
20: 				если r > R тогда:
21: 					далее
22: 				DialecticValue = min(D; d·R/r):
23: 					этот минимум учитывает обе точки
24: 					D это расстояние до R из глаз p
25: 				если DialecticValue < GlobalMinimum тогда:
26:  					GlobalMinimum = DialecticValue
27:  					GM-Edge = (p, nn, DialecticValue)
28:  		Добавить GM-Edge в Дерево
29:  	пока все рёбра не будут соединены
30:  	возврат Дерево: сохранён порядок присоединения рёбер и весов
```

Полный перебор в нахождении глобального оптимума даёт  O(n^2) сложность, которую можно значительно улучшить. Например, dialecticValue растёт монотонно по каждой точке, и внешний цикл может быть упрощён с помощью кучи(heap)[11] или с помощью разбиения всей системы на сектора.

```python
31: procedure Разметка (Дерево): ◃ Поддеревья могут стать кластерами
32: 	начальные Внутренности всехточек = 0
33: 	для всех рёбер(Пара поддеревьев, ВесРебра) в Дереве:
34: 		для Левого и Правого поддерева:
35: 			Граница = ВесРебра · ∑ 1/ki:
36: 				где ki = кол-во кластеров в поддереве
37: 			Внутренности = ∑ gi:
38: 				где gi = качество кластера в поддереве
39: 			если Граница > Внутренности тогда:
40: 				*Это поддерево есть кластер*
41: 				Его ki · gi = Граница
42: 		Слияние поддеревьев:
43: 			Сложение обратных размеров, качеств, количеств кластеров
```

Разметка очень быстра O(n).

### Перевод
Перевод на английский [17].