# Предсказание срыва поставок
## Banana Overfit Capybaras

- содержит код для запуска приложения на основе streamlit (**st_main.py**)

Наше финальное решение оформлено в виде сайта на streamlit. Простой и понятный интерфейс для запуска моделей и интерпретаций их выводов, только самая необходимая информация для принятия решений по потенциально сорванным поставкам.

# Пояснения к приложению:
- После подгрузки данных появляется флажок "Использовать большую модель". По дефолту используется простая модель CatBoost, хоть и небольшая и быстрая, она выдает хороший скор и позволяет интерпретировать свой вывод. Под большой моделью подразумевается стекинг из нескольких моделей, который занимает гораздо больше времени. Такая модель немного точнее, зато позволяет взглянуть на интерпретацию выводов по разному, так как работает много моделей, а это может быть иногда полезно.
- Во время предсказаний на данные появится слайдер с разными инсайдами по тренировочным данным, мы решили добавить это, чтобы крутилась полезная информация, пока пользователь не может ничего делать. Некоторые инсайды могут быть интересны, так как влияют на распределение срывов поставок/ключевые параметры. Как только предсказания выполнятся слайдер пропадет, но если пользователю будет интересно почитать и посмотреть еще, то они будут внизу странички.
- Как только модель предскажет датасет должно появиться несколько полей:
    -  Текстовое поле, в которое пользователь может ввести интересующий срез данных. (есть инструкция на сайте)
    -  Таблица с лучшими/худшими агентами (поставщик/материал/итд - параметр, который может задать пользователь). (есть инструкция на сайте)
    -  Таблица с ожидаемыми потерями - произведение вероятности срыва на сумму. (есть инструкция на сайте)

    при желании таблицы можно выгрузить для дальнейшего использования пользователем.

- Дальше следует подгрузка графиков и таблиц с вкладом параметров. Из этих данных пользователь сможет понять вплоть до отдельных заказов, что повлияло на успех или срыв поставки. Здесь тоже приложены подробные инструкции как и что делать.

    ##### ВАЖНО: в работе с вкладом параметров (в особенности для больших моделей) 
    рекомендуется запустить вычисления на фон. Они могут занимать от 10 до 40 минут (для большой модели) и от 2 до 6 минут (для маленькой модели) в зависимости от используемого железа.
    **Расчет вклада параметров происходит так, что 1-ый запуск всегда самый долгий, а все последующие после него происходят моментально. Если пользователь захочет смотреть отдельные поставки или брать выборочные данные из датасета по условиям, то после 1-ого просчета он сможет очень быстро получить всю нужную информацию.**
    
    Для быстрого принятия решения и анализа зачастую должно хватит маленькой модели.

### Пояснения к директориям:
1. в js лежит код от слайдера
2. models разделена на 4 фолда, где лежат веса бустингов
3. stacking_model хранит веса моделей стекинга
4. ph содержит png графиков для слайдера
5. скачать энкодер по ссылке https://drive.google.com/file/d/1gJVTJVbAbjf2KzufBhvh3Hmt1LLUX2Jp/view?usp=sharing и загрузить его в папку с кодом 
6. скачать папку models по ссылке https://drive.google.com/drive/folders/133iUbi0rhEATBwkm8IEY0nQV0hMxw3e8?usp=sharing и загрузить ее в папку с кодом (если скачано с aiijc)
7. скачать папку stacking_model по ссылке https://drive.google.com/drive/folders/1Jc55A36QIL0RYVZFitQqUDybf8cQYO5Z?usp=sharing и загрузить ее в папку с кодом (если скачано с aiijc)

### Для запуска слайдера
1. pip install streamlit
2. в '.\python\Lib\site-packages\streamlit\static\' поместить js и ph папки
3. запуск производится через streamlit run st_main.py


