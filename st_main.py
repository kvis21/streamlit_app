import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import torch
import itertools
from io import BytesIO
import xgboost as xgb
import pickle
import joblib
import json
import base64
import shap
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 485px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)


html_code='''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=1"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/Swiper/4.5.1/css/swiper.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Swiper/4.5.1/js/swiper.min.js"></script>
  <link rel="stylesheet" href="/js/slider.css">
</head>
<body>
<div class="swiper swiper-container">

    <div class="image-wrapper swiper-wrapper">
        <div class="slide swiper-slide">
          <div class="im">
            <img src="/ph/image.png" alt="" class="i">
            <div class="text">Вероятность срыва поставки в "выбросах" (случаи, сильно отличающихся от основной массы данных) такая же, как и в основной массе данных.</div>
          </div>
        </div>
        <div class="slide swiper-slide">
          <div class="im">
            <img src="/ph/image1.png" alt="" class="i">
            <div class="text">В крупных уникальных цепочках поставок (уникальная совокупность материала, поставщика, менеджеров итд...) срывы поставок встречаются реже</div>    
          </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image2.png" alt="" class="i">
                <div class="text">Количество обработчиков заказа почти не меняется с момента заведения заказа, при кол-ве обработчиков<11 может присоединиться максимум 2-3 обработчика, при этом в заказах с большим кол-вом обработчиков (>11) сильно возрастает кол-во срывов поставок.</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image3.png" alt="" class="i">
                <div class="text">В случаях, когда поставка не пришла раньше назначенного срока, у операционных менеджеров 1, 2, 11 и 16 в основном происходят срывы поставок.</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image4.png" alt="" class="i">
                <div class="text">Если заказ приходит раньше планируемого поступления, то в 98% случаев он не будет сорван.</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image5.png" alt="" class="i">
                <div class="text">В случаях, когда поставка не пришла раньше назначенного срока, группы закупок 8 (топ 3 по величине) и 3 (топ 8 по величине), имеют значительное количество срывов поставок</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image6.png" alt="" class="i">
                <div class="text">В случаях, когда поставка не пришла раньше назначенного срока, чем чаще происходит отмена полного деблокирования, тем выше шанс на срыв поставки</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image7.png" alt="" class="i">
                <div class="text">В случаях, когда поставка не пришла раньше назначенного срока, если 1-ое согласование заказа проходит долго или, в особенности, 1-ое согласование прошло быстро, а 2-ое наоборот затянулось, то это может быть свидетельством возможного срыва поставки</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image8.png" alt="" class="i">
                <div class="text">Дата поставки в основном согласуется за 1-ю неделю, а позже может измениться буквально пару раз , при этом в случаях, когда поставка не пришла раньше назначенного срока, чем больше корректировок в дате поставки, тем меньше наблюдается срывов поставок</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image9.png" alt="" class="i">
                <div class="text">Чем меньше частота нахождения заказа на 1 стадии, тем больше длительность заказа</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image10.png" alt="" class="i">
                <div class="text">Если была высокая частота нахождения заказа на 2 стадии согласования, и финальное решение было принято в тот же день, то есть высокий шанс на срыв поставки</div>
            </div>
        </div>
        <div class="slide swiper-slide">
            <div class="im">
                <img src="/ph/image11.png" alt="" class="i">
                <div class="text">В случае если кол-во дней между созданием заявки и создание позиции меньше 50 и при этом позиция на заказ создавалась меньше 25 дней, то наблюдается большой рост случаев срывов по сравнению со случаями где позиция на заказ создавалась больше 25 дней</div>
            </div>
        </div>
    </div>

    <div class="swiper-pagination"></div>
  
    <!-- If we need navigation buttons -->
    <div class="prev swiper-button-prev">

    </div>
    <div class="next swiper-button-next">

    </div>
  
    <!-- If we need scrollbar -->
    <div class="swiper-scrollbar"></div>
    <script src="/js/slider.js"></script>
</div>
</body>'''
help_to_expected_loss='''<style>
.block_exp{position:relative;
        width: 300px;
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        }
.hidden_exp{display: none;
        position:absolute;
        width: 300px;
        background: #F0F0F3;
        outline: 2px solid #000;
        border-radius: 5px ;
        z-index: 1;
        right:25px}
.focus_exp{width: 100px;
	    height: 100px; 
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        position:relative;
        left: 0px
        text-decoration: none
        }
.focus_exp:hover{text-decoration: none;
             }
.focus_exp:active{text-decoration: none;
             color: #000;
             }

a.focus_exp{position:relative;
        left:0px}
.focus_exp:focus + .hidden_exp{display: block;
                        right:}

</style>

<div class="block_exp"> <!-- контейнер -->
  <a href="#" class="focus_exp">Инструкция к ожидаемым потерям:</a> <!-- видимый элемент -->
  <span class="hidden_exp" >Данная таблица показывает ожидаемые потери по каждой поставке.
Ожидаемая потеря - произведение суммы поставки на вероятность ее срыва, под таблицей показана суммарная ожидаемая потеря. В случае, если вы хотите провести подробный анализ, таблицу можно скачать в формате .xlsx
</span> <!-- скрытый элемент -->
</div>'''
help_to_text_input='''<style>
.block_text{position:relative;
        width: 500px;
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        right: 50px;
        z-index:1
        }
.hidden_text{display: none;
        position: absolute;
        width: 375px;
        right:150px;
        background: #F0F0F3;
        outline: 2px solid #000;
        border-radius: 5px ;
        z-index: 2}
.focus_text{width: 100px;
	    height: 100px; 
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        position:relative;
        left: 0px
        text-decoration: none
        }
.focus_text:hover{text-decoration: none;

             }
.focus_text:active{text-decoration: none;
             color: #000;
             }
             
a.focus_text{position:relative;
        left:50px}
.focus_text:focus + .hidden_text{display: block;
                        right:}

</style>

<div class="block_text"> <!-- контейнер -->
  <a href="#" class="focus_text">Инструкция к текстовому полю</a> <!-- видимый элемент -->
  <span class="hidden_text" >С помощью этого поля вы можете выбрать интересующую вас часть заказов
Для выбора интересующей вас части данных нужно вводить текст правильным форматом:
<p>-Оставьте поле пустым для вывода статистики по всему датасету</p>
<p>-Для вывода статистики для определенных заказов введите ID через ; без пробелов (пример: 1;2;3;4;5)</p>
<p>-Если вам нужно выбрать кусок данных по условию, то вводите интересующие вас колонки, условия =,>,<,>=,<= и значения без пробелов(пример: Материал=5;Сумма<=6;Согласование заказа 1>2)</p>
</span> <!-- скрытый элемент -->
</div>'''
help_to_best = '''<style>
.block_best{position:relative;
        width: 400px;
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        }
.hidden_best{display: none;
        position:absolute;
        width: 450px;
        background: #F0F0F3;
        outline: 2px solid #000;
        border-radius: 5px ;
        z-index: 1;
        right:25px}
.focus_best{width: 100px;
	    height: 100px; 
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        position:relative;
        left: 0px
        text-decoration: none
        }
.focus_best:hover{text-decoration: none;
             }
.focus_best:active{text-decoration: none;
             color: #000;
             }

a.focus_best{position:relative;
        left:0px}
.focus_best:focus + .hidden_best{display: block;
                        right:}

</style>

<div class="block_best"> <!-- контейнер -->
  <a href="#" class="focus_best">Инструкция к таблице лучшим/худшим</a> <!-- видимый элемент -->
  <span class="hidden_best" >Эта таблица показывает лучших и худших поставщиков/материалы и т.д по доле сорванных поставок (1 - все поставки сорваны, 0 - все поставки доставлены). 
Через всплывающий список вы можете выбрать интересующий параметр для просмотра лучших/худших агентов в нем. В случае, если вы хотите провести подробный анализ, таблицу можно скачать в формате .xlsx
</span> <!-- скрытый элемент -->
</div>'''
help_to_shap = '''<style>
.block_shap{position:relative;
        width: 450px;
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        }
.hidden_shap{display: none;
        position:absolute;
        width: 450px;
        background: #F0F0F3;
        outline: 2px solid #000;
        border-radius: 5px ;
        z-index: 1;
        right:25px}
.focus_shap{width: 100px;
	    height: 100px; 
        background: #fff; /* Цвет фона */
        outline: 0 px solid #EEE; /* Чёрная рамка */
        border: 20 px solid #0f0; /* Белая рамка */
        border-radius: 5px ;
        color: #000;
        position:relative;
        left: 0px
        text-decoration: none
        }
.focus_shap:hover{text-decoration: none;
             }
.focus_shap:active{text-decoration: none;
             color: #000;
             }

a.focus_shap{position:relative;
               }
.focus_shap:focus + .hidden_shap{display: block;
                        right:}

</style>

<div class="block_shap"> <!-- контейнер -->
  <a href="#" class="focus_shap">Инструкция к вкладу параметров</a> <!-- видимый элемент -->
  <span class="hidden_shap" >Данные графики показывают вклад каждого параметра в срыв поставки.
На левом графике сразу видно в целом как разные значения влияют на предсказание. Отрицательный вклад означает, что модель посчитала данный параметр как понижающий риск срыва. Положительный означает повышение риска срыва соответственно. Цвет отвечает за величину значения.
На правом графике можно подробнее рассмотреть отдельные поставки и также вклад каждого параметра в повышение/понижение риска срыва поставки. Для подробного рассмотрения одного отдельного случая можно ввести в текстовое поле справа ID поставки и график покажет его отдельно.
</span> <!-- скрытый элемент -->
</div>
'''
info='''
<html lang="en">
<head>
  <style>
  .carousel-inner > .item > img,
  .carousel-inner > .item > a > img {
    width: 100%;
    margin: auto;
  }
  a{text-decoration: none
  }
  </style>
</head>
<body>
<h4>О команде:</h4>
<p> -богатый опыт участия в разных соревнованиях и хакатонах, победы на международных соревнованиях
<a href="https://www.kaggle.com/competitions/icr-identify-age-related-conditions">Kaggle (Топ - 2)</a>
и 
<a href="https://2022.hacks-ai.ru/">на всероссийских чемпионатах (ЦП)</a>.
В ходе работы над данной задачей был проведен широчайший анализ данных и работа над интерпретацией, отобраны самые полезные и важные для принятия решения выводы по данным.
</p>
<h4>Контакты:</h4>
<p>-Вяхирев Иван: i.viakhirev@mail.ru</p>
<p>-Панченко Антон: toni.panchenko77@inbox.ru
</p>
</body>
</html>
'''
help_to_switch='''
Большая модель занимает больше ресурсов и времени выполнения, но дает более подробную информацию по предсказаниям
'''

use_cols=['Поставщик', 'Операционный менеджер', 'Завод', 'Закупочная организация',
       'Группа закупок', 'Балансовая единица', 'Длительность', 'До поставки',
       'Месяц2', 'День недели 2', 'Количество позиций',
       'Количество обработчиков 30', 'Согласование заказа 1',
       'Изменение даты поставки 30',
       'Изменение позиции заказа на закупку: изменение даты поставки на бумаге',
       'Изменение позиции заказа на закупку: дата поставки',
       'Количество циклов согласования',
       'Количество изменений после согласований', 'Дней между 0_1',
       'ДлитlowerДо',
       'Поставщик_Материал_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2']#pd.read_csv(r'train1314.csv', ).drop('y', axis=1).columns

print(use_cols)
uploaded_file = st.sidebar.file_uploader("Choose a file")

cont_side=st.sidebar.container()
side_col1, side_col2, side_col3 = cont_side.columns([1.2, 0.95, 0.95])

st.sidebar.markdown(info, unsafe_allow_html=True)

#@st.cache_data#(suppress_st_warning=True)
def preprocessig(df) -> pd.DataFrame:
    percent = 0
    my_bar=cont_side.progress(percent, text='Выполняется...')


    highly_correlated = ['Количество обработчиков 15', 'Количество обработчиков 7', 'Изменение даты поставки 7',
                         'Изменение даты поставки 15', 'Отмена полного деблокирования заказа на закупку',
                         'Согласование заказа 2']
    categorical = ['Поставщик', 'Материал', 'Категорийный менеджер', 'Операционный менеджер', 'Завод',
                   'Закупочная организация',
                   'Группа закупок', 'Группа материалов', 'Вариант поставки', 'Балансовая единица', 'ЕИ', 'Месяц1',
                   'Месяц2', 'Месяц3', 'День недели 2']
    df = df.drop(columns=highly_correlated)

    df['ДлитlowerДо'] = df['До поставки'] < df['Длительность']  # 0.01 improvement
    df['Op1'] = df['Операционный менеджер'] == 1  # 0.001 improvement
    my_bar.progress(percent := percent + 15, f'{percent}%  создание признаков!')

    combinations = []
    df[categorical] = df[categorical].astype(str)
    for i in range(13, 15):
        comb = list(itertools.combinations(categorical, i))
        if len(comb) > 0:
            combinations += comb

    to_LE = []

    for feats in list(map(lambda x: x.split('_'), not_inter)):
        df['_'.join(feats)] = df[feats].apply(lambda x: "_".join(x), axis=1)
        to_LE.append('_'.join(feats))
        my_bar.progress(percent := percent + 5, f'{percent}%  создание комбинаций...')

    my_bar.progress(percent := percent + 20, f'{percent}%  загрузка энкодера...')
    with open(r'le_enc_2.pkl', 'rb') as f:
        OrdEnc = pickle.load(f)
    my_bar.progress(percent := percent + 15, f'{percent}%  трансформация категорий...')

    df[to_LE] = OrdEnc.transform(df[to_LE])
    st.session_state['Сумма']=df['Сумма']
    df = df[use_cols]

    my_bar.empty()
    cont_side.empty().success('выполнено!!!')
    categorical = ['Поставщик',
                   'Операционный менеджер',
                   'Завод',
                   'Закупочная организация',
                   'Группа закупок',
                   'Балансовая единица',
                   'Месяц2',
                   'День недели 2',
                   'Поставщик_Материал_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3',
                   'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц2_Месяц3_День недели 2',
                   'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3_День недели 2',
                   'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
                   'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
                   'Поставщик_Категорийный менеджер_Операционный менеджер_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2']
    df[categorical] = df[categorical].astype(int)
    return df

def models(make_pred: bool, large_model: bool, device)->None:
    pred_cv = []

    model_names = ['_cb_0', '_cb_10', '_cb_42', '_cb_7', '_xgb_42']

    if make_pred:
        if large_model:
            percent=0
            my_bar=cont_side.progress(percent, '...')
            models = [[] for i in model_names]
            meta_models=[]
            meta_data=[]
            xgb_pred=np.zeros(len(df))
            for fold, path in enumerate(os.listdir('./models/')):
                temp=df.copy()
                for i, model_path in enumerate(os.listdir(f'./models/{path}/')):
                    my_bar.progress(percent:= percent+3, f'{percent}%: {model_names[i]}, fold: {fold}')
                    if model_path[-4:] == '.pkl':
                        with open(f'./models/{path}/{model_path}', 'rb') as f:
                            model = joblib.load(f)
                    elif model_path[-5:] == '.json':
                        model = xgb.Booster()
                        model.load_model(f"./models/{path}/{model_path}")
                    if model_path[:2] == 'cb':
                        if fold % 2 == 1:
                            temp[model_names[i]] = model.predict_proba(df, task_type=device)[:, 1]
                            models[i].append(model)
                    elif model_path[:3] == 'xgb':
                        temp[model_names[i]] = model.predict(xgb.DMatrix(df))
                        if fold % 2 == 1:
                            xgb_pred+=model.predict(xgb.DMatrix(df))
                            models[i].append(model)
                my_bar.progress(percent := percent + 10, f'{percent}%: meta_model_{fold}')
                with open(f'./stacking_model/cb_meta_{fold}.pkl', 'rb') as f:
                    stack = joblib.load(f)
                if fold%2==1:
                    meta_models.append(stack)
                    meta_data.append(temp)
                    pred_cv.append(stack.predict_proba(temp, task_type=device)[:, 1])
            my_bar.empty()
            cont_side.success('выполнено!!!')
            st.session_state['meta_models'] = meta_models
            st.session_state['meta_data'] = meta_data
            st.session_state['xgb_pred'] = (xgb_pred/4).round()
        else:
            with open(f'./models/fold_0/cb_42_0.pkl', 'rb') as f:
                model = joblib.load(f)
            #model = Tabr('./streamlit/models/fold_0/best_0.pt')

            #model = xgb.Booster()
            #model.load_model(f'./streamlit/models/fold_0/xgb_42_0.json')
            #pred_cv = model.predict(xgb.DMatrix(df))
            #pred_cv = model.predict_proba(df)
            pred_cv = model.predict_proba(df, task_type=device)[:,1]
            st.session_state['pred_proba'] = pred_cv
            st.session_state['pred'] = pred_cv.round()
            st.session_state['models'] = [model]
            return 0
        st.session_state['pred_proba'] = np.array(pred_cv).mean(axis=0)
        pred_cv = np.array(pred_cv).mean(axis=0).round()
        st.session_state['pred'] = pred_cv
        st.session_state['models'] = models

def best_worst(df1, col) -> (pd.DataFrame, pd.DataFrame):
    df = df1.copy()

    df.loc[df['value'] >= 0.5, 'value'] = 1
    df.loc[df['value'] < 0.5, 'value'] = 0

    df['comp_losses'] = df.groupby(col)['value'].transform('sum') / df.groupby(col)['value'].transform('count')

    df = df.drop_duplicates(subset=[col])

    best = df.sort_values(by='comp_losses', ascending=True)
    worst = df.sort_values(by='comp_losses', ascending=False)

    return best[[col, 'comp_losses']], worst[[col, 'comp_losses']]

def define_df(inp: str, df1) -> pd.DataFrame:
    df = df1.copy()

    if len(inp) < 1:
        return df
    try:

        geterr = int(inp[0])

        ids = list(map(lambda x: int(x), inp.split(';')))
        df = df[df.index.isin(ids)]

        return df

    except:

        ifs = inp.split(';')

        if len(ifs) == 0:
            ifs = [inp]
        try:
            for cond in ifs:

                if len(cond.split('>=')) > 1:
                    df = df[df[cond.split('>=')[0]] >= int(cond.split('>=')[1])]
                elif len(cond.split('<=')) > 1:
                    df = df[df[cond.split('<=')[0]] <= int(cond.split('<=')[1])]
                elif len(cond.split('=')) > 1:
                    df = df[df[cond.split('=')[0]] == int(cond.split('=')[1])]
                elif len(cond.split('>')) > 1:
                    df = df[df[cond.split('>')[0]] > int(cond.split('>')[1])]
                elif len(cond.split('<')) > 1:
                    df = df[df[cond.split('<')[0]] < int(cond.split('<')[1])]
                else:
                    col1.error(f'{cond} нет знака')
                    return pd.DataFrame()

            return df

        except KeyError:
            col1.error(f'{cond} неправильное название параметра')
            return pd.DataFrame()
        except ValueError:
            col1.error(f'{cond} значение должно быть числом')
            return pd.DataFrame()

def expected_loss(df1) -> (pd.DataFrame, float):
    df = df1.copy()

    df['expected_loss'] = df['value'] * df['Сумма']
    return df[['value', 'Сумма', 'expected_loss']].sort_values(by='expected_loss', ascending=False), sum(
        df['expected_loss'])

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data
@st.cache_data
def convert_df(df, t='csv'):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if t == 'csv':
        return df.to_csv().encode('utf-8')
    if t=='excel':
        return to_excel(df)
device= 'GPU' if torch.cuda.is_available() else 'cpu'
not_inter=['Поставщик_Материал_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Завод_Закупочная организация_Группа закупок_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2',
       'Поставщик_Категорийный менеджер_Операционный менеджер_Закупочная организация_Группа закупок_Группа материалов_Вариант поставки_Балансовая единица_ЕИ_Месяц1_Месяц2_Месяц3_День недели 2']
def visualize_dataset_predictions_summary(features_data: pd.DataFrame, shap_values, tab, i, cont):

    features_data['Уникальная цепочка поставки']=features_data[not_inter].sum(axis=1)
    features_data=features_data.drop(columns=not_inter)
    features_data=features_data.rename(columns={'Изменение позиции заказа на закупку: изменение даты поставки на бумаге': 'изменение даты поставки на бумаге',
                                        'Изменение позиции заказа на закупку: дата поставки':'дата поставки',})

    shap_values.T[20] = shap_values.T[20:26].sum(axis=0)
    shap_values=np.delete(shap_values, list(range(21, 26)), axis=1)

    data=pd.DataFrame(shap_values, columns=features_data.columns)

    cont.download_button('Скачать .csv', file_name='pred.csv',
                                  data=convert_df(data), key=f'csv_{i}'
                                  )
    cont.download_button('Скачать .xlsx', file_name='pred.xlsx',
                           data=convert_df(data), key=f'xlsx_{i}'
                           )
    components.html(paint)
    fig= plt.figure()
    shap.summary_plot(shap_values=shap_values,
                      features=features_data,
                      feature_names=list(features_data.columns),
                      plot_size=(15, 10),
                      )
    ax_list = fig.axes  # see https://stackoverflow.com/a/24107230/11148296
    ax = ax_list[0]
    ax.set_xlabel('Вклад параметра в срыв поставки', fontsize=14)
    ax.set_ylabel('Параметры', fontsize=14)
    tab.pyplot()
    return shap_values

def visualize_dataset_predictions_decision(features_data: pd.DataFrame, shap_values, exp_val, tab, i):
    try:
        features_data['Уникальная цепочка поставки']=features_data[not_inter].sum(axis=1)
    except ValueError:
        features_data=pd.DataFrame(features_data).transpose()
        features_data['Уникальная цепочка поставки'] = features_data[not_inter].sum(axis=1)
    features_data=features_data.drop(columns=not_inter)
    features_data=features_data.rename(columns={'Изменение позиции заказа на закупку: изменение даты поставки на бумаге': 'изменение даты поставки на бумаге',
                                        'Изменение позиции заказа на закупку: дата поставки':'дата поставки',})

    shap_values.T[20] = shap_values.T[20:26].sum(axis=0)
    shap_values = np.delete(shap_values, list(range(21, 26)), axis=1)

    fig= plt.figure(figsize=(15, 10))
    shap.decision_plot(base_value=exp_val,
                        shap_values=shap_values,
                        features=features_data,
                        feature_names=list(features_data.columns),
                        ignore_warnings=True,
                      )
    ax_list = fig.axes  # see https://stackoverflow.com/a/24107230/11148296
    ax = ax_list[0]
    ax.set_xlabel('Вклад параметра в срыв поставки', fontsize=14)
    ax.set_ylabel('Параметры', fontsize=14)
    tab.pyplot()
    return shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.session_state['df_'] = df
    if 'df' not in st.session_state:
        df = preprocessig(df)
        st.session_state['df']=df
    df_=st.session_state['df_']
    df=st.session_state['df']
    large_model = side_col1.toggle('Использовать большую модель?', disabled=False, help = help_to_switch)
    make_pred = side_col2.button('Предсказать', disabled =False)
    if 'pred' not in st.session_state or make_pred:
        if 'el_lst' in st.session_state:
            el_lst=st.session_state['el_lst']
            for i in range(len(el_lst)):
                el_lst[i].empty()

        if 'to_None' in st.session_state:
            to_None=st.session_state['to_None']
            for i, el in enumerate(to_None):
                to_None[i]=None
        loading_screen=components.html(html_code, height=800)

    models(make_pred, large_model, device)

    if 'pred' in st.session_state:
        try:
            loading_screen.empty()
        except:
            bin=1
        pred=st.session_state['pred']
        pred_proba=st.session_state['pred_proba']
        side_col3.download_button('Скачать предсказания', file_name='pred.csv',
                                  data=convert_df(
                                      pd.concat([df_, pd.Series(pred, name='value')], axis=1))
                                  )
        text_area, intro  = st.columns([4, 1])
        intro.markdown(help_to_text_input, unsafe_allow_html=True)
        holder=text_area.empty()
        text=holder.text_input('Текстовое поле', key='text')

        col1, col2, col3 = st.columns([1.1, 0.65, 1.25])
        col3_1, col3_2 = col3.columns([1.1, 1])

        df_=pd.concat([df_, pd.Series(pred_proba, name='value')], axis=1)
        filtred_df=define_df(text, df_)

        col1_1, col1_2=col1.columns([1, 1])
        col2.markdown(help_to_best, unsafe_allow_html=True)
        cont1_1 = col1_1.container()
        cont1_2 = col1_2.container()
        el_lst=[]
        to_None=[]
        if not filtred_df.empty:
            holder_col1_1=col1_1.empty()
            col = holder_col1_1.selectbox('Выберите колонку:', filtred_df.columns, label_visibility='collapsed')
            best_df, worst_df = best_worst(filtred_df, col)
            el1 = cont1_1.dataframe(best_df, hide_index=False, use_container_width=True, column_config={col: st.column_config.TextColumn(col, width='small')})
            el2 = cont1_2.dataframe(worst_df, hide_index=False, use_container_width=True, column_config={col: st.column_config.TextColumn(col, width='small')})

            dw1=col1_2.download_button('Скачать таблицу .csv', file_name='best_df.csv', data=convert_df(df_, 'csv'), key='dw1')
            dw4=col1_2.download_button('Скачать таблицу .xlsx', file_name='best_df.xlsx', data=convert_df(df_, 'excel'))

            df_expected_loss, expected_loss = expected_loss(filtred_df)

            el4=col3_1.dataframe(df_expected_loss, hide_index=False)
            col3_1_, col3_2_ = col3.columns([0.5, 1])
            text2 = col3_1_.write(f'Ожидаемые потери: {expected_loss:.7}')
            dw2 = col3_2_.download_button('Скачать таблицу .csv', file_name='expected_df.csv',data=convert_df(df_expected_loss, 'csv'))
            dw3 = col3_2_.download_button('Скачать таблицу .xlsx', file_name='expected_df.xlsx', data=convert_df(df_expected_loss, 'excel'))

            el_lst.extend([el1, el2, el4])
            to_None.extend([dw1, dw2, dw3, dw4, col])
        paint='''<script>
                                const elements = window.parent.document.querySelectorAll('.stDownloadButton > button')
                                for (let i = 1; i < elements.length; i+=2) elements[i].style.backgroundColor = 'lightgray'; 
                                for (let i = 2; i < elements.length; i+=2) elements[i].style.backgroundColor = 'lightgreen'
                                
                                </script>'''
        components.html(paint)
        el5=col3_2.markdown(help_to_expected_loss, unsafe_allow_html=True)
        st.session_state['el_lst'] = el_lst+[el5]
        st.session_state['to_None'] = to_None+[text]
        shap_values_list=[]
        col__1, col__2, col__3 = st.columns([0.2, 0.8, 1])
        col__2.markdown(help_to_shap, unsafe_allow_html=True)
        gr=col__1.button('Посчитать вклад параметров')
        t = col__3.text_input(f'Введите индекс', help='', key=f'text_-1')

        df = df.loc[filtred_df.index]
        idx=df.index
        if t == '':
            t=df.index
        try:
            t=int(t)
            if t not in idx:
                col__3.error("Данного индекса нет в данных")
        except ValueError:
            col__3.error('Введите индекс')
        except TypeError:
            pass

        tmp_df=st.session_state['df']
        if gr:
            with st.spinner('Загрузка графиков...'):

                col_graph_1, col_graph_2 = st.columns([1, 1])
                models=st.session_state['models']

                if len(models)==1:
                    components.html(html_code, height=800)
                    if 'single_shap' not in st.session_state:
                        explainer = shap.TreeExplainer(models[0])
                        shap_values = explainer.shap_values(tmp_df)
                        print(shap_values.shape)
                        exp_val=explainer.expected_value
                        st.session_state['single_shap']=shap_values
                        st.session_state['single_exp']=exp_val
                    else:
                        shap_values=st.session_state['single_shap']
                        exp_val=st.session_state['single_exp']
                    visualize_dataset_predictions_summary(df.loc[idx], shap_values[idx], col_graph_1, -1, col__1)
                    if type(t)==int:
                        visualize_dataset_predictions_decision(df.loc[t], np.array([shap_values[t]]), exp_val, col_graph_2, 0)
                    else:
                        visualize_dataset_predictions_decision(df.loc[t], shap_values[t], exp_val, col_graph_2, 0)
                    components.html(paint)
                else:

                    meta_data=st.session_state['meta_data'].copy()
                    meta_models=st.session_state['meta_models'].copy()
                    shap_lst = []
                    if 'meta_shap' in st.session_state:
                        shap_values=st.session_state['meta_shap']
                        exp_val=st.session_state['meta_exp_val']
                    else:
                        exp_val=0
                        for i, model in enumerate(meta_models):

                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(meta_data[i])
                            exp_val+=explainer.expected_value
                            shap_lst.append(shap_values)
                        shap_values = np.array(shap_lst).mean(axis=0)
                        exp_val/=len(meta_models)
                        st.session_state['meta_shap']=shap_values
                        st.session_state['meta_exp_val']=exp_val
                    visualize_dataset_predictions_summary(meta_data[0].loc[idx], shap_values[idx], col_graph_1, 0, col__1)
                    if type(t)==int:
                        visualize_dataset_predictions_decision((meta_data[0].loc[t]), np.array([shap_values[t]]), exp_val, col_graph_2, 0)
                    else:
                        visualize_dataset_predictions_decision(meta_data[0].loc[t], shap_values[t], exp_val, col_graph_2, 0)

                    tabs = st.tabs(['_cb_0', '_cb_10', '_cb_42', '_cb_7', '_xgb_42'])
                    components.html(html_code, height=800)
                    for i, model in enumerate(models):
                        shap_lst=[]
                        if f'shap_values_{i}' not in st.session_state:
                            for mod in model:
                                exp_val=0
                                if i == 4:
                                    shap_values=mod.predict(xgb.DMatrix(df, enable_categorical=True), pred_contribs=True)
                                    exp_val+=shap_values[0, -1]
                                    shap_values=shap_values[:, :-1]

                                else:

                                    explainer = shap.TreeExplainer(mod)
                                    shap_values = explainer.shap_values(df.copy())

                                    exp_val+=explainer.expected_value
                                shap_lst.append(shap_values)
                            shap_values = np.array(shap_lst).mean(axis=0)
                            exp_val/=len(model)

                            st.session_state[f'exp_val_{i}'] = exp_val
                            st.session_state[f'shap_values_{i}'] = shap_values
                        else:
                            exp_val = st.session_state[f'exp_val_{i}']
                            shap_values = st.session_state[f'shap_values_{i}']
                        cont_tab=tabs[i].container()
                        cols_tab = tabs[i].columns([1, 1])
                        visualize_dataset_predictions_summary(df.copy().loc[idx], shap_values[idx], cols_tab[0], i+1, cont_tab)

                        if type(t)==int:
                            visualize_dataset_predictions_decision(df.copy().loc[t], np.array([shap_values[t]]), exp_val, cols_tab[1], i + 1)
                        else:
                            visualize_dataset_predictions_decision(df.copy().loc[t], shap_values[t], exp_val, cols_tab[1], i + 1)









