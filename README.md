# Решение задачи №1 в хакатоне Норникеля от команды MSUBIGDATA

Видео с кратким описанием проекта: ./Поясняющее видео.webm или https://drive.google.com/file/d/1NBMRv68ZFYNutNhbAqmyUiW7aQgqbmxq/view

2 части: бэкенд и фронтенд (демонстрационный), которые общаются посредством json.

Бэкенд на Python, лежит в ./backend. Основной файл, который все запускает - backend/analize.py. Для его запуска необходимо указать путь до видео (73 строка, к примеру vid_path = '/Nornikel/dataset1-1/F1_1_4_2.ts'). Тогда после запуска analize.py сгенерирует dataanswer.json для общения с фронтендом.

Фронтенд. В ./frontend/build лежат 2 сбилженные версии, а в ./frontnend/front - исходники. Использует ReactJS, ChartJS. 
