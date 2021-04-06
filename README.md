## Подготовка данных
<p> Чистые данные с речью были сформированы из набора данных Russian Open Speech To Text <a href = 'https://azure.microsoft.com/en-us/services/open-datasets/catalog/open-speech-to-text/#AzureNotebooks>[source]</a> - это большой открытый корпуса устной русской речи, который содержит в себе 20 тысяч часов устной речи, из разных предметных областей – звонки, youtube, лекции, телефонные разговоры, книги. </p>
<p>Для разработки своего набора данных был взят как раз данных корпус, но только из аудиокниг. Так как аудио здесь разной длительности от менее секунды до 17 минут, то была написана функция по нарезки аудио и сохранения ее в папку. Не брались файлы менее, чем 2 секунды. Нарезка производилась по 2 секунды.</p>
<p>Описание аудио данных с чистой речью можно посмотреть <a href='manifest_clean_signal_with_frequency (2).csv'>здесь</a></p>
<p>В ходе проверки, было предположение, что мужских голосов в наборе данных больше, чем женских. Был взят и оптимизирован готовый классификатор, для определения пола говорящего <a href = 'https://www.youtube.com/watch?v=fodf4Pttve4&t=3624s&ab_channel=DeepLearningSchool'> [source]</a> <a href ='prepare_clean_and_noises_audio/classificator.ipynb'>(код можно посмотреть здесь)</a> и доработан. Гипотеза подтвердилась.</p>
<p>Далее был произведен поиск шумов. Было сформулировано 200 типов шумов, где на каждый тип было найдено по 5 аудиофайлов. Аудио были найдены в <a href = 'https://zvukogram.com'> [source]</a>.</p> 
<p>Все эти шумы можно выделить в такие классы: аудиофайлы в виде хлопка, хруста, шипения, жевания, бурления, отрывания, кляцания, скрежета, короткого шума, бьющиеся предметы, скрип, шумы, связанные с водой – капание, береговые, шуршание, музыкальные, стук, шумы, издаваемые людьми – кашель, смех, сморкание, лепет, шумы от техники, звонкие шумы – от сирен, будильников, шумы, которые относятся к месту (окружающая среда) – в стадионе, с кафе, ресторане и т.д.</p>
<p>Итоговое описание шумов можно посмотреть <a href='manifest_noises_with_frequences (2).csv'>здесь</a></p>
<p>Для аугментации – наложения шума на аудиозаписи был разработан конвейер при помощи библиотеки Apache Beam, об этом описано в отдельной части.</p>
  <p>Описание зашумленных данных с речью можно посмотреть <a href='noises_manifest (3).csv'>здесь</a></p>
<p>После аугментации был проведен анализ полученного набора данных. Его результаты представлены <a href = 'https://github.com/Alena0704/intelligent-noise-reduction/blob/analize_audio/analize%20audio/analize_dataset.ipynb'>здесь</a></p>
<p>Сами данные храняться на моем google drive диске и ссылки на них приведены ниже:<p>
  <ul>
  <li><a href = 'https://drive.google.com/drive/folders/1CVhGtImdSdV8VUXZjNQ-4GusS-Vdq3SW?usp=sharing'>Ссылка на чистые данные с речью</a></li>
  <li><a href = 'https://drive.google.com/drive/folders/14f-yg-YtWLuo65ZLpvJvl6ZPq6XMJLdP?usp=sharing'>Ссылка на шумы</a></li>
  <li><a href = 'https://drive.google.com/drive/folders/17WgVXXtG3f6WQjmC-aIQGF2r609gt73U?usp=sharing'>Ссылка на зашумленные файлы</a></li>
  </ul>
