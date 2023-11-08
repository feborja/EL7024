# EL7024
Information Plane on Corrupted Training

Hecho por Ammi Beltrán, Fernanda Borja y Luciano Vidal

**Para la ejecución de la seeción de audio, es necesario tener previamente descargado el dataset URBANSOUNDS8K**
Este repositorio posee todos los bloques de codigo necesarios para entrenar redes, corromper datos y trabajar con el plano de información desde los cuadernos de Jupyter, el resto de archivos corresponde a parámetros, librerías e imágenes
***
* audio_analysis: Permite visualizar las corrupciones de audio
* audio_training: Permite entrenar la red de audio y guardar los parámetros, la corrupción se debe cambiar manualmente en audio/audio_data.py en base a las corrupciones presentes en audio/audio_corrupts.py además de que se debe indicar la dirección del dataset de forma manual
* audio_plotter: Grafica curvas de entrenamiento
* VGG_training: contiene todo el pipeline de entrenamiento con imágenes, utiliza las corrupciones presentes en img_corrupts.py

***
Los parámetros de audio se encuentran en audio_params de forma comprimida, de ser necesario se deben descomprimir
Los parámetros de imagenes se encuentran \href{aquí}{https://drive.google.com/drive/folders/1Z-6cn6ZPKIJFL83lxCQgFRx05dqf7jSU?usp=sharing} para descargar
