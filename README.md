# EL7024
Information Plane on Corrupted Training

Hecho por Ammi Beltrán, Fernanda Borja y Luciano Vidal

**Para la ejecución de la sección de audio, es necesario tener previamente descargado el dataset URBANSOUNDS8K**
Este repositorio posee todos los bloques de codigo necesarios para entrenar redes, corromper datos y trabajar con el plano de información desde los cuadernos de Jupyter, el resto de archivos corresponde librerías e imágenes
***
* audio_analysis: Permite visualizar las corrupciones de audio
* audio_training: Permite entrenar la red de audio y guardar los parámetros, la corrupción se debe cambiar manualmente en audio/audio_data.py en base a las corrupciones presentes en audio/audio_corrupts.py además de que se debe indicar la dirección del dataset de forma manual
* audio_plotter: Grafica curvas de entrenamiento
* VGG_training: contiene todo el pipeline de entrenamiento con imágenes, utiliza las corrupciones presentes en img_corrupts.py
* audio_correlation: contiene el código que permite realizar las estimaciones de informacion mutua en los datos de audio. Los resultados de estas estimaciones son guardados en el directorio /audio_mutualinf/
* img_mutual_information: contiene el código que permite realizar las estimaciones de informacion mutua en el dataset de imagenes. Los resultados de estas estimaciones son guardados en el directorio /img_mutualinf/
* mutual_inf_plots: utiliza las estimaciones guardadar en /audio_mutualinf/ y /img_mutualinf/ para generar los information planes

***
Los parámetros de audio se encuentran [aquí](https://drive.google.com/drive/folders/1Vf6-zDiAvkhi2rkxzDpHAdZQUmnGg6oc?usp=sharing).

Los parámetros de imagenes se encuentran [aquí](https://drive.google.com/drive/folders/1Z-6cn6ZPKIJFL83lxCQgFRx05dqf7jSU?usp=sharing) para descargar, en donde los parámetros para realizar los plots se encuentran en el zip VGGstate_params.zip

