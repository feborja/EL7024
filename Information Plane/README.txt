Hecho por Ammi Beltrán, Fernanda Borja y Luciano Vidal

En esta carpeta se incluyen los archivos con los cuales se construyeron los information plane. A
continuación se detalla el contenido de cada archivo y como ejecutarlos.

MINE_VGG16.ipynb: este archivo lee los parámetros del modelo entrenado para clasificación y calcula
               I(X,T) y  I(Y,T) para unas capas definidas. Para realizar el calculo de la información mutua
               para distintas épocas se deben cargar los parámetros de la red entrenada en las distintas épocas.
               Este código fue diseñado para ejecutarse en google colab.

eachVGGParams_Clean_10.pt: parámetros de la red cuando se entrenó durante 10 épocas, se incluyen en esta carpeta   
                           por si se desea ejecutar el código  MINE_VGG16.ipynb.

X_IP_VGG.npy: valores de I(X,T) estimados para todas las capas analizadas en todas las épocas.

Y_IP_VGG.npy: valores de I(Y,T) estimados para todas las capas analizadas en todas las épocas.

info_plane_plots.py: genera los gráficos del information plane a partir de los valores estimados
                     para I(X,T) y I(Y,T).