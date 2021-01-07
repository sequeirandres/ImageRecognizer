# Entrenar una Red Neuronal para detectar y clasificar varios objetos usando TensorFlow (GPU) en Linux o Windos.

## Introduction
El propósito de este documento es explicar cómo entrenar su propio clasificador de detección de objetos de [red neuronal convolucional](https://www.juanbarrios.com/redes-neurales-convolucionales) para múltiples objetos, comenzando desde cero. Al final de este documento, tendrá un programa que puede identificar y dibujar cuadros alrededor de objetos específicos en imágenes(.jpg), videos(.mp4) o en una cámara web(streaming).
Este Documento está escrito para ubuntu 18.04 lts (64-bits). Tambien funcionará para Windows 10. El procedimiento general también se puede utilizar para los sistemas operativos Linux, pero las rutas de archivo y los comandos de instalación de paquetes deberán cambiar en consecuencia. Para el entrenamiento usé TensorFlow-GPU v1.5, pero es probable que funcione para futuras versiones de TensorFlow como  GPU-2.3 Cuda.
TensorFlow-GPU permite que la PC use la tarjeta de video para proporcionar potencia de procesamiento adicional durante el entrenamiento, por lo que se usará para este proposito. En mi experiencia, usar TensorFlow-GPU en lugar de TensorFlow-cpu regular reduce el tiempo de entrenamiento en un factor considerable, desde 10 a 100 veces mas rápido. La versión solo para CPU de TensorFlow también se puede usar para este instructivo, pero llevará más tiempo en el entrenamiendo de la red neuronal. El tiempo de penderá de la cantidad de imagenes que se utilice para entrenar y la cantidad de objetos que se desea clasificar. Si usa TensorFlow solo para CPU, no necesita instalar CUDA.

## objetivo
Se pevee reconocer 3 objetos dentro de una imagen(.jpeg), un video(.mp4) o desde la webcam(streaming). Los 3 objetos que se utilizaran para entrenar la red neruronal son
1. Modulo de arduino esp8266 
2. Modulo de arduino sensor ultrasonido (hcsr04)
3. placa programador (ch341) usb de memorias eeprom 24x/25x 

## resultados previstos
Se desea recuadrar sobre una imagen o video un rectangular que indica el objeto que la red neuronal detectó.
<p align="center">
  <img src="result/image_detected.jpg">
</p>


## Desarrollo

 En este apartado se describen los paquetes, programas y pasos a seguir para entrenar una red neuronal convolucionar y detectar los objetos aprendidos.

## Los pasos para crear, entrenar una red y hacer predicciones son las siguientes : 
1. [Instalar Anaconda, CUDA, and cuDNN](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
2. [Setting up the Object Detection directory structure and Anaconda Virtual Environment](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
3. [Gathering and labeling pictures](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures)
4. [Generating training data](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#4-generate-training-data)
5. [Creating a label map and configuring training](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#5-create-label-map-and-configure-training)
6. [Training](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#6-run-the-training)
7. [Exporting the inference graph](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#7-export-inference-graph)
8. [Testing and using your newly trained object detection classifier](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)

1. Instatalar los paquetes necesarios.
2. Generar el entorno (enviroment) para trabajar con la terminal. 
3. Utilizar un modelo pre-entrenado que permite reconocer objetos
4. Prepara los datos para entrenamiento.
4.1. Obtener las imagenes (.jpeg) para entrenar (train) y testear (test)
4.2. Instalar LabImag :
                4.2.1. paquetes requeridos  python 3.5    matplotlib
4.3. Generar los archivos CSV, para entrenar y para testear. 

## 1.a- Instalar los paquetes necesarios y crear un entorno de trabajo virtual.
Instalar [Anaconda](https://www.anaconda.com/products/individual) para ubuntu o windows
Instalar [python](https://www.python.org/downloads/) 

Una vez instalado crear un nuevo entorno de trabajo desde la terminal

```
$ conda create -n CNN pip python=3.5
```
activar el nuevo enviroment

```
$ activate CNN 
```

Instalar los siguientes paquetes necesarios 
```
(CNN) $ conda install -c anaconda protobuf
(CNN) $ pip install pillow
(CNN) $ pip install lxml
(CNN) $ pip install Cython
(CNN) $ pip install contextlib2
(CNN) $ pip install jupyter
(CNN) $ pip install matplotlib
(CNN) $ pip install pandas
(CNN) $ pip install opencv-python==3.6

(Nota:Los paquetes ‘pandas’ and ‘opencv-python’  no son necesarios para TensorFlow, pero son usados por los difernetes scripts en python para generar los archivos TFRecords y para trabajar con las imagenes, videos y webcam.
```

## 1b. Configurar los PYTHONPATH environment
```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```



