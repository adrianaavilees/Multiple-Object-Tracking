# Multiple-Object-Tracking
L'objectiu d'aquest repte és saber quants automobils es mouen en cada un dels carrils d'entrada / sortida. Això podria ser la solució al problema de cotrolar el fluxe de vehicles en una carretera o en grau d'ocupació d'un parking.


## DIARI DE DESENVOLUPAMENT:

14/10/2025:

    ✓ Configuració de l'entorn / Reader del video OPENCV
    ✓ Detecció de cotxes amb YOLOv8n (perque va molt més rapid i al no tenir GPU inclus funciona amb molta velocitat i continua sent precís)
    ✓ Seguiment de vehicles: assignat un ID per cada cotxe per seguir la seva trajectoria
        -> deepSORT: Millora les oclusions i canvis de forma. A més del filtre de Kalman (per al moviment), DeepSORT utilitza una xarxa neuronal de Re-identificació pre-entrenada per extreure un "vector de característiques" de l'aparença de cada vehicle detectat. Aquest vector és com una "empremta dactilar" visual.
    

TODO:
    - Comptatge per creuament de linea: definir linea (part de sota de la imatge) i comptar el cotxe quan passi per allí en una direcció (pujada o baixada)

17/10/2025

    ✓ Cambiar DEEPSORT per TRACKER SIMPLE AMB OPEN CV, perque anirà millor en temps de resposta i a temps real (recomanació del profe)
        1. Agafar les deteccions (bounding box) que ens dona el YOLO
        2. calcular centroides per cada rectangle
        3. associar IDS
    ✓ Definir una linea virtual: punt d'inici (x1, y1) i un punt final (x2, y2). Mes o menys quasi al final de la imatge, on el pas de peatons. 
    ✓ Obtenir el punt central de cada vechicle per determinar si ha pasat per la linea. 
    ✓ Determinar la Direcció: Un cop detectat un creuament, mirarem si el moviment ha estat cap amunt o cap avall de la rampa
        --> El primer video da correcto. Up: 6. Down: 2

TODO:
    - Ejecutar con todos los videos y guardar los resultados en CSV y de los videos.
    - Validar los resultados con los del profe
    

## REFERÈNCIES

https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
https://pypi.org/project/deepsort/ 

Tracker:
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
