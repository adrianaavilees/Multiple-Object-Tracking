# Multiple-Object-Tracking
L'objectiu d'aquest repte és saber quants automobils es mouen en cada un dels carrils d'entrada / sortida. Això podria ser la solució al problema de cotrolar el fluxe de vehicles en una carretera o en grau d'ocupació d'un parking.







DIARI DE DESENVOLUPAMENT:

14/10/2024:
    ✓ Configuració de l'entorn / Reader del video OPENCV
    ✓ Detecció de cotxes amb YOLOv8n (perque va molt més rapid i al no tenir GPU inclus funciona amb molta velocitat i continua sent precís)
    ✓ Seguiment de vehicles: assignat un ID per cada cotxe per seguir la seva trajectoria
        -> deepSORT: Millora les oclusions i canvis de forma. A més del filtre de Kalman (per al moviment), DeepSORT utilitza una xarxa neuronal de Re-identificació pre-entrenada per extreure un "vector de característiques" de l'aparença de cada vehicle detectat. Aquest vector és com una "empremta dactilar" visual.

TODO:
    - Comptatge per creuament de linea: definir linea (part de sota de la imatge) i comptar el cotxe quan passi per allí en una direcció (pujada o baixada)




REFERÈNCIES

https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
https://pypi.org/project/deepsort/ 
