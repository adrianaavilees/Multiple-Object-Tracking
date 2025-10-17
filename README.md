# Multiple-Object-Tracking
L'objectiu d'aquest repte és saber quants automobils es mouen en cada un dels carrils d'entrada / sortida. Això podria ser la solució al problema de cotrolar el fluxe de vehicles en una carretera o en grau d'ocupació d'un parking.







DIARI DE DESENVOLUPAMENT:

14/10/2025:
    ✓ Configuració de l'entorn / Reader del video OPENCV
    ✓ Detecció de cotxes amb YOLOv8n (perque va molt més rapid i al no tenir GPU inclus funciona amb molta velocitat i continua sent precís)
    ✓ Seguiment de vehicles: assignat un ID per cada cotxe per seguir la seva trajectoria
        -> deepSORT: Millora les oclusions i canvis de forma. A més del filtre de Kalman (per al moviment), DeepSORT utilitza una xarxa neuronal de Re-identificació pre-entrenada per extreure un "vector de característiques" de l'aparença de cada vehicle detectat. Aquest vector és com una "empremta dactilar" visual.
        
* Empezar haciendo un algoritmo que detecte los coches (Yolo) y hacer luego un tracker que permita a partir de un centroide (calculado o se lo pasamos), haga la predicción de dirección del objeto, etc.
* Si un objeto desaparece, no lo descarto directamente, evalúo los siguientes frames para comprobar si realmente ha desaparecido o no.
* A tener en cuenta: tiene que ser en tiempo real (30 fps/s)

TODO:
    - Comptatge per creuament de linea: definir linea (part de sota de la imatge) i comptar el cotxe quan passi per allí en una direcció (pujada o baixada)

17/10/2025
    ✓
    Definir una linea virtual: punt d'inici (x1, y1) i un punt final (x2, y2). Mes o menys quasi al final de la imatge, on el pas de peatons. 

    Obtenir el punt central de cada vechicle per determinar si ha pasat per la linea. 

    Guardar historial de posicions: Necessitem saber no només on està un cotxe ara, sinó també on estava en el fotograma anterior. Farem servir un diccionari per guardar l'última posició coneguda de cada ID. Perque els cotxes parats no volem tenir-los en compte, per molt que el seu punt central esitigui on la linea. 

    Determinar la Direcció: Un cop detectat un creuament, mirarem si el moviment ha estat cap amunt o cap avall de la rampa

    * mirar el simple tracker que le pasas un centroide y analiza a partir de eso, los trackers de OpenCV son más robustos pero captan toda la imagen
    * hacer contador a ser posible para numerar cuántos coches suben y cuantos bajan




REFERÈNCIES

https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
https://pypi.org/project/deepsort/ 
