## Validierung des vortrainierten Netzwerks:
    
load_and_evaluate.py mit parameter --val_data='path' (path kann relativ zum Projektordner oder absolut sein) aufrufen <br>
default ist 'dataset_mmp/test'.<br>
Beispiel:

`load_and_evaluate.py --val_data='path/to/dataset'`

oder für validierung auf 'dataset_mmp/test':

`load_and_evaluate.py
`

Nach dem Ausführen wird ein result ordner mit der detection file detection_bbr.txt erstellt. Darin befindet sich auch
die MAP kurve.
## Basismodell trainieren:

in config.py den ersten parameter `base_model=False` auf True setzen und danach
main.py ausführen.<br>
(das konnten wir nicht per parameter übergeben da, die main.py ausgeführt werden soll und nicht von der config.py 
importiert werden darf)

## Performancemodell trainieren:

in config.py den ersten parameter `base_model=False` auf False setzen und danach
main.py ausführen

Hierfür muss sich das passende dataset im folgenden relativen ordner befinden:<br>
_../datasets/dataset_2_crowd_min_plus_mmp_dataset_train_<br>
Dieser parameter kann optional unten als letzter parameter in der config.py verändert werden:<br>
`train_dataset = "../datasets/dataset_2_crowd_min_plus_mmp_dataset_train"`

Das passende Dataset befindet sich auf dem server im ordner:

_/home/student/datasets/dataset_2_crowd_min_plus_mmp_dataset_train_

Das password für den server lautet: gZ4Tnmzpt73QcG46


Beispiel für eine consolen ausführung auf dem server:

`ssh -p 6102 student@seppel.informatik.uni-augsburg.de`<br>
`gZ4Tnmzpt73QcG46`<br>
`cd Project_Florian/`<br>
`nohup python3 main.py > log_train.txt 2>&1 &`<br>
`tail -f log_train.txt`<br>