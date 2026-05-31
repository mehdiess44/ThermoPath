# ThermoPath

ThermoPath est un MVP SaaS de surveillance predictive pour la chaine du froid
logistique. Le projet couple un modele MATLAB Simulink, un broker MQTT
Mosquitto, un moteur Python/Scikit-Learn et une interface Streamlit temps reel.

La version courante stabilise la boucle cyber-physique V1 avec :

- un broker MQTT isole dans Docker ;
- un dashboard Streamlit connecte en temps reel ;
- un pont Simulink -> Python -> MQTT ;
- un tampon glissant de 5 points protege par mutex cote dashboard ;
- une velocite thermique corrigee sur une periode ;
- une acceleration temporelle 60x pour aligner simulation et entrainement.

## 1. Architecture rapide

```text
MATLAB Simulink
    |
    | appel systeme: python ../src/simulink_bridge.py <temp> <g_force>
    v
src/simulink_bridge.py
    |
    | MQTT host: 127.0.0.1:1884
    v
Docker host port 1884
    |
    | proxy vers mqtt_broker:1883
    v
Mosquitto (thermopath_net)
    |
    | topic: thermopath/sensor
    v
Streamlit dashboard + Isolation Forest
```

En mode conteneurise, `docker-compose.yml` cree deux services :

- `mqtt_broker`, base sur `eclipse-mosquitto`, expose sur l'hote via
  `1884:1883` ;
- `dashboard`, construit depuis le `Dockerfile`, expose Streamlit sur
  `8501:8501` et se connecte au broker interne avec
  `BROKER_HOST=mqtt_broker` et `BROKER_PORT=1883`.

Le reseau Docker dedie `thermopath_net` permet au dashboard de joindre le broker
par son nom de service sans passer par les ports de l'hote.

## 2. Installation bare metal

### Prerequis

- Git
- Python 3.10 ou superieur
- Docker Desktop si vous utilisez le mode conteneurise
- MATLAB/Simulink pour executer `app/publish_mqtt.slx`

### Cloner le depot

```powershell
git clone https://github.com/mehdiess44/ThermoPath.git
cd ThermoPath
```

### Creer et activer l'environnement virtuel

PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

CMD :

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### Installer les dependances

Le Dockerfile installe les paquets sans cache. Vous pouvez reproduire ce
comportement localement :

```powershell
pip install --no-cache-dir -r requirements.txt
```

Dependances principales : Streamlit, Pandas, NumPy, Scikit-Learn, Plotly,
Joblib et Paho MQTT.

## 3. Demarrage Docker

Depuis la racine du projet :

```powershell
docker compose up --build -d
```

Verifier l'etat :

```powershell
docker compose ps
docker compose logs -f mqtt_broker
docker compose logs -f dashboard
```

Arreter l'infrastructure :

```powershell
docker compose down
```

Reconstruire proprement apres modification de dependances ou du Dockerfile :

```powershell
docker compose build --no-cache
docker compose up -d
```

Acces dashboard :

```text
http://localhost:8501
```

### Pourquoi le port 1884 ?

Le broker Mosquitto ecoute sur `1883` dans le reseau Docker, mais l'hote Windows
publie vers `1884`. Cette redirection `1884:1883` evite les conflits avec un
broker MQTT deja installe sur Windows/WSL2 et contourne certains blocages de
pare-feu ou de services locaux.

## 4. Couplage MATLAB Simulink

Le modele Simulink se trouve dans :

```text
app/publish_mqtt.slx
```

Ouvrez ce fichier dans MATLAB/Simulink, puis configurez le bloc d'appel systeme
pour appeler le bridge Python avec les deux signaux physiques :

```text
python ../src/simulink_bridge.py <temp> <g_force>
```

Dans le projet, `src/simulink_bridge.py` :

- lit deux arguments CLI : temperature et g-force ;
- construit un payload JSON `{"temp": ..., "g_force": ...}` ;
- publie sur le topic MQTT `thermopath/sensor` ;
- cible explicitement `127.0.0.1:1884`.

Exemple de test manuel depuis la racine du projet :

```powershell
python src/simulink_bridge.py -70.5 1.02
```

### Acceleration temporelle 60x

La simulation Simulink doit conserver le facteur d'acceleration 60x :

```text
1 seconde reelle = 1 minute simulee
```

Cette dilatation temporelle aligne la fenetre de production du dashboard
(`BUFFER_SIZE = 5`, donc 5 messages) avec la fenetre de 5 minutes utilisee lors
de l'entrainement. Sans ce facteur, les features temporelles, notamment la
velocite thermique, ne representent plus la meme physique que celle apprise par
le modele.

## 5. Lancement Streamlit

### Mode conteneurise recommande

```powershell
docker compose up --build -d
```

Puis ouvrir :

```text
http://localhost:8501
```

Dans ce mode, le dashboard utilise automatiquement :

```text
BROKER_HOST=mqtt_broker
BROKER_PORT=1883
```

### Mode bare metal

Si vous lancez Streamlit hors Docker, le dashboard lit par defaut :

```text
BROKER_HOST=localhost
BROKER_PORT=1883
```

Pour viser le broker Docker depuis l'hote Windows, utilisez le port proxy
`1884` :

```powershell
$env:BROKER_HOST="127.0.0.1"
$env:BROKER_PORT="1884"
streamlit run app/app.py
```

En CMD :

```cmd
set BROKER_HOST=127.0.0.1
set BROKER_PORT=1884
streamlit run app/app.py
```

## 6. Tests MQTT rapides

Les scripts de test `src/test_publisher.py` et `src/test_subscriber.py`
utilisent actuellement `localhost:1883`. Ils sont adaptes a un broker local
bare metal. Si vous testez contre le broker Docker depuis Windows, utilisez
plutot `src/simulink_bridge.py`, deja configure sur `127.0.0.1:1884`, ou
ajustez temporairement ces scripts vers le port `1884`.

Test avec le bridge :

```powershell
python src/simulink_bridge.py -70.5 1.02
python src/simulink_bridge.py -70.0 1.03
python src/simulink_bridge.py -69.5 1.04
python src/simulink_bridge.py -69.0 1.05
python src/simulink_bridge.py -68.5 1.06
```

Le dashboard commence l'inference apres 5 messages, car son tampon glissant
contient `BUFFER_SIZE = 5` points.

## 7. Troubleshooting Windows / WSL2

### Silent drop MQTT avec localhost

Sur certaines configurations Windows/WSL2, `localhost` peut etre resolu en IPv6
vers `::1`. Si le broker ou le proxy Docker n'ecoute pas correctement sur IPv6,
les publications MQTT peuvent sembler partir sans etre recues.

Solution : ciblez explicitement IPv4.

```text
127.0.0.1
```

C'est pour cette raison que `src/simulink_bridge.py` publie vers
`127.0.0.1:1884` au lieu de `localhost`.

### Conflit de port MQTT

Verifier les ports occupes :

```powershell
netstat -ano | findstr :1883
netstat -ano | findstr :1884
netstat -ano | findstr :8501
```

Identifier le processus :

```powershell
tasklist /FI "PID eq <PID>"
```

Arreter le processus si necessaire :

```powershell
taskkill /PID <PID> /F
```

Si `1883` est deja occupe par un Mosquitto local, ce n'est pas bloquant pour le
mode Docker tant que `1884` reste libre sur l'hote.

### Dashboard connecte mais aucune prediction

Verifiez les points suivants :

- le dashboard affiche le bon broker en bas de page ;
- le broker Docker est actif avec `docker compose ps` ;
- au moins 5 messages ont ete publies sur `thermopath/sensor` ;
- les fichiers `models/model.pkl` et `models/scaler.pkl` existent ;
- Simulink appelle bien `python ../src/simulink_bridge.py <temp> <g_force>`.

### Rebuild apres modification Python

Si vous modifiez le code Python et utilisez Docker :

```powershell
docker compose up --build -d
```

Puis rechargez `http://localhost:8501`.

## 8. Structure du projet

```text
app/
  app.py                 Dashboard Streamlit temps reel
  publish_mqtt.slx       Modele MATLAB Simulink
src/
  simulink_bridge.py     Pont Simulink -> MQTT
  realtime_engine.py     Moteur console de prediction temps reel
  test_publisher.py      Publisher MQTT de test local
  test_subscriber.py     Subscriber MQTT de test local
models/
  model.pkl              Modele Isolation Forest
  scaler.pkl             Scaler d'entrainement
mosquitto/
  mosquitto.conf         Configuration Mosquitto
docker-compose.yml       Broker + dashboard + reseau thermopath_net
Dockerfile               Image Streamlit
requirements.txt         Dependances Python
```
