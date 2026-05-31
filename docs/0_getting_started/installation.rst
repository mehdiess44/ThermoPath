Installation et demarrage
=========================

Cette page decrit le chemin complet pour installer ThermoPath depuis GitHub,
lancer l'infrastructure MQTT/Streamlit et connecter la simulation Simulink.

Pre-requis
----------

Installez au minimum :

* Git ;
* Python 3.10 ou une version compatible avec les dependances du projet ;
* Docker Desktop avec Docker Compose ;
* MATLAB Simulink pour executer ``app/publish_mqtt.slx``.

Sous Windows avec WSL2, lancez Docker Desktop avant les commandes Compose et
verifiez que l'integration WSL est active si vous travaillez depuis une
distribution Linux.

Clonage du depot
----------------

Depuis le dossier de travail souhaite :

.. code-block:: powershell

   git clone https://github.com/mehdiess44/ThermoPath.git
   cd ThermoPath

Installation bare metal
-----------------------

Le mode bare metal est utile pour developper, deboguer le dashboard Streamlit
ou appeler directement le bridge Python depuis Simulink.

Creation et activation de l'environnement virtuel sous PowerShell :

.. code-block:: powershell

   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install --no-cache-dir -r requirements.txt

Activation equivalente sous ``cmd.exe`` :

.. code-block:: bat

   .venv\Scripts\activate.bat

L'option ``--no-cache-dir`` reproduit le comportement du ``Dockerfile`` et evite
des ecarts lies a un cache local de paquets Python.

Demarrage Docker
----------------

Pour lancer l'infrastructure complete :

.. code-block:: powershell

   docker compose up --build -d

Verification de l'etat :

.. code-block:: powershell

   docker compose ps
   docker compose logs -f mqtt_broker
   docker compose logs -f dashboard

Le dashboard conteneurise est disponible sur :

.. code-block:: text

   http://localhost:8501

Pour arreter la demonstration :

.. code-block:: powershell

   docker compose down

Demarrage Streamlit bare metal
------------------------------

En mode bare metal, Streamlit doit cibler le broker expose sur la machine hote.
Le port par defaut de l'application est ``1883`` ; il faut donc forcer le port
proxy ``1884`` si Mosquitto tourne dans Docker Compose :

.. code-block:: powershell

   $env:BROKER_HOST = "127.0.0.1"
   $env:BROKER_PORT = "1884"
   streamlit run app/app.py

Dans ``cmd.exe`` :

.. code-block:: bat

   set BROKER_HOST=127.0.0.1
   set BROKER_PORT=1884
   streamlit run app/app.py

Couplage Simulink
-----------------

Ouvrez ``app/publish_mqtt.slx`` dans MATLAB Simulink. Le bloc charge d'appeler
le systeme externe doit executer le bridge Python avec deux arguments :
temperature et force mecanique.

Commande de reference si le modele est execute depuis le dossier ``app`` :

.. code-block:: text

   python ../src/simulink_bridge.py <temp> <g_force>

Remplacez ``<temp>`` et ``<g_force>`` par les signaux ou variables Simulink
correspondants selon la configuration du bloc. Le script publie ensuite un JSON
sur ``thermopath/sensor`` via ``127.0.0.1:1884``.

Convention temporelle 60x
-------------------------

ThermoPath utilise la convention suivante :

.. code-block:: text

   1 seconde reelle = 1 minute simulee

Cette acceleration aligne la production temps reel sur les donnees
d'entrainement, preparees a une frequence d'une ligne par minute. Une fenetre
de ``5`` messages dans le dashboard correspond donc a ``5`` minutes simulees,
comme les features d'entrainement. Cette coherence evite de transformer une
derive thermique lente en variation artificiellement brutale, ou inversement.

