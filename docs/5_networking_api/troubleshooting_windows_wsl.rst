Troubleshooting Windows et WSL
==============================

Cette page regroupe les blocages les plus frequents au demarrage de ThermoPath
sur Windows, Docker Desktop et WSL2.

Silent drop avec localhost
--------------------------

Symptome typique : Simulink ou un script Python semble publier, mais le
dashboard ne recoit aucun message, ou la connexion MQTT echoue sans erreur
explicite dans l'interface.

Cause probable : selon la configuration Windows/WSL2, ``localhost`` peut etre
resolu en priorite vers ``::1`` en IPv6. Si le mapping Docker ou Mosquitto
ecoute seulement sur IPv4 pour ce chemin, la connexion n'atteint pas le broker.

Solution recommandee : utiliser strictement ``127.0.0.1`` pour les processus
executes sur la machine hote, notamment le bridge Simulink.

.. code-block:: python

   publish.single("thermopath/sensor", payload, hostname="127.0.0.1", port=1884)

Cette contrainte ne s'applique pas entre conteneurs Docker : dans Compose, le
dashboard doit utiliser ``mqtt_broker:1883``.

Conflit sur le port MQTT
------------------------

ThermoPath mappe le broker Docker sur le port hote ``1884`` afin de reduire les
conflits avec un service Mosquitto local deja installe sur ``1883``.

Pour verifier quel processus occupe un port sous PowerShell :

.. code-block:: powershell

   netstat -ano | findstr :1884
   netstat -ano | findstr :1883

La derniere colonne indique le PID. Pour identifier le processus :

.. code-block:: powershell

   Get-Process -Id <PID>

Pour l'arreter proprement depuis PowerShell si vous savez qu'il s'agit d'un
processus de test :

.. code-block:: powershell

   Stop-Process -Id <PID>

Evitez d'arreter un processus systeme ou un service d'entreprise sans verifier
son role.

Dashboard connecte mais aucune donnee
-------------------------------------

Verifiez les points suivants :

* ``docker compose ps`` montre ``mqtt_broker`` et ``dashboard`` en etat actif ;
* le dashboard conteneurise affiche ``Broker : mqtt_broker:1883`` ;
* le dashboard bare metal affiche ``Broker : 127.0.0.1:1884`` ;
* Simulink appelle bien ``python ../src/simulink_bridge.py <temp> <g_force>`` ;
* le bridge affiche un message de succes vers ``127.0.0.1:1884``.

Logs utiles
-----------

.. code-block:: powershell

   docker compose logs -f mqtt_broker
   docker compose logs -f dashboard

Si le dashboard tente de joindre ``localhost:1883`` alors qu'il tourne dans
Docker, les variables ``BROKER_HOST`` et ``BROKER_PORT`` du service Compose ne
sont pas appliquees comme prevu.

