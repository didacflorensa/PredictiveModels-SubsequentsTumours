# PredictiveModel-SubsequentsTumours
Predictive models using machine learning algorithms to calculate the risk of suffer a secondary primary tumour

## About

* **Author**: Didac Florensa Cazorla <didac.florensa@udl.cat>
* **PH.D supervisors**:
  * Jordi Mateo Fornés <jordi.mateo@udl.cat>, @github/JordiMateoUdL
  * Francesc Solsona Tehàs <francesc.solsona@udl.cat>
  * Pere Godoy Garcia <pere.godoy@gencat.cat>
* **Collaborators**:
  * Ramon Piñol <rpinol@catsalut.cat>,
  * Miquel Mesas <mmesas@gss.cat>
  * Tere Pedrol <mtpedrol.lleida.ics@gencat.cat>

## Background

Previous works have shown that risk factors for secondary primaries cancers depend on people’s lifestyle (e.g. alcohol and smoking consumption). This repository contains the code implemented to build a predictive model to calculate the risk of a secondary primary cancer.

## This repository

This repository contains all the code, scripts and services built in the context of my Ph.D Studies related with secondary primary cancer in the region of Lleida. Inside this repository you will find:

* _data_: This folder contains mock data to use and understand the tools developed. This is not the data used in my research, I can not publish raw data for logical reasons.
* _docker_: This folder contains the required container to deploy the scripts. Alternatively, you can install R, python or jupyter or shinny in your personal laptop or server and run them in a more traditional way.
* _results_: This folder is need by the containers to store inside the results. To add the results in this folder, create it manually. This folder will not upload in github because it was added in .gitignore.
* _python-scripts_: This folder contains the scripts in python to clean and transform data.


## Running python scripts

First of all, you need to build the custom image to deploy the container in docker environment:

```sh
chmod +x create-python-image.sh
./create-python-image.sh
```

Execute the container that will act as a docker-server, for example if we want to clean data:

```sh
docker run --rm  python-computing-service python parse_extraction.py
```
