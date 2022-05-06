# Elenco dei codice scritto

## ExploreTrips.ipynb
Analisi esplorativa dei dati contenuti in `trips_pointv3.parquet`; contiene:
- calcolo della durata media di un viaggio 
- plot del numero di viaggi per giorno
- plot del numero di viaggi per giorno della settimana
- plot dei viaggi per ora del giorno 
- plot dei viaggi per ora per ogni operatore 
- plot dell'uso mensile
- plot del tempo cumulativo dell'uso di e-scooter al giorno

## traiettorie.ipynb
Visualizzazione delle traiettorie in `trips_pointv3.parquet` e inizio a comparare punti di fermata; contiene:
- Plot dei punti visitati durante le corse in e-scooter, con gradazione di colore diversa in base alla frequenza con cui quel luogo appare nei dati
- Plot delle traiettorie di un giorno random con layer control per selezionare le tratte, ordinate per lunghezza del percorso
- Individuare i punti di fermata più comuni, servendosi di Nominatim per ottenere il nome del luogo. Plot barchart con le fermate che appaiono più di 50 volte, riportando il nome del luogo; mappa con le fermate che appaiono più di 50 volte, con markers di colori diversi in base alla frequenza
- Uso di pyrosm per ottenere i dati della posizione delle fermate del trasporto pubblico. Plot di questi punti e dei più frequenti punti di fine corsa degli e-scooter per controllare se le corse dei monopattini si concludono nelle vicinanze delle fermate del trasporto pubblico. In caso affermativo, questo suggerirebbe che gli e-scooters potrebbero essere stati usati come mezzo per raggiungere un altro trasporto e non solo per divertimento. Infatti l'uso di e-scooter per tratte molto brevi suggerisce che potrebbero essere stati usati solo per il divertimento della corsa e in questo contesto non sarebbero propriamente una risorsa per la mobilità sostenibile.

## traiettorie_con_MovingPandas.ipynb
Riproduce le traiettorie del precendente notebook usando MovingPandas, che ha il vantaggio di eliminare i "punti di rumore" presenti nei dati. 
Utilizza inoltre varie funzioni di generalizzazione, tra cui l'algoritmo Douglas-Peucker, studiandone la tolleranza.

## monopattini_streets.ipynb
Estrae i percorsi dai monopattini (da verificare meglio) ed applica il map matching.

## dataviz_spostamenti_monopattini.ipynb (to fix)
Riproduce quanto presente nell'articolo [Stop detection in GPS tracks — Movingpandas & KeplerGl](https://towardsdatascience.com/stop-detection-in-gps-tracks-movingpandas-keplergl-point-map-with-stops-duration-in-bird-664064b3ccbc). 