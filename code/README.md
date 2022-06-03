# Elenco dei codice scritto

## 0_clean_data_trips_pointv3.ipynb
Notebook utilizzato per la pulizia del dataset `trips_pointv3.parquet`.
Le operazioni svolte sono:
- rimuovere le coordinate errate data l'area di interesse (latitude e/o longitude minori o uguali a zero)
- sistemazione ID: la variabile *point_trip_id* si ripete spesso per corse differenti, in giorni differenti, viene pertanto prodotta una nuova variabile *unique_id* che riporta un identificativo univoco per ogni trip, data dalla concatenazione del precedente ID e dalla data della corsa.
- utilizzare un alphashape per filtrare le coordinate al di fuori dell'area di interesse (fuori dall'area delineata dal comune per l'utilizzo dei monopattini elettrici)
- rimuovere le corse anomale: un numero minimo di tratte sembra durare per molte ore (a volte anche più di 20). Ciò è probabilmente il risultato di un errore nella registrazione o di una concatenazione di corse differenti, pertanto questi dati "errati" sono stati rimossi.
Il notebook produce `trips_pointv3_cleaned.parquet`, salvato in `data`.

## 0_clean_data_tripsv3.ipynb
Notebook utilizzato per la pulizia del dataset `tripsv3.parquet`.
- rimuovere le coordinate errate data l'area di interesse (latitude e/o longitude minori o uguali a zero)
- sistemazione ID: la variabile *trip_id* si ripete spesso per corse differenti, viene pertanto prodotta una nuova variabile *unique_id* che riporta un identificativo univoco per ogni trip, seguendo la stessa procedura del notebook precendente.
- utilizzare un alphashape per filtrare le coordinate al di fuori dell'area di interesse (fuori dall'area delineata dal comune per l'utilizzo dei monopattini elettrici).
- rimuovere le corse anomale (e.g., corse che durano per molte ore).
Il notebook produce `tripsv3_cleaned.parquet`, salvato in `data`.

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

## dataviz_spostamenti_monopattini.ipynb 
Partendo da quanto presente nell'articolo [Stop detection in GPS tracks — Movingpandas & KeplerGl](https://towardsdatascience.com/stop-detection-in-gps-tracks-movingpandas-keplergl-point-map-with-stops-duration-in-bird-664064b3ccbc), produce l'animazione presente in `dataviz_spostamenti_2022.html`, dove è possibile visualizzare, in base al periodo, le corse avvenute nei primi mesi del 2022, distinte per operatore e con la presenza dei vari punti di "stop". Le traiettorie sono invece visualizzate in `dataviz_traiettorie_2022.html` con spessore e colore differente in base alla velocità, calcolata mediante MovingPandas.

## costs_trips.ipynb 
Calcola i costi delle varie corse presenti in `trips_pointv3_cleaned.parquet` secondo le regole vigenti.

## vehicles_usage.ipynb (to do)
Analizza, attraverso l'identificativo unico del veicolo, l'utilizzo dei vari e-scooters. In particolare, elabora periodi e tempi di utilizzo e visualizza i periodi di lunga inattività di alcuni e-scooters attraverso l'utilizzo dello space time cube.

## MapMatching_ValhallaDocker.ipynb
Effettua il map matching delle corse utilizzando il Docker di Valhalla. Produce `mapmatching-example.html`(un esempio di una trip dopo il map matching) e `map-matching.html` (tutte le trips per cui è stato fatto il map matching, in una data giornata).
Inoltre, salva identificativi e routes risultanti dal map-matching in un nuovo dataframe (`map_matched_routes.parquet`).

## ztl-p.ipynb
Controlla se i monopattini passano attraverso le zone ZTL-P e quelle a circolazione interdetta, utilizzando prima i dati originali e poi quelli risultanti dal map-matching. Ci si aspetta di trovare pochi percorsi passanti attraverso queste aree vietate al transito dei monopattini, tuttavia i risultati appaiono differenti da quelli attesi.

## tratte_forti.ipynb & interactive_tratte_forti.ipynb
Producono alcune visualizzazioni per osservare le tratte più percorse in base all'arco di tempo selezionato.
Il secondo notebook rende interattiva una delle mappe permettendo di selezionare il mese oppure la fascia oraria da visualizzare.

## cfr_autobus.ipynb
Compara inzio e fine corsa dei viaggi in e-scooter con le fermate del trasporto pubblico per verificare se esiste un autobus che, in quella fascia oraria e periodo dell'anno, avrebbe potuto fungere da mezzo di trasporto alternativo. Per far ciò utilizza i files seguenti lo standard GTFS forniti dalla Provincia (`data/google_transit_urbano_tte.zip`).